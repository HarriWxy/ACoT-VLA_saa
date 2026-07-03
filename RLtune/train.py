"""RL fine-tuning training script for ACoT-VLA.

Implements GRPO (Group Relative Policy Optimization) for flow-matching VLA models.
Adapted from SimpleVLA-RL's approach, but using JAX/Flax and advantage-weighted
flow matching loss instead of PPO log-prob ratio clipping.

Training loop:
    for epoch in range(total_epochs):
        1. Collect rollouts: run policy in environment, get trajectories
        2. Compute rewards: binary success/failure from environment
        3. Filter by accuracy: only train on medium-difficulty tasks
        4. Compute advantages: GRPO group normalization
        5. Update policy: weighted flow matching loss on demonstration data
        6. Log metrics and save checkpoints

Usage:
    # From the project root:
    python -m RLtune.train --config_name srb_train --checkpoint_dir path/to/sft_ckpt

    # Or using the launch script:
    bash RLtune/run_rl_train.sh
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import pathlib
import platform
import sys
import time
from typing import Any

# Ensure project root is in sys.path so 'RLtune' can be imported
# regardless of whether we run as `python -m RLtune.train`
# or `python RLtune/train.py`.
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from RLtune.env_runner import EnvRunner
from RLtune.env_runner import compute_rollout_metrics
from RLtune.grpo_algo import compute_advantages
from RLtune.grpo_algo import filter_by_accuracy
from RLtune.reward_manager import create_reward_manager
from RLtune.rl_config import GRPOConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def init_wandb(
    rl_config: GRPOConfig,
    base_config: _config.TrainConfig,
    *,
    resuming: bool,
    enabled: bool = True,
):
    """Initialize WandB logging for RL training."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = base_config.checkpoint_dir
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=f"{base_config.project_name}_rl")
    else:
        wandb.init(
            name=f"rl_grpo_{base_config.exp_name}",
            config={
                **dataclasses.asdict(rl_config),
                "base_config": dataclasses.asdict(base_config),
            },
            project=f"{base_config.project_name}_rl",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


# ---------------------------------------------------------------------------
# Model initialization (reused from SFT training)
# ---------------------------------------------------------------------------


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    """Initialize training state from SFT config.

    This reuses the SFT model initialization but prepares for RL training.
    """
    # Use a lower learning rate for RL fine-tuning
    rl_lr = 5e-6
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=rl_lr,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-4,
        ),
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ---------------------------------------------------------------------------
# RL Training Step
# ---------------------------------------------------------------------------


@at.typecheck
def rl_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    advantages: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """RL training step using advantage-weighted flow matching loss.

    Unlike SFT where we minimize the standard flow matching loss uniformly,
    here we weight each sample's loss by its GRPO advantage:
        L_rl = mean(A_i * L_fm_i)

    This encourages the model to:
    - Better fit trajectories that led to success (positive advantage).
    - Move away from trajectories that led to failure (negative advantage).

    Args:
        config: Training configuration.
        rng: Random number generator.
        state: Current training state.
        batch: (Observation, Actions) tuple.
        advantages: Per-sample advantages from GRPO.

    Returns:
        new_state: Updated training state.
        info: Training metrics dictionary.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        adv: jnp.ndarray,
    ):
        # Compute per-sample flow matching loss
        per_sample_loss = model.compute_loss(rng, observation, actions, train=True)
        # per_sample_loss shape: [batch_size, action_horizon] or [batch_size]

        # Average over action horizon if needed
        if per_sample_loss.ndim > 1:
            per_sample_loss = jnp.mean(per_sample_loss, axis=-1)

        # Weight by advantages
        clipped_adv = jnp.clip(adv, -3.0, 3.0)
        rl_loss = jnp.mean(clipped_adv * per_sample_loss)

        return rl_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, advantages)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Compute metrics
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "rl_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "mean_advantage": jnp.mean(advantages),
    }
    return new_state, info


@at.typecheck
def rl_train_step_acot(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
    advantages: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """RL training step for ACoT-VLA model (with coarse actions).

    Same as rl_train_step but handles the additional coarse_actions input
    required by the ACoT-VLA dual-expert architecture.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        adv: jnp.ndarray,
    ):
        per_sample_loss = model.compute_loss(rng, observation, actions, coarse_actions, train=True)
        if per_sample_loss.ndim > 1:
            per_sample_loss = jnp.mean(per_sample_loss, axis=-1)
        clipped_adv = jnp.clip(adv, -3.0, 3.0)
        rl_loss = jnp.mean(clipped_adv * per_sample_loss)
        return rl_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, coarse_actions, advantages
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "rl_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "mean_advantage": jnp.mean(advantages),
    }
    return new_state, info


# ---------------------------------------------------------------------------
# Offline RL Training Step (using demonstration data weighted by advantages)
# ---------------------------------------------------------------------------


@at.typecheck
def offline_rl_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    advantages: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Offline RL training step using demonstration data weighted by GRPO advantages.

    This is the primary training mode for ACoT-VLA RL fine-tuning:

    1. We have a dataset of demonstrations (from SFT data).
    2. We run rollouts to evaluate the current policy (getting success/failure).
    3. We compute GRPO advantages based on rollout results.
    4. We weight the standard flow matching loss on demonstrations by advantages.

    This approach:
    - Doesn't require on-policy data collection (can use existing SFT data).
    - Uses the environment evaluation to assign credit to different demonstrations.
    - Gradually shifts the distribution toward successful behaviors.

    For on-policy training, use rl_train_step instead.
    """
    return rl_train_step(config, rng, state, batch, advantages)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    env_runner: EnvRunner,
    policy_host: str,
    policy_port: int,
    num_episodes: int = 20,
    prompt: str = "collect the sample",
) -> dict[str, float]:
    """Evaluate the current policy by running episodes.

    Args:
        env_runner: Environment runner instance.
        policy_host: Policy server host.
        policy_port: Policy server port.
        num_episodes: Number of evaluation episodes.
        prompt: Task prompt.

    Returns:
        Dictionary of evaluation metrics.
    """
    trajectories = env_runner.collect_rollouts(
        policy_host=policy_host,
        policy_port=policy_port,
        n_episodes=num_episodes,
        prompt=prompt,
    )
    return compute_rollout_metrics(trajectories)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(rl_config: GRPOConfig):
    """Main RL training loop.

    Orchestrates the full GRPO training pipeline:
    1. Initialize model from SFT checkpoint
    2. Start policy server for rollouts
    3. For each epoch:
       a. Collect rollouts in environment
       b. Compute rewards and advantages
       c. Optionally filter by accuracy
       d. Update policy with advantage-weighted loss
       e. Log metrics and save checkpoints
    """
    init_logging()
    logging.info(f"Running RL fine-tuning on: {platform.node()}")
    logging.info(f"RL Config: {dataclasses.asdict(rl_config)}")

    # Get base training config
    base_config = _config.get_config(rl_config.config_name)

    # Override checkpoint dir if specified
    if rl_config.checkpoint_dir:
        base_config = dataclasses.replace(
            base_config,
            checkpoint_dir=pathlib.Path(rl_config.checkpoint_dir),
        )

    # Initialize JAX mesh
    rng = jax.random.key(rl_config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(base_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize checkpoint manager
    rl_ckpt_dir = base_config.checkpoint_dir / "rl_grpo"
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        rl_ckpt_dir,
        keep_period=rl_config.save_interval,
        overwrite=True,
        resume=True,
    )

    # Initialize wandb
    init_wandb(rl_config, base_config, resuming=resuming)

    # Initialize model
    train_state, train_state_sharding = init_train_state(base_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, None)

    # Select appropriate train step based on model type
    if base_config.model.model_type in (_model.ModelType.ACOT_VLA_PI05, _model.ModelType.ACOT_VLA_PI0):
        ptrain_step = jax.jit(
            functools.partial(rl_train_step_acot, base_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(rl_train_step, base_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    # Initialize components
    reward_fn = create_reward_manager(rl_config)
    env_runner = EnvRunner(rl_config)

    # Create data loader for offline RL (using SFT demonstration data)
    data_loader = _data_loader.create_data_loader(
        base_config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)

    # ── Main training loop ──
    logging.info(f"Starting RL training for {rl_config.total_epochs} epochs")
    best_success_rate = 0.0

    for epoch in range(rl_config.total_epochs):
        epoch_start = time.time()
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch + 1}/{rl_config.total_epochs}")
        logging.info(f"{'='*60}")

        # ── Phase 1: Collect rollouts ──
        logging.info("Phase 1: Collecting rollouts...")
        trajectories = env_runner.collect_rollouts()

        # Compute rollout metrics
        rollout_metrics = compute_rollout_metrics(trajectories)
        logging.info(f"Rollout metrics: {rollout_metrics}")

        # ── Phase 2: Compute rewards and advantages ──
        logging.info("Phase 2: Computing rewards and advantages...")
        rewards, reward_metrics = reward_fn(trajectories)
        prompt_indices = np.array([t.prompt_index for t in trajectories])

        # ── Phase 3: Filter by accuracy (optional) ──
        if rl_config.filter_by_accuracy:
            raw_rewards = np.array([t.reward for t in trajectories])
            mask, filter_metrics = filter_by_accuracy(
                raw_rewards,
                prompt_indices,
                lower_bound=rl_config.accuracy_lower_bound,
                upper_bound=rl_config.accuracy_upper_bound,
            )
            logging.info(f"Filter metrics: {filter_metrics}")

            if mask.sum() == 0:
                logging.warning("All samples filtered out! Skipping this epoch.")
                continue

            # Filter trajectories and rewards
            filtered_indices = np.where(mask)[0]
            filtered_rewards = rewards[filtered_indices]
            filtered_prompt_indices = prompt_indices[filtered_indices]
        else:
            filtered_rewards = rewards
            filtered_prompt_indices = prompt_indices

        # ── Phase 4: Compute GRPO advantages ──
        logging.info("Phase 3: Computing GRPO advantages...")
        advantages = compute_advantages(
            filtered_rewards,
            filtered_prompt_indices,
            estimator=rl_config.adv_estimator,
            n_samples=rl_config.n_samples,
            gamma=rl_config.gamma,
        )

        # ── Phase 5: Policy update ──
        logging.info("Phase 4: Updating policy...")
        step_infos = []

        for step in range(rl_config.num_train_steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # Convert advantages to JAX array and broadcast to batch size
            adv_jax = jnp.array(advantages[: len(batch[0].state)])
            if len(adv_jax) < len(batch[0].state):
                # Repeat advantages if batch is larger
                adv_jax = jnp.tile(adv_jax, (len(batch[0].state) // len(adv_jax) + 1,))[: len(batch[0].state)]

            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch, adv_jax)

            step_infos.append(info)

            if step % rl_config.log_interval == 0:
                stacked = jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *step_infos)
                reduced = jax.device_get(stacked)
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
                logging.info(f"  Step {step}: {info_str}")
                step_infos = []

        # ── Logging ──
        epoch_time = time.time() - epoch_start
        all_metrics = {
            **rollout_metrics,
            **reward_metrics,
            "epoch": epoch + 1,
            "epoch_time": epoch_time,
        }

        # Add advantage statistics
        all_metrics["advantage_mean"] = float(np.mean(advantages))
        all_metrics["advantage_std"] = float(np.std(advantages))

        wandb.log(all_metrics, step=epoch)
        logging.info(f"Epoch {epoch + 1} summary: {all_metrics}")

        # ── Save checkpoint ──
        if (epoch + 1) % rl_config.save_interval == 0:
            logging.info(f"Saving checkpoint at epoch {epoch + 1}...")
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, epoch)

        # ── Track best model ──
        if rollout_metrics["success_rate"] > best_success_rate:
            best_success_rate = rollout_metrics["success_rate"]
            logging.info(f"New best success rate: {best_success_rate:.4f}")

    # Final save
    logging.info("Training complete. Saving final checkpoint...")
    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, rl_config.total_epochs)
    checkpoint_manager.wait_until_finished()

    wandb.log({"best_success_rate": best_success_rate})
    logging.info(f"Best success rate achieved: {best_success_rate:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tyro

    rl_config = tyro.cli(GRPOConfig)
    main(rl_config)

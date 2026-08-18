"""JAX offline RL training for ACoT-VLA using pre-collected demonstration data.

This script mirrors the logic in RLtune/train.py but replaces online rollout
collection with offline episode sampling from the standard LeRobot data pipeline.
It is intended as a lightweight JAX version of the PyTorch offline GRPO trainer.

Usage:
    python -m RLtune.train_offline_jax --config_name srb_train_tracking
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import logging
import os
import pathlib
import platform
import sys
import time
from typing import Literal

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency fallback
    SummaryWriter = None

# Ensure project root is in sys.path.
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import RLtune.data_loader_rl as _data_loader
from RLtune.grpo_algo import compute_advantages
from RLtune.grpo_algo import filter_by_accuracy
from RLtune.train import init_logging
from RLtune.train import init_train_state
from RLtune.train import offline_rl_train_step

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Disable GPU usage for JAX offline RL training.

@dataclasses.dataclass(frozen=True)
class OfflineJAXConfig:
    """Minimal configuration for JAX offline RL training."""

    n_samples: int = 4
    clip_ratio_high: float = 0.28
    clip_ratio_low: float = 0.2
    gamma: float = 1.0
    adv_estimator: Literal["grpo", "rloo", "reinforce_plus_plus"] = "grpo"
    advantage_temperature: float = 1.0
    max_advantage_weight: float = 20.0

    filter_by_accuracy: bool = False
    accuracy_lower_bound: float = 0.1
    accuracy_upper_bound: float = 0.9

    total_epochs: int = 80
    num_train_steps_per_epoch: int = 10
    learning_rate: float = 5e-6
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4

    reward_aggregation: Literal["sum", "mean", "last", "binary_sum"] = "sum"
    reward_threshold: float = 0.0
    frames_per_episode: int = 2
    batch_size: int = 32

    config_name: str = "physics_aware_srb_train_tracking"
    checkpoint_dir: str | None = None
    action_horizon: int = 16
    action_dim: int = 19
    seed: int = 42
    log_interval: int = 1
    save_interval: int = 5
    wandb_enabled: bool = False
    tensorboard_enabled: bool = True
    fsdp_devices: int | None = None
    debug_train_step: bool = False  # Set to True to disable JIT compilation for debugging purposes.


def prepare_advantages(advantages: jnp.ndarray, *, frame_counts: Sequence[int]) -> jnp.ndarray:
    """Expand one episode-level advantage for every sampled frame."""
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    frame_counts = tuple(int(frame_count) for frame_count in frame_counts)
    if advantages.size != len(frame_counts):
        raise ValueError(
            f"Got {advantages.size} episode advantages for {len(frame_counts)} sampled episode frame counts."
        )
    if any(frame_count <= 0 for frame_count in frame_counts):
        raise ValueError("Every sampled episode must contribute at least one frame.")

    return jnp.repeat(advantages, jnp.asarray(frame_counts, dtype=jnp.int32))


def _to_jax_array(value):
    if isinstance(value, (np.ndarray,)):
        return jnp.asarray(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return jnp.asarray(value.detach().cpu().numpy())
    if isinstance(value, (list, tuple)):
        return jnp.asarray(value)
    return value


def _prepare_batch_for_sharding(observation, actions, *, mesh, batch_sharding, data_sharding):
    """Move the observation/action batch onto the configured sharding."""
    observation = jax.tree.map(lambda x: jax.device_put(x, batch_sharding), observation)
    actions = jax.device_put(actions, batch_sharding)
    return observation, actions


def init_wandb(rl_config: OfflineJAXConfig, base_config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging for the offline RL run."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = base_config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=f"{base_config.project_name}_rl")
    else:
        wandb.init(
            name=f"rl_offline_jax_{base_config.exp_name}",
            config={**dataclasses.asdict(rl_config), "base_config": dataclasses.asdict(base_config)},
            project=f"{base_config.project_name}_rl",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def init_tensorboard(
    rl_config: OfflineJAXConfig,
    base_config: _config.TrainConfig,
    *,
    resuming: bool,
    enabled: bool = True,
):
    """Initialize TensorBoard logging for the offline RL run."""
    if not enabled or SummaryWriter is None:
        return None

    if jax.process_index() != 0:
        return None

    log_dir = base_config.checkpoint_dir / "rl_grpo_offline_jax" / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    hparams_text = (
        f"## Offline JAX RL Config\n```json\n{dataclasses.asdict(rl_config)}\n```\n\n"
        f"## Base Config\n{rl_config.config_name}"
    )
    writer.add_text("hyperparameters", hparams_text)
    logging.info(f"TensorBoard logs -> {log_dir}")
    return writer


def maybe_initialize_distributed() -> None:
    """Initialize JAX distributed runtime when launched under multi-process settings."""
    if jax.distributed.is_initialized():
        return

    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator_address:
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if master_addr and master_port:
            coordinator_address = f"{master_addr}:{master_port}"

    num_processes = os.environ.get("JAX_NUM_PROCESSES") or os.environ.get("WORLD_SIZE")
    process_id = os.environ.get("JAX_PROCESS_ID") or os.environ.get("RANK")

    if coordinator_address and num_processes and process_id:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=int(num_processes),
            process_id=int(process_id),
        )


def get_resume_start_epoch(checkpoint_manager, *, resuming: bool) -> int:
    """Return the completed epoch represented by the latest checkpoint."""
    if not resuming:
        return 0

    checkpoint_steps = checkpoint_manager.all_steps()
    if not checkpoint_steps:
        raise RuntimeError("Cannot resume because no completed checkpoint step was found.")
    return int(max(checkpoint_steps))


def main(rl_config: OfflineJAXConfig):
    """Main offline RL training loop."""
    init_logging()
    maybe_initialize_distributed()

    logging.info(f"Running JAX offline RL on: {platform.node()}")
    logging.info(
        "JAX distributed info: process=%d/%d local_devices=%d global_devices=%d",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )
    logging.info(f"Offline RL config: {dataclasses.asdict(rl_config)}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    base_config = _config.get_config(rl_config.config_name)
    if rl_config.checkpoint_dir:
        base_config = dataclasses.replace(
            base_config,
            checkpoint_dir_override=pathlib.Path(rl_config.checkpoint_dir),
        )

    if base_config.model.action_horizon != rl_config.action_horizon:
        object.__setattr__(base_config.model, "action_horizon", rl_config.action_horizon)

    rng = jax.random.PRNGKey(rl_config.seed + jax.process_index())
    train_rng, init_rng = jax.random.split(rng)

    num_fsdp_devices = rl_config.fsdp_devices or base_config.fsdp_devices
    mesh = sharding.make_mesh(num_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.BATCH_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rl_ckpt_dir = base_config.checkpoint_dir / "rl_grpo_offline_jax"
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        rl_ckpt_dir,
        keep_period=rl_config.save_interval,
        overwrite=False,
        resume=True,
    )

    writer = init_tensorboard(
        rl_config,
        base_config,
        resuming=resuming,
        enabled=rl_config.tensorboard_enabled,
    )

    # init_wandb(rl_config, base_config, resuming=resuming, enabled=rl_config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        base_config,
        sharding=data_sharding,
        shuffle=True,
        episode_aware=True,
        reward_aggregation=rl_config.reward_aggregation,
        reward_threshold=rl_config.reward_threshold,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    dataset = data_loader.episode_dataset
    logging.info(
        f"Offline dataset ready: {dataset.total_episodes} episodes, {dataset.total_frames} frames"
    )

    train_state, train_state_sharding = init_train_state(
        base_config,
        init_rng,
        mesh,
        resume=resuming,
        learning_rate=rl_config.learning_rate,
        grad_clip_norm=rl_config.grad_clip_norm,
        weight_decay=rl_config.weight_decay,
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    start_epoch = get_resume_start_epoch(checkpoint_manager, resuming=resuming)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info("Resuming from completed epoch %d.", start_epoch)

    if start_epoch > rl_config.total_epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} exceeds configured total_epochs={rl_config.total_epochs}."
        )
    initial_train_step = int(jax.device_get(train_state.step))

    debug_train_step = rl_config.debug_train_step or os.environ.get("RLTUNE_DEBUG_TRAIN_STEP", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if debug_train_step:   
        logging.info("Debug train-step mode enabled; using unjitted train step for stepping.")
        ptrain_step = functools.partial(
            offline_rl_train_step,
            base_config,
            advantage_temperature=rl_config.advantage_temperature,
            max_advantage_weight=rl_config.max_advantage_weight,
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(
                offline_rl_train_step,
                base_config,
                advantage_temperature=rl_config.advantage_temperature,
                max_advantage_weight=rl_config.max_advantage_weight,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, batch_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    np_rng = np.random.default_rng(rl_config.seed + jax.process_index())
    best_reward = -float("inf")

    last_saved_epoch = start_epoch
    for epoch in range(start_epoch, rl_config.total_epochs):
        epoch_start = time.time()
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Epoch {epoch + 1}/{rl_config.total_epochs}")
        logging.info(f"{'=' * 60}")

        group = dataset.sample_episodes_for_grpo(
            n_groups=1,
            n_samples_per_group=rl_config.n_samples,
            rng=np_rng,
        )[0]

        rewards = np.array([dataset.get_episode_reward(ep_idx) for ep_idx in group], dtype=np.float32)
        prompt_indices = np.zeros(len(group), dtype=np.int16)

        if rl_config.filter_by_accuracy:
            binary_rewards = np.array([1.0 if dataset.get_episode_success(ep_idx) else 0.0 for ep_idx in group])
            mask, filter_metrics = filter_by_accuracy(
                binary_rewards,
                prompt_indices,
                lower_bound=rl_config.accuracy_lower_bound,
                upper_bound=rl_config.accuracy_upper_bound,
            )
            logging.info(f"Filter metrics: {filter_metrics}")
            if mask.sum() == 0:
                logging.warning("All episodes filtered out; skipping epoch.")
                continue
            filtered_indices = np.where(mask)[0]
            group = [group[i] for i in filtered_indices]
            rewards = rewards[filtered_indices]
            prompt_indices = prompt_indices[filtered_indices]

        advantages = compute_advantages(
            rewards,
            prompt_indices,
            estimator=rl_config.adv_estimator,
            n_samples=rl_config.n_samples,
            gamma=rl_config.gamma,
        )
        logging.info(
            f"Episode rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
            f"adv_mean={advantages.mean():.4f}, adv_std={advantages.std():.4f}"
        )

        step_infos = []
        local_batch_size = max(1, rl_config.batch_size // max(1, jax.process_count()))
        step_group = group
        frame_counts = [
            min(rl_config.frames_per_episode, dataset.get_episode_length(episode_index))
            for episode_index in step_group
        ]
        expected_batch_size = sum(frame_counts)
        if expected_batch_size > local_batch_size:
            raise ValueError(
                "A complete GRPO group produces "
                f"{expected_batch_size} local frames, which exceeds batch_size={local_batch_size}. "
                "Increase batch_size or reduce n_samples/frames_per_episode instead of truncating the group."
            )

        for step in range(rl_config.num_train_steps_per_epoch):
            obs_batch, actions_np, _ = dataset.sample_batch_from_episodes(
                step_group,
                frames_per_episode=rl_config.frames_per_episode,
                rng=np_rng,
            )

            obs_dict = jax.tree.map(_to_jax_array, obs_batch.to_dict())
            observation = _model.Observation.from_dict(obs_dict)
            actions = jnp.asarray(actions_np, dtype=jnp.float32)
            adv_jax = prepare_advantages(jnp.asarray(advantages, dtype=jnp.float32), frame_counts=frame_counts)
            if adv_jax.shape[0] != actions.shape[0]:
                raise RuntimeError(
                    f"Advantage batch size {adv_jax.shape[0]} does not match action batch size {actions.shape[0]}."
                )
            observation, actions = _prepare_batch_for_sharding(
                observation, actions, mesh=mesh, batch_sharding=batch_sharding, data_sharding=batch_sharding
            )
            adv_jax = jax.device_put(adv_jax, batch_sharding)

            train_rng, step_rng = jax.random.split(train_rng)
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(step_rng, train_state, (observation, actions), adv_jax)
            step_infos.append(info)

            if step % rl_config.log_interval == 0:
                stacked = jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *step_infos)
                reduced = jax.device_get(stacked)
                logging.info(f"  Step {step}: {', '.join(f'{k}={v:.4f}' for k, v in reduced.items())}")
                if writer is not None:
                    global_step = (
                        initial_train_step
                        + (epoch - start_epoch) * rl_config.num_train_steps_per_epoch
                        + step
                        + 1
                    )
                    for key, value in reduced.items():
                        try:
                            writer.add_scalar(f"train/{key}", float(value), global_step)
                        except (TypeError, ValueError):
                            continue
                    writer.flush()
                step_infos = []

        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch completed in {epoch_time:.2f}s")

        mean_reward = float(np.mean(rewards))
        if mean_reward > best_reward:
            best_reward = mean_reward
            logging.info(f"New best reward: {best_reward:.4f}")

        if writer is not None:
            writer.add_scalar("epoch/mean_reward", mean_reward, epoch + 1)
            writer.add_scalar("epoch/advantage_mean", float(np.mean(advantages)), epoch + 1)
            writer.add_scalar("epoch/advantage_std", float(np.std(advantages)), epoch + 1)
            writer.flush()

        if (epoch + 1) % rl_config.save_interval == 0 and jax.process_index() == 0:
            logging.info(f"Saving checkpoint at epoch {epoch + 1}...")
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, epoch + 1)
            last_saved_epoch = epoch + 1
            # wandb.log({"epoch": epoch + 1, "best_reward": best_reward}, step=epoch + 1)

    if jax.process_index() == 0:
        if last_saved_epoch != rl_config.total_epochs:
            logging.info("Training complete. Saving final checkpoint...")
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, rl_config.total_epochs)
        checkpoint_manager.wait_until_finished()
        # wandb.log({"best_reward": best_reward}, step=rl_config.total_epochs)


if __name__ == "__main__":
    # main(tyro.cli(OfflineJAXConfig))
    main(OfflineJAXConfig())  # TODO: remove this line and uncomment the above line when tyro is fixed for dataclasses with frozen=True

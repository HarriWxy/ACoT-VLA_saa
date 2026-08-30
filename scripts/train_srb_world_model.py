"""Train a JAX/Flax ensemble world model from SRB transition arrays.

The script uses episode-level splits and computes all affine statistics from
the training split only.  The resulting directory can be loaded by
``openpi.control.world_model_io.load_world_model`` and consumed by CEM-MPC.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.control.transition_dataset import TransitionDataset
from openpi.control.world_model_io import save_world_model
from openpi.models.srb_world_model import SRBWorldModel
from openpi.models.srb_world_model import SRBWorldModelConfig
from openpi.models.srb_world_model import stack_ensemble

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    data: Path = Path("data/srb_dynamics/transitions.npz")
    output: Path = Path("checkpoints/srb_world_model")

    ensemble_size: int = 5
    hidden_dim: int = 512
    num_layers: int = 3
    batch_size: int = 1024
    epochs: int = 200
    steps_per_epoch: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    reward_loss_weight: float = 0.2
    fall_loss_weight: float = 0.1
    validation_fraction: float = 0.2
    validation_batch_size: int = 8192
    seed: int = 0


def _make_model_and_parameters(
    dataset: TransitionDataset,
    args: Args,
) -> tuple[SRBWorldModel, Any]:
    config = SRBWorldModelConfig(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        command_dim=dataset.command_dim,
        physics_dim=dataset.physics_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    model = SRBWorldModel(config)
    sample_state = jnp.zeros((1, dataset.state_dim), dtype=jnp.float32)
    sample_action = jnp.zeros((1, dataset.action_dim), dtype=jnp.float32)
    sample_command = jnp.zeros((1, dataset.command_dim), dtype=jnp.float32)
    sample_physics = jnp.zeros((1, dataset.physics_dim), dtype=jnp.float32)
    keys = jax.random.split(jax.random.key(args.seed), args.ensemble_size)
    parameter_trees = [
        model.init(key, sample_state, sample_action, sample_command, sample_physics)["params"] for key in keys
    ]
    return model, stack_ensemble(parameter_trees)


def _make_train_step(
    model: SRBWorldModel,
    optimizer: optax.GradientTransformation,
    reward_loss_weight: float,
    fall_loss_weight: float,
):
    @jax.jit
    def train_step(parameters, optimizer_state, batch, member_indices):
        def member_loss(member_parameters, indices):
            member_batch = jax.tree.map(lambda value: value[indices], batch)
            pred_delta, pred_reward, pred_fall = model.apply(
                {"params": member_parameters},
                member_batch["state"],
                member_batch["action"],
                member_batch["command"],
                member_batch["physics"],
            )
            target_delta = member_batch["next_state"] - member_batch["state"]
            delta_loss = jnp.mean(jnp.square(pred_delta - target_delta))
            reward_loss = jnp.mean(jnp.square(pred_reward[..., 0] - member_batch["reward"]))
            fall_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_fall[..., 0], member_batch["fall"]))
            return delta_loss + reward_loss_weight * reward_loss + fall_loss_weight * fall_loss

        def loss_fn(all_parameters):
            return jnp.mean(jax.vmap(member_loss)(all_parameters, member_indices))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, optimizer_state, loss

    return train_step


def _evaluate(
    model: SRBWorldModel,
    parameters: Any,
    batch: dict[str, jax.Array],
    reward_loss_weight: float,
    fall_loss_weight: float,
) -> float:
    def member_loss(member_parameters):
        pred_delta, pred_reward, pred_fall = model.apply(
            {"params": member_parameters},
            batch["state"],
            batch["action"],
            batch["command"],
            batch["physics"],
        )
        target_delta = batch["next_state"] - batch["state"]
        delta_loss = jnp.mean(jnp.square(pred_delta - target_delta))
        reward_loss = jnp.mean(jnp.square(pred_reward[..., 0] - batch["reward"]))
        fall_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_fall[..., 0], batch["fall"]))
        return delta_loss + reward_loss_weight * reward_loss + fall_loss_weight * fall_loss

    return float(jax.device_get(jnp.mean(jax.vmap(member_loss)(parameters))))


def main(args: Args) -> None:
    if args.ensemble_size < 2:
        raise ValueError("ensemble_size must be at least 2.")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive.")
    if args.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive.")

    dataset = TransitionDataset.load(args.data)
    train_indices, validation_indices = dataset.split_by_episode(args.validation_fraction, args.seed)
    stats = dataset.statistics(train_indices)
    model, parameters = _make_model_and_parameters(dataset, args)
    optimizer = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    optimizer_state = optimizer.init(parameters)
    train_step = _make_train_step(
        model,
        optimizer,
        reward_loss_weight=args.reward_loss_weight,
        fall_loss_weight=args.fall_loss_weight,
    )

    train_batch = {
        name: jnp.asarray(values, dtype=jnp.float32)
        for name, values in (
            ("state", stats["state"].normalize(dataset.state[train_indices])),
            ("action", stats["action"].normalize(dataset.action[train_indices])),
            ("command", stats["command"].normalize(dataset.command[train_indices])),
            ("physics", stats["physics"].normalize(dataset.physics[train_indices])),
            ("next_state", stats["state"].normalize(dataset.next_state[train_indices])),
            ("reward", dataset.reward[train_indices]),
            (
                "fall",
                dataset.fall[train_indices].astype(np.float32),
            ),
        )
    }
    validation_slice = validation_indices[: args.validation_batch_size]
    validation_batch = {
        name: jnp.asarray(values, dtype=jnp.float32)
        for name, values in (
            ("state", stats["state"].normalize(dataset.state[validation_slice])),
            ("action", stats["action"].normalize(dataset.action[validation_slice])),
            ("command", stats["command"].normalize(dataset.command[validation_slice])),
            ("physics", stats["physics"].normalize(dataset.physics[validation_slice])),
            ("next_state", stats["state"].normalize(dataset.next_state[validation_slice])),
            ("reward", dataset.reward[validation_slice]),
            (
                "fall",
                dataset.fall[validation_slice].astype(np.float32),
            ),
        )
    }

    rng = jax.random.key(args.seed + 1)
    steps_per_epoch = args.steps_per_epoch or max(1, train_indices.size // args.batch_size)
    best_validation_loss = float("inf")
    best_parameters = parameters

    for epoch in range(args.epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            rng, index_key = jax.random.split(rng)
            batch_indices = jax.random.randint(
                index_key,
                (args.ensemble_size, args.batch_size),
                minval=0,
                maxval=train_indices.size,
            )
            parameters, optimizer_state, loss = train_step(
                parameters,
                optimizer_state,
                train_batch,
                batch_indices,
            )
            epoch_losses.append(float(jax.device_get(loss)))

        validation_loss = _evaluate(
            model,
            parameters,
            validation_batch,
            reward_loss_weight=args.reward_loss_weight,
            fall_loss_weight=args.fall_loss_weight,
        )
        train_loss = float(np.mean(epoch_losses))
        logger.info(
            "epoch=%d/%d train_loss=%.6f validation_loss=%.6f",
            epoch + 1,
            args.epochs,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_parameters = parameters

    metadata = dict(dataset.metadata)
    metadata.update(
        {
            "train_size": int(train_indices.size),
            "validation_size": int(validation_indices.size),
            "train_episode_count": int(np.unique(dataset.episode_id[train_indices]).size),
            "validation_episode_count": int(np.unique(dataset.episode_id[validation_indices]).size),
            "validation_loss": best_validation_loss,
            "normalization": "fixed affine stats computed on train episodes only",
        }
    )
    if "action_low" not in metadata or "action_high" not in metadata:
        raise ValueError(
            "Dataset metadata must include action_low/action_high; recollect with collect_srb_dynamics.py."
        )
    save_world_model(
        output_dir=args.output,
        model_config=model.config,
        parameters=flax.core.unfreeze(best_parameters),
        stats=stats,
        metadata=metadata,
    )
    logger.info("Saved world-model checkpoint to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

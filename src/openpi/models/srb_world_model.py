"""Small physics-conditioned ensemble dynamics model for SRB control.

This is intentionally separate from the VLA models.  It consumes normalized
low-dimensional state, action, command and physics vectors and predicts a one
decision-step residual.  Multiple independently trained copies provide a cheap
epistemic-uncertainty signal for CEM/MPC.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SRBWorldModelConfig:
    state_dim: int
    action_dim: int
    command_dim: int
    physics_dim: int
    hidden_dim: int = 512
    num_layers: int = 3

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "command_dim", "physics_dim", "hidden_dim", "num_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SRBWorldModelConfig:
        return cls(
            **{
                key: int(value[key])
                for key in (
                    "state_dim",
                    "action_dim",
                    "command_dim",
                    "physics_dim",
                    "hidden_dim",
                    "num_layers",
                )
            }
        )


class SRBWorldModel(nn.Module):
    """Predict state residual, reward and fall probability logits."""

    config: SRBWorldModelConfig

    @nn.compact
    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        command: jax.Array,
        physics: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        command = jnp.asarray(command, dtype=jnp.float32)
        physics = jnp.asarray(physics, dtype=jnp.float32)
        x = jnp.concatenate([state, action, command, physics], axis=-1)

        for layer_index in range(self.config.num_layers):
            residual = x
            x = nn.Dense(self.config.hidden_dim, name=f"dense_{layer_index}")(x)
            x = nn.silu(x)
            x = nn.LayerNorm(name=f"norm_{layer_index}")(x)
            if residual.shape[-1] == self.config.hidden_dim:
                x = x + residual

        output = nn.Dense(self.config.state_dim + 2, name="head")(x)
        delta_state = output[..., : self.config.state_dim]
        reward = output[..., self.config.state_dim : self.config.state_dim + 1]
        fall_logit = output[..., self.config.state_dim + 1 : self.config.state_dim + 2]
        return delta_state, reward, fall_logit


def stack_ensemble(parameters: list[Any]) -> Any:
    """Stack a list of Linen parameter trees on a leading ensemble axis."""

    if not parameters:
        raise ValueError("At least one parameter tree is required.")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *parameters)


def apply_ensemble(
    model: SRBWorldModel,
    parameters: Any,
    state: jax.Array,
    action: jax.Array,
    command: jax.Array,
    physics: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply all ensemble members with a leading ensemble dimension."""

    return jax.vmap(
        lambda member_parameters: model.apply(
            {"params": member_parameters},
            state,
            action,
            command,
            physics,
        )
    )(parameters)

"""Cross-Entropy Method model-predictive controller for SRB.

The planner samples noisy action paths, evaluates them with a learned ensemble
world model, fits the distribution to elite paths, and executes only the first
action.  It therefore uses random actions as search proposals rather than as
behavior-cloning targets.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.control.srb_state import AffineStats
from openpi.models.srb_world_model import SRBWorldModel


@dataclasses.dataclass(frozen=True)
class CEMConfig:
    horizon: int = 20
    num_samples: int = 2048
    num_iterations: int = 5
    elite_size: int = 128
    initial_std_fraction: float = 0.25
    minimum_std_fraction: float = 0.02
    colored_noise_rho: float = 0.8
    mean_update_rate: float = 0.8
    uncertainty_cost: float = 1.0
    fall_cost: float = 10.0
    action_rate_cost: float = 0.1
    action_rate_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.num_samples <= 0 or self.num_iterations <= 0:
            raise ValueError("horizon, num_samples and num_iterations must be positive.")
        if not 0 < self.elite_size <= self.num_samples:
            raise ValueError("elite_size must be in (0, num_samples].")
        if not 0 <= self.colored_noise_rho < 1:
            raise ValueError("colored_noise_rho must be in [0, 1).")
        if not 0 < self.mean_update_rate <= 1:
            raise ValueError("mean_update_rate must be in (0, 1].")
        if self.initial_std_fraction <= 0 or self.minimum_std_fraction <= 0:
            raise ValueError("std fractions must be positive.")
        if self.action_rate_fraction <= 0:
            raise ValueError("action_rate_fraction must be positive.")


@dataclasses.dataclass(frozen=True)
class CEMPlan:
    """Planner result returned in the original (unnormalized) action space."""

    actions: np.ndarray
    predicted_scores: np.ndarray
    predicted_uncertainty: np.ndarray

    @property
    def first_action(self) -> np.ndarray:
        return self.actions[0]


def colored_noise(
    key: jax.Array,
    shape: tuple[int, int, int],
    rho: float,
) -> jax.Array:
    """Generate temporally correlated standard-normal noise."""

    if len(shape) != 3:
        raise ValueError("shape must be (num_samples, horizon, action_dim).")
    samples, horizon, action_dim = shape
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1).")
    keys = jax.random.split(key, horizon)
    innovations = jax.vmap(lambda item: jax.random.normal(item, (samples, action_dim)))(keys)
    if horizon == 1:
        return innovations.transpose(1, 0, 2)

    def step(previous: jax.Array, innovation: jax.Array) -> tuple[jax.Array, jax.Array]:
        current = rho * previous + jnp.sqrt(1.0 - rho * rho) * innovation
        return current, current

    _, tail = jax.lax.scan(step, innovations[0], innovations[1:])
    return jnp.concatenate([innovations[0:1], tail], axis=0).transpose(1, 0, 2)


class CEMPlanner:
    """CEM planner backed by a stacked :class:`SRBWorldModel` ensemble."""

    def __init__(
        self,
        model: SRBWorldModel,
        parameters: Any,
        action_low: np.ndarray,
        action_high: np.ndarray,
        stats: dict[str, AffineStats] | None = None,
        config: CEMConfig | None = None,
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if self.action_low.shape != self.action_high.shape or np.any(self.action_high <= self.action_low):
            raise ValueError("Action bounds must have matching, non-empty intervals.")
        self.stats = dict(stats or {})
        self.config = config or CEMConfig()
        if self.config.elite_size > self.config.num_samples:
            raise ValueError("elite_size cannot exceed num_samples.")
        for name, dim in (
            ("state", model.config.state_dim),
            ("action", model.config.action_dim),
            ("command", model.config.command_dim),
            ("physics", model.config.physics_dim),
        ):
            if name in self.stats and self.stats[name].dim != dim:
                raise ValueError(f"{name} stats has dim {self.stats[name].dim}, expected {dim}.")
        if model.config.action_dim != self.action_low.size:
            raise ValueError("Model action_dim and action bounds have different sizes.")

        self._key = jax.random.key(self.config.seed)
        self._warm_start: np.ndarray | None = None
        self._rollout_jit = jax.jit(self._rollout)

    def reset(self) -> None:
        """Reset random state and the receding-horizon warm start."""

        self._key = jax.random.key(self.config.seed)
        self._warm_start = None

    def _normalize(self, name: str, value: jax.Array) -> jax.Array:
        stats = self.stats.get(name)
        if stats is None:
            return value
        mean = jnp.asarray(stats.mean)
        std = jnp.asarray(stats.std)
        return (value - mean) / std

    def _project_actions(self, actions: jax.Array, previous_action: jax.Array) -> jax.Array:
        """Apply box and per-step rate constraints to [K, H, A] actions."""

        low = jnp.asarray(self.action_low)
        high = jnp.asarray(self.action_high)
        actions = jnp.clip(actions, low, high)
        rate_limit = (high - low) * self.config.action_rate_fraction

        def step(previous: jax.Array, current: jax.Array) -> tuple[jax.Array, jax.Array]:
            current = jnp.clip(current, previous - rate_limit, previous + rate_limit)
            current = jnp.clip(current, low, high)
            return current, current

        previous = jnp.broadcast_to(previous_action, (actions.shape[0], actions.shape[-1]))
        _, projected = jax.lax.scan(
            step,
            previous,
            actions.transpose(1, 0, 2),
        )
        return projected.transpose(1, 0, 2)

    def _rollout(
        self,
        state: jax.Array,
        command: jax.Array,
        physics: jax.Array,
        previous_action: jax.Array,
        candidate_actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Roll out all ensemble members and return score/uncertainty per candidate."""

        candidate_actions = self._project_actions(candidate_actions, previous_action)
        state = self._normalize("state", state)
        command = self._normalize("command", command)
        physics = self._normalize("physics", physics)
        normalized_actions = self._normalize("action", candidate_actions)

        num_members = jax.tree.leaves(self.parameters)[0].shape[0]
        num_samples = candidate_actions.shape[0]
        state_members = jnp.broadcast_to(state, (num_members, num_samples, state.shape[-1]))
        command_members = jnp.broadcast_to(command, (num_members, num_samples, command.shape[-1]))
        physics_members = jnp.broadcast_to(physics, (num_members, num_samples, physics.shape[-1]))
        returns = jnp.zeros((num_members, num_samples), dtype=jnp.float32)
        uncertainty = jnp.zeros((num_samples,), dtype=jnp.float32)

        def apply_members(
            member_parameters: Any,
            member_state: jax.Array,
            member_action: jax.Array,
            member_command: jax.Array,
            member_physics: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return self.model.apply(
                {"params": member_parameters},
                member_state,
                member_action,
                member_command,
                member_physics,
            )

        for timestep in range(self.config.horizon):
            actions_t = normalized_actions[:, timestep, :]
            actions_members = jnp.broadcast_to(
                actions_t,
                (num_members, num_samples, actions_t.shape[-1]),
            )
            delta_state, reward, fall_logit = jax.vmap(apply_members)(
                self.parameters,
                state_members,
                actions_members,
                command_members,
                physics_members,
            )
            next_state_members = state_members + delta_state
            returns = returns + reward[..., 0]
            returns = returns - self.config.fall_cost * jax.nn.sigmoid(fall_logit[..., 0])
            uncertainty = uncertainty + jnp.mean(jnp.var(next_state_members, axis=0), axis=-1)
            state_members = next_state_members

        raw_delta = candidate_actions[:, 1:, :] - candidate_actions[:, :-1, :]
        rate_penalty = (
            jnp.mean(jnp.square(raw_delta), axis=(1, 2))
            if self.config.horizon > 1
            else jnp.zeros((num_samples,), dtype=jnp.float32)
        )
        returns = returns - self.config.action_rate_cost * rate_penalty[None, :]
        mean_returns = jnp.mean(returns, axis=0)
        scores = mean_returns - self.config.uncertainty_cost * uncertainty
        return scores, uncertainty

    def rollout(
        self,
        state: np.ndarray,
        command: np.ndarray,
        physics: np.ndarray,
        previous_action: np.ndarray,
        candidate_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Public, unoptimized rollout useful for diagnostics and tests."""

        scores, uncertainty = self._rollout_jit(
            jnp.asarray(state, dtype=jnp.float32),
            jnp.asarray(command, dtype=jnp.float32),
            jnp.asarray(physics, dtype=jnp.float32),
            jnp.asarray(previous_action, dtype=jnp.float32),
            jnp.asarray(candidate_actions, dtype=jnp.float32),
        )
        return np.asarray(scores), np.asarray(uncertainty)

    def plan(
        self,
        state: np.ndarray,
        command: np.ndarray,
        physics: np.ndarray,
        previous_action: np.ndarray,
        warm_start: np.ndarray | None = None,
    ) -> CEMPlan:
        """Optimize an action sequence and return it in raw action units."""

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        physics = np.asarray(physics, dtype=np.float32).reshape(-1)
        previous_action = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        expected = {
            "state": self.model.config.state_dim,
            "command": self.model.config.command_dim,
            "physics": self.model.config.physics_dim,
            "previous_action": self.model.config.action_dim,
        }
        for name, value in (
            ("state", state),
            ("command", command),
            ("physics", physics),
            ("previous_action", previous_action),
        ):
            if value.size != expected[name]:
                raise ValueError(f"{name} has dim {value.size}, expected {expected[name]}.")

        if warm_start is None:
            warm_start = self._warm_start
        if warm_start is None:
            mean = (self.action_low + self.action_high) / 2.0
            mean = np.broadcast_to(mean, (self.config.horizon, mean.size)).copy()
        else:
            mean = np.asarray(warm_start, dtype=np.float32)
            if mean.shape != (self.config.horizon, self.model.config.action_dim):
                raise ValueError(
                    f"warm_start shape {mean.shape} does not match "
                    f"({self.config.horizon}, {self.model.config.action_dim})."
                )
        mean = jnp.asarray(mean)
        span = jnp.asarray(self.action_high - self.action_low)
        std = jnp.broadcast_to(span * self.config.initial_std_fraction, mean.shape)
        minimum_std = span * self.config.minimum_std_fraction

        for _ in range(self.config.num_iterations):
            self._key, noise_key = jax.random.split(self._key)
            noise = colored_noise(
                noise_key,
                (self.config.num_samples, self.config.horizon, self.model.config.action_dim),
                self.config.colored_noise_rho,
            )
            candidates = mean[None, :, :] + noise * std[None, :, :]
            candidates = self._project_actions(candidates, jnp.asarray(previous_action))
            scores, _ = self._rollout_jit(
                jnp.asarray(state),
                jnp.asarray(command),
                jnp.asarray(physics),
                jnp.asarray(previous_action),
                candidates,
            )
            _, elite_indices = jax.lax.top_k(scores, self.config.elite_size)
            elites = candidates[elite_indices]
            updated_mean = jnp.mean(elites, axis=0)
            updated_std = jnp.maximum(jnp.std(elites, axis=0), minimum_std)
            rate = self.config.mean_update_rate
            mean = (1.0 - rate) * mean + rate * updated_mean
            std = (1.0 - rate) * std + rate * updated_std

        actions = np.asarray(mean)
        final_scores, final_uncertainty = self._rollout_jit(
            jnp.asarray(state),
            jnp.asarray(command),
            jnp.asarray(physics),
            jnp.asarray(previous_action),
            mean[None, ...],
        )
        self._warm_start = np.concatenate([actions[1:], actions[-1:]], axis=0)
        return CEMPlan(
            actions=actions,
            predicted_scores=np.asarray(final_scores),
            predicted_uncertainty=np.asarray(final_uncertainty),
        )

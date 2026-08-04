from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from openpi_client import base_policy as _base_policy

DebugPolicyMode = Literal["random", "zero"]


@dataclasses.dataclass(frozen=True)
class DebugPolicyConfig:
    action_horizon: int
    action_dim: int
    mode: DebugPolicyMode = "random"
    seed: int = 0
    action_scale: float = 0.25


class DebugChunkPolicy(_base_policy.BasePolicy):
    def __init__(self, config: DebugPolicyConfig) -> None:
        if config.action_horizon <= 0:
            raise ValueError(f"Expected action_horizon > 0, got {config.action_horizon}")
        if config.action_dim <= 0:
            raise ValueError(f"Expected action_dim > 0, got {config.action_dim}")
        if config.action_scale < 0:
            raise ValueError(f"Expected action_scale >= 0, got {config.action_scale}")
        if config.mode not in ("random", "zero"):
            raise ValueError(f"Unsupported debug policy mode: {config.mode}")

        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def infer(self, obs: dict) -> dict:
        del obs

        if self._config.mode == "zero":
            actions = np.zeros((self._config.action_horizon, self._config.action_dim), dtype=np.float32)
        else:
            actions = self._rng.uniform(
                low=-self._config.action_scale,
                high=self._config.action_scale,
                size=(self._config.action_horizon, self._config.action_dim),
            ).astype(np.float32)

        return {"actions": actions}

    @property
    def metadata(self) -> dict:
        return {
            "policy_mode": self._config.mode,
            "action_horizon": self._config.action_horizon,
            "action_dim": self._config.action_dim,
            "random_seed": self._config.seed,
            "random_action_scale": self._config.action_scale,
        }
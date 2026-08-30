"""Temporally smooth, bounded exploration for SRB transition collection."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SmoothExplorationConfig:
    """Configuration for OU plus random Fourier exploration."""

    periodic_scale: float = 0.25
    ou_sigma: float = 0.08
    ou_theta: float = 0.15
    action_rate_fraction: float = 0.15
    num_sinusoids: int = 3
    min_frequency_hz: float = 0.3
    max_frequency_hz: float = 2.0
    dt: float = 0.04

    def __post_init__(self) -> None:
        if self.num_sinusoids < 0:
            raise ValueError("num_sinusoids must be non-negative.")
        if self.min_frequency_hz < 0 or self.max_frequency_hz < self.min_frequency_hz:
            raise ValueError("Invalid sinusoid frequency range.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        for name in ("periodic_scale", "ou_sigma", "action_rate_fraction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")


class SmoothRandomExplorer:
    """Generate bounded exploration without independently-jittered joints.

    The generated actions are not demonstrations.  They are deliberately
    structured inputs for system identification, with bounded action rates so
    that transitions remain useful for a learned dynamics model.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        config: SmoothExplorationConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float32).reshape(-1)
        self.high = np.asarray(high, dtype=np.float32).reshape(-1)
        if self.low.shape != self.high.shape or np.any(self.high <= self.low):
            raise ValueError("Action bounds must have matching, non-empty intervals.")
        self.config = config or SmoothExplorationConfig()
        self._rng = np.random.default_rng(seed)
        self._span = (self.high - self.low) / 2.0
        self._center = (self.high + self.low) / 2.0
        self._ou_state = np.zeros_like(self.low)
        self._previous = self._center.copy()
        self._frequencies = np.zeros(self.config.num_sinusoids, dtype=np.float32)
        self._amplitudes = np.zeros((self.config.num_sinusoids, self.low.size), dtype=np.float32)
        self._phases = np.zeros_like(self._amplitudes)
        self.reset()

    @property
    def action_dim(self) -> int:
        return int(self.low.size)

    def reset(self, nominal: np.ndarray | None = None) -> None:
        """Reset OU state and sample a new random Fourier episode."""

        if nominal is None:
            nominal_array = self._center.copy()
        else:
            nominal_array = np.asarray(nominal, dtype=np.float32).reshape(-1)
            if nominal_array.shape != self.low.shape:
                raise ValueError("nominal action has the wrong dimension.")
            nominal_array = np.clip(nominal_array, self.low, self.high)

        self._ou_state.fill(0.0)
        self._previous = nominal_array
        if self.config.num_sinusoids:
            self._frequencies = self._rng.uniform(
                self.config.min_frequency_hz,
                self.config.max_frequency_hz,
                size=self.config.num_sinusoids,
            ).astype(np.float32)
            self._amplitudes = self._rng.normal(size=(self.config.num_sinusoids, self.action_dim)).astype(
                np.float32
            ) * (self._span[None, :] * self.config.periodic_scale)
            self._phases = self._rng.uniform(
                -np.pi,
                np.pi,
                size=(self.config.num_sinusoids, self.action_dim),
            ).astype(np.float32)

    def sample(self, time_seconds: float, nominal: np.ndarray | None = None) -> np.ndarray:
        """Return one bounded, rate-limited action."""

        if nominal is None:
            nominal_array = self._center
        else:
            nominal_array = np.asarray(nominal, dtype=np.float32).reshape(-1)
            if nominal_array.shape != self.low.shape:
                raise ValueError("nominal action has the wrong dimension.")

        cfg = self.config
        self._ou_state += cfg.ou_theta * (0.0 - self._ou_state) * cfg.dt + cfg.ou_sigma * np.sqrt(
            cfg.dt
        ) * self._rng.normal(size=self.action_dim).astype(np.float32)
        ou = self._ou_state * self._span

        periodic = np.zeros_like(self.low)
        for frequency, amplitude, phase in zip(
            self._frequencies,
            self._amplitudes,
            self._phases,
            strict=True,
        ):
            periodic += amplitude * np.sin(2.0 * np.pi * frequency * time_seconds + phase)

        candidate = np.clip(nominal_array + periodic + ou, self.low, self.high)
        max_delta = self._span * cfg.action_rate_fraction
        candidate = np.clip(candidate, self._previous - max_delta, self._previous + max_delta)
        candidate = np.clip(candidate, self.low, self.high).astype(np.float32)
        self._previous = candidate
        return candidate

    def sample_sequence(
        self,
        length: int,
        nominal: np.ndarray | None = None,
        start_time_seconds: float = 0.0,
    ) -> np.ndarray:
        """Generate a complete exploration sequence for tests or rollouts."""

        if length <= 0:
            raise ValueError("length must be positive.")
        self.reset(nominal=nominal)
        return np.stack(
            [self.sample(start_time_seconds + i * self.config.dt, nominal=nominal) for i in range(length)],
            axis=0,
        )

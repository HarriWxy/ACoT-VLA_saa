"""State, physics and normalization contracts used by the SRB MPC pipeline.

The regular VLA data path is intentionally kept separate from this contract.
For model-based control we need a fixed, named state layout and the exact same
affine normalization at collection, training and inference time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy-like, torch and JAX values to a numpy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup(mapping: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted observation path.

    A direct key match is attempted first because some wrappers expose keys that
    themselves contain dots.
    """

    if path in mapping:
        return mapping[path]

    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(f"Observation field '{path}' is missing (failed at '{component}').")
        current = current[component]
    return current


def _leaf_paths(value: Any, prefix: str) -> list[tuple[str, np.ndarray]]:
    """Flatten nested mappings into deterministic leaf paths."""

    if isinstance(value, Mapping):
        leaves: list[tuple[str, np.ndarray]] = []
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_leaf_paths(value[key], child_prefix))
        return leaves
    return [(prefix, _to_numpy(value))]


@dataclass(frozen=True)
class FieldSpec:
    """One flattened numeric observation field."""

    path: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FieldSpec:
        return cls(path=str(value["path"]), shape=tuple(int(x) for x in value["shape"]))


@dataclass(frozen=True)
class ObservationSchema:
    """A stable flattened layout for nested SRB observations.

    ``root_paths`` passed to :meth:`from_observation` may point to either an
    array (for example ``"state"``) or a nested mapping (for example
    ``"proprio_dyn"``).  Nested mappings are expanded in sorted key order and
    the resulting leaf paths are stored in the schema.
    """

    fields: tuple[FieldSpec, ...]

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, Any],
        root_paths: tuple[str, ...] | list[str],
    ) -> ObservationSchema:
        fields: list[FieldSpec] = []
        seen: set[str] = set()
        for root_path in root_paths:
            value = _lookup(observation, root_path)
            for leaf_path, leaf_value in _leaf_paths(value, root_path):
                if leaf_path in seen:
                    continue
                array = np.asarray(leaf_value)
                if not np.issubdtype(array.dtype, np.number):
                    raise TypeError(f"Observation field '{leaf_path}' is not numeric: {array.dtype}")
                fields.append(FieldSpec(path=leaf_path, shape=tuple(int(x) for x in array.shape)))
                seen.add(leaf_path)
        if not fields:
            raise ValueError("ObservationSchema cannot be empty.")
        return cls(fields=tuple(fields))

    @property
    def dim(self) -> int:
        return sum(field.size for field in self.fields)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(field.path for field in self.fields)

    def pack(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Flatten an observation according to the persisted schema."""

        parts: list[np.ndarray] = []
        for field in self.fields:
            value = np.asarray(_lookup(observation, field.path), dtype=np.float32)
            if value.size != field.size:
                raise ValueError(f"Observation field '{field.path}' has {value.size} values; expected {field.size}.")
            parts.append(value.reshape(-1))
        return np.concatenate(parts, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [field.to_dict() for field in self.fields], "dim": self.dim}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ObservationSchema:
        schema = cls(fields=tuple(FieldSpec.from_dict(item) for item in value["fields"]))
        expected_dim = value.get("dim")
        if expected_dim is not None and int(expected_dim) != schema.dim:
            raise ValueError(f"Schema dimension mismatch: metadata={expected_dim}, fields={schema.dim}.")
        return schema


@dataclass(frozen=True)
class AffineStats:
    """Fixed per-dimension affine statistics.

    The transform is invertible and must be persisted with the checkpoint.  A
    separate set of statistics should be used for state, action, command and
    physics vectors; never compute statistics independently for each inference
    sample or planet.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        if mean.shape != std.shape:
            raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Normalization statistics must be finite.")
        if np.any(std <= 0):
            raise ValueError("Normalization standard deviations must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def from_array(cls, values: np.ndarray, min_std: float = 1e-6) -> AffineStats:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        values = values.reshape(-1, values.shape[-1])
        if values.shape[0] < 2:
            raise ValueError("At least two samples are required to compute statistics.")
        mean = values.mean(axis=0)
        std = np.maximum(values.std(axis=0), min_std)
        return cls(mean=mean, std=std)

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {values.shape[-1]}.")
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {values.shape[-1]}.")
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AffineStats:
        return cls(mean=np.asarray(value["mean"], dtype=np.float32), std=np.asarray(value["std"], dtype=np.float32))


# A descriptive alias used by callers that want to make the physics contract
# explicit without introducing a second, subtly different normalization type.
PhysicsNormalizer = AffineStats


def parse_paths(value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI field list."""

    paths = tuple(path.strip() for path in value.split(",") if path.strip())
    if not paths:
        raise ValueError("At least one observation path is required.")
    return paths

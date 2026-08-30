"""A compact numpy transition format for the low-dimensional SRB world model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.control.srb_state import AffineStats


@dataclass
class TransitionDataset:
    """Flat transitions with explicit episode boundaries.

    The format intentionally avoids images and LeRobot transforms.  It is meant
    for the low-dimensional dynamics model; the original VLA dataset remains
    untouched.
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    command: np.ndarray
    physics: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_id: np.ndarray
    fall: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float32)
        self.action = np.asarray(self.action, dtype=np.float32)
        self.next_state = np.asarray(self.next_state, dtype=np.float32)
        self.command = np.asarray(self.command, dtype=np.float32)
        self.physics = np.asarray(self.physics, dtype=np.float32)
        self.reward = np.asarray(self.reward, dtype=np.float32).reshape(-1)
        self.terminated = np.asarray(self.terminated, dtype=bool).reshape(-1)
        self.truncated = np.asarray(self.truncated, dtype=bool).reshape(-1)
        self.episode_id = np.asarray(self.episode_id, dtype=np.int64).reshape(-1)
        if self.fall is None:
            self.fall = self.terminated.copy()
        else:
            self.fall = np.asarray(self.fall, dtype=bool).reshape(-1)

        arrays = {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "command": self.command,
            "physics": self.physics,
        }
        if any(array.ndim != 2 for array in arrays.values()):
            raise ValueError("state/action/next_state/command/physics must all be rank-2 arrays.")
        lengths = {name: array.shape[0] for name, array in arrays.items()}
        lengths.update(
            reward=self.reward.size,
            terminated=self.terminated.size,
            truncated=self.truncated.size,
            episode_id=self.episode_id.size,
            fall=self.fall.size,
        )
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Transition arrays have inconsistent lengths: {lengths}.")
        if self.state.shape != self.next_state.shape:
            raise ValueError("state and next_state must have the same shape.")
        if self.state.shape[0] == 0:
            raise ValueError("TransitionDataset cannot be empty.")
        if not all(np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError("Transition inputs must be finite.")
        if not np.all(np.isfinite(self.reward)):
            raise ValueError("Rewards must be finite.")
        self.metadata = dict(self.metadata or {})

    @property
    def size(self) -> int:
        return int(self.state.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def command_dim(self) -> int:
        return int(self.command.shape[1])

    @property
    def physics_dim(self) -> int:
        return int(self.physics.shape[1])

    @property
    def episode_count(self) -> int:
        return int(np.unique(self.episode_id).size)

    def split_by_episode(
        self,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split without allowing adjacent frames from one episode to leak."""

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        episodes = np.unique(self.episode_id)
        if episodes.size < 2:
            raise ValueError("At least two episodes are required for a split.")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(episodes)
        num_validation = max(1, round(episodes.size * validation_fraction))
        num_validation = min(num_validation, episodes.size - 1)
        validation_episodes = shuffled[:num_validation]
        validation_mask = np.isin(self.episode_id, validation_episodes)
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    def statistics(self, indices: np.ndarray) -> dict[str, AffineStats]:
        """Compute fixed normalization from the training subset only."""

        indices = np.asarray(indices, dtype=np.int64)
        return {
            "state": AffineStats.from_array(self.state[indices]),
            "action": AffineStats.from_array(self.action[indices]),
            "command": AffineStats.from_array(self.command[indices]),
            "physics": AffineStats.from_array(self.physics[indices]),
        }

    def save(self, path: str | Path) -> None:
        """Save arrays to ``path`` and metadata to a same-stem JSON sidecar."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state=self.state,
            action=self.action,
            next_state=self.next_state,
            command=self.command,
            physics=self.physics,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id=self.episode_id,
            fall=self.fall,
        )
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> TransitionDataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Transition file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                name: data[name]
                for name in (
                    "state",
                    "action",
                    "next_state",
                    "command",
                    "physics",
                    "reward",
                    "terminated",
                    "truncated",
                    "episode_id",
                )
            }
            fall = data.get("fall", None)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return cls(**arrays, fall=fall, metadata=metadata)

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TransitionDataset:
        if not records:
            raise ValueError("At least one transition is required.")
        required = (
            "state",
            "action",
            "next_state",
            "command",
            "physics",
            "reward",
            "terminated",
            "truncated",
            "episode_id",
        )
        missing = [key for key in required if key not in records[0]]
        if missing:
            raise KeyError(f"Transition record is missing fields: {missing}")
        return cls(
            **{key: np.stack([record[key] for record in records]) for key in required},
            fall=np.asarray([record.get("fall", record["terminated"]) for record in records], dtype=bool),
            metadata=metadata,
        )

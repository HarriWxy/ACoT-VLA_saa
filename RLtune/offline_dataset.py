"""Offline dataset loader for RL fine-tuning from pre-collected demonstration data.

Loads episodes from LeRobot v2.1 parquet format and provides:
- Episode-level access (for GRPO advantage computation)
- Frame-level sampling (for policy gradient updates)
- Reward aggregation (per-episode rewards from per-frame signals)

Usage:
    dataset = OfflineRLDataset("dataset/srb_tracking")
    episodes = dataset.sample_episodes(n=8)
    batch_obs, batch_actions, batch_rewards = dataset.sample_batch(episodes)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """A single offline episode loaded from parquet."""

    episode_index: int
    task: str
    length: int

    # Per-frame arrays (T, ...)
    observation_state: np.ndarray  # (T, 24) float32
    observation_image_front: np.ndarray  # (T, 224, 224, 3) uint8
    actions: np.ndarray  # (T, 37) float32
    rewards: np.ndarray  # (T,) float32

    @property
    def episode_reward(self) -> float:
        """Aggregated episode reward (sum of per-frame rewards)."""
        return float(self.rewards.sum())

    @property
    def mean_reward(self) -> float:
        """Mean per-frame reward."""
        return float(self.rewards.mean())

    @property
    def success(self) -> bool:
        """Binary success signal (positive episode reward)."""
        return self.episode_reward > 0


@dataclass
class OfflineRLDataset:
    """Offline RL dataset from LeRobot parquet files.

    Loads all episodes from a dataset directory and provides methods
    for episode sampling and batch construction for GRPO training.

    Args:
        data_dir: Path to the dataset root (e.g., "dataset/srb_tracking").
        reward_aggregation: How to aggregate per-frame rewards to episode reward.
            "sum" | "mean" | "last" | "binary_sum".
        reward_threshold: Threshold for binary success detection.
    """

    data_dir: str
    reward_aggregation: str = "sum"
    reward_threshold: float = 0.0

    episodes: list[EpisodeData] = field(default_factory=list, init=False, repr=False)
    task_to_indices: dict[str, list[int]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._load()

    def _load(self):
        """Load all episodes from parquet files."""
        data_path = Path(self.data_dir)
        meta_path = data_path / "meta"

        # Load task mapping
        tasks_map: dict[int, str] = {}
        tasks_file = meta_path / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks_map[entry["task_index"]] = entry["task"]

        # Load episode metadata
        episodes_meta: dict[int, dict] = {}
        episodes_file = meta_path / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    episodes_meta[entry["episode_index"]] = entry

        # Load info for feature specs
        info_file = meta_path / "info.json"
        info = {}
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)

        # Find all parquet files
        parquet_files = sorted(data_path.glob("data/chunk-*/episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}/data/")

        logger.info(f"Loading {len(parquet_files)} episodes from {data_path}")

        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)

            # Determine episode index
            ep_idx_raw = df["episode_index"].iloc[0]
            ep_idx = int(ep_idx_raw) if not isinstance(ep_idx_raw, (list, np.ndarray)) else int(ep_idx_raw[0])

            # Get task from metadata or use default
            task = "walk and track the velocity command"
            if ep_idx in episodes_meta:
                ep_meta = episodes_meta[ep_idx]
                task_indices = ep_meta.get("tasks", [0])
                if isinstance(task_indices, list) and len(task_indices) > 0:
                    tid = task_indices[0]
                    task = tasks_map.get(tid, task)

            # Extract observation.state
            obs_state_raw = df["observation.state"].values
            obs_state = np.stack([np.asarray(v, dtype=np.float32).flatten() for v in obs_state_raw])

            # Extract actions
            action_raw = df["action"].values
            actions = np.stack([np.asarray(v, dtype=np.float32).flatten() for v in action_raw])

            # Extract rewards
            reward_raw = df["reward"].values
            rewards = np.array([float(np.asarray(v).flatten()[0]) for v in reward_raw], dtype=np.float32)

            # Extract images
            img_raw = df["observation.images.image_front"].values
            images = np.stack([np.asarray(v, dtype=np.uint8) for v in img_raw])

            episode = EpisodeData(
                episode_index=ep_idx,
                task=task,
                length=len(df),
                observation_state=obs_state,
                observation_image_front=images,
                actions=actions,
                rewards=rewards,
            )

            self.episodes.append(episode)

            if task not in self.task_to_indices:
                self.task_to_indices[task] = []
            self.task_to_indices[task].append(len(self.episodes) - 1)

        # Sort by episode index
        self.episodes.sort(key=lambda e: e.episode_index)

        # Rebuild task_to_indices after sorting
        self.task_to_indices.clear()
        for i, ep in enumerate(self.episodes):
            if ep.task not in self.task_to_indices:
                self.task_to_indices[ep.task] = []
            self.task_to_indices[ep.task].append(i)

        total_frames = sum(ep.length for ep in self.episodes)
        logger.info(f"Loaded {len(self.episodes)} episodes, {total_frames} frames, {len(self.task_to_indices)} tasks")
        for task, indices in self.task_to_indices.items():
            rewards = [self.episodes[i].episode_reward for i in indices]
            logger.info(
                f"  Task '{task}': {len(indices)} episodes, "
                f"reward mean={np.mean(rewards):.3f} std={np.std(rewards):.3f}"
            )

    # ------------------------------------------------------------------
    # Episode access
    # ------------------------------------------------------------------

    def get_episode_reward(self, ep: EpisodeData) -> float:
        """Get aggregated episode reward."""
        if self.reward_aggregation == "sum":
            return ep.episode_reward
        if self.reward_aggregation == "mean":
            return ep.mean_reward
        if self.reward_aggregation == "last":
            return float(ep.rewards[-1])
        if self.reward_aggregation == "binary_sum":
            return 1.0 if ep.episode_reward > self.reward_threshold else 0.0
        raise ValueError(f"Unknown aggregation: {self.reward_aggregation}")

    def get_episode_success(self, ep: EpisodeData) -> bool:
        """Get binary success for episode."""
        if self.reward_aggregation == "binary_sum":
            return ep.episode_reward > self.reward_threshold
        return ep.success

    # ------------------------------------------------------------------
    # Sampling for GRPO
    # ------------------------------------------------------------------

    def sample_episodes_for_grpo(
        self,
        n_groups: int = 1,
        n_samples_per_group: int = 8,
        task: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[list[EpisodeData]]:
        """Sample episode groups for GRPO advantage computation.

        Each group contains n_samples_per_group episodes sharing the same task.
        Returns a list of groups, each group being a list of EpisodeData.

        Args:
            n_groups: Number of prompt groups.
            n_samples_per_group: Episodes per group (GRPO group size).
            task: Specific task to sample from (None = random).
            rng: Numpy random generator.

        Returns:
            List of groups, each group is a list of EpisodeData.
        """
        if rng is None:
            rng = np.random.default_rng()

        groups = []
        available_tasks = list(self.task_to_indices.keys())

        for _ in range(n_groups):
            if task is not None:
                selected_task = task
            else:
                selected_task = rng.choice(available_tasks)

            indices = self.task_to_indices[selected_task]
            if len(indices) < n_samples_per_group:
                # Not enough episodes: sample with replacement
                chosen = rng.choice(indices, size=n_samples_per_group, replace=True)
            else:
                chosen = rng.choice(indices, size=n_samples_per_group, replace=False)

            group = [self.episodes[i] for i in chosen]
            groups.append(group)

        return groups

    def sample_batch_from_episodes(
        self,
        episodes: list[EpisodeData],
        frames_per_episode: int = 1,
        action_horizon: int = 16,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """Sample a training batch from a list of episodes.

        For each episode, randomly samples `frames_per_episode` frames.
        Each frame provides:
        - observation: state + image
        - action: action sequence of length `action_horizon` starting from frame

        Args:
            episodes: List of episodes to sample from.
            frames_per_episode: Number of frames to sample per episode.
            action_horizon: Length of action sequence per frame.
            rng: Numpy random generator.

        Returns:
            obs_dict: Observation dict with 'state' and 'image_base' arrays.
            action_batch: (B, action_horizon, action_dim) array.
            reward_batch: (B,) per-episode rewards (same for all frames in an episode).
        """
        if rng is None:
            rng = np.random.default_rng()

        states = []
        images = []
        action_seqs = []
        rewards = []

        for ep in episodes:
            ep_reward = self.get_episode_reward(ep)

            # Sample frame indices within this episode
            max_frame = ep.length - 1
            frame_indices = rng.integers(0, max(1, max_frame + 1), size=frames_per_episode)

            for fi in frame_indices:
                fi = int(fi)
                # State
                states.append(ep.observation_state[fi])

                # Image
                images.append(ep.observation_image_front[fi])

                # Action sequence: fi to fi + action_horizon, padded if needed
                action_seq = []
                for dt in range(action_horizon):
                    idx = min(fi + dt, ep.length - 1)
                    action_seq.append(ep.actions[idx])
                action_seqs.append(np.stack(action_seq, axis=0))

                # Episode-level reward
                rewards.append(ep_reward)

        obs_dict = {
            "state": np.stack(states, axis=0).astype(np.float32),
            "image_base": np.stack(images, axis=0).astype(np.uint8),
        }
        action_batch = np.stack(action_seqs, axis=0).astype(np.float32)
        reward_batch = np.array(rewards, dtype=np.float32)

        return obs_dict, action_batch, reward_batch

    def create_trajectory_data(
        self,
        episodes: list[EpisodeData],
        prompt_index: int = 0,
    ) -> list:
        """Create TrajectoryData objects from episodes for GRPO.

        Args:
            episodes: List of episodes.
            prompt_index: Shared prompt index for the group.

        Returns:
            List of TrajectoryData objects.
        """
        from RLtune.grpo_algo import TrajectoryData

        trajectories = []
        for ep in episodes:
            # Build observation dict
            obs = {
                "state": ep.observation_state,
                "image_base": ep.observation_image_front,
            }

            traj = TrajectoryData(
                observation=obs,
                actions=ep.actions,
                reward=self.get_episode_reward(ep),
                success=self.get_episode_success(ep),
                episode_length=ep.length,
                prompt_index=prompt_index,
            )
            trajectories.append(traj)

        return trajectories

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(ep.length for ep in self.episodes)

    def get_reward_stats(self) -> dict[str, float]:
        """Get reward statistics across all episodes."""
        rewards = [self.get_episode_reward(ep) for ep in self.episodes]
        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
        }


def create_offline_dataset(
    data_dir: str,
    reward_aggregation: str = "sum",
    reward_threshold: float = 0.0,
) -> OfflineRLDataset:
    """Factory function to create an offline RL dataset."""
    return OfflineRLDataset(
        data_dir=data_dir,
        reward_aggregation=reward_aggregation,
        reward_threshold=reward_threshold,
    )

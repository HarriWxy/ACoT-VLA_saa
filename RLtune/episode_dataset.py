"""Episode-aware wrapper around the standard LeRobot data pipeline.

Wraps a transformed LeRobotDataset (loaded via `create_torch_dataset` +
`transform_dataset`) to provide:
- Episode-level access for GRPO advantage computation
- Per-episode reward aggregation from per-frame rewards
- Episode-group sampling for GRPO training
- Frame-level batch sampling within episodes

This ensures offline RL uses the exact same normalization, tokenization,
and model transforms as SFT training.

The episode metadata (episode_index, reward, task) is extracted directly
from the underlying LeRobotDataset's ``hf_dataset`` for fast batch access,
avoiding expensive per-sample image loading.
"""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Literal, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareDataset:
    """Wraps a standard LeRobot dataset with episode-level access for RL.

    Uses TWO views of the same underlying data:
    - transformed_dataset: AFTER all transforms (repack, SRBInputs, normalize,
      tokenize, pad) — used to get model-ready observations. The transforms
      drop metadata keys (episode_index, frame_index, etc.).
    - For episode metadata (index, reward, task), we access the underlying
      LeRobotDataset's ``hf_dataset`` directly for fast batch access,
      avoiding expensive per-sample image loading.

    Args:
        transformed_dataset: Output of ``transform_dataset()`` — a
            ``TransformedDataset(LeRobotDataset, [all_transforms])``.
        reward_aggregation: How to aggregate per-frame rewards into an
            episode-level scalar ("sum", "mean", "last", "binary_sum").
        reward_threshold: Threshold for binary_sum aggregation.
    """

    def __init__(
        self,
        transformed_dataset,
        *,
        reward_aggregation: Literal["sum", "mean", "last", "binary_sum"] = "sum",
        reward_threshold: float = 0.0,
    ):
        self._transformed = transformed_dataset
        self._reward_aggregation = reward_aggregation
        self._reward_threshold = reward_threshold

        # Access underlying LeRobotDataset's hf_dataset for fast batch access
        # transformed_dataset._dataset is the LeRobotDataset
        raw_ds = transformed_dataset._dataset
        while hasattr(raw_ds, "_dataset") and not hasattr(raw_ds, "hf_dataset"):
            raw_ds = raw_ds._dataset
        hf = raw_ds.hf_dataset

        logger.info("Building episode index from hf_dataset...")
        ep_col = np.array(hf["episode_index"])
        reward_col = np.array(hf["reward"]).flatten()

        # Build episode index: episode_index -> list of sample indices
        self._episode_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, ep_idx in enumerate(ep_col):
            self._episode_to_indices[int(ep_idx)].append(i)

        # Compute episode rewards and cache prompts
        self._episode_rewards: dict[int, float] = {}
        self._episode_length: dict[int, int] = {}
        self._episode_prompts: dict[int, str] = {}

        # Try to get tasks from metadata
        task_names = None
        if hasattr(raw_ds, "meta") and hasattr(raw_ds.meta, "tasks"):
            task_names = raw_ds.meta.tasks

        for ep_idx, indices in self._episode_to_indices.items():
            self._episode_length[ep_idx] = len(indices)
            ep_rewards = reward_col[indices]
            self._episode_rewards[ep_idx] = self._aggregate_reward(ep_rewards.tolist())

            # Cache prompt from task_index
            if task_names is not None and "task_index" in hf.column_names:
                task_idx = int(hf[indices[0]]["task_index"])
                if 0 <= task_idx < len(task_names):
                    self._episode_prompts[ep_idx] = task_names[task_idx]
                else:
                    self._episode_prompts[ep_idx] = ""
            else:
                self._episode_prompts[ep_idx] = ""

        self._episode_indices = sorted(self._episode_to_indices.keys())

        # Cache rewards array for fast batch access in sample_batch_from_episodes
        self._reward_array = reward_col

        logger.info(
            f"EpisodeAwareDataset: {self.total_episodes} episodes, "
            f"{self.total_frames} frames, "
            f"reward_agg={reward_aggregation}"
        )

    def _aggregate_reward(self, rewards: list[float]) -> float:
        arr = np.array(rewards, dtype=np.float32)
        if self._reward_aggregation == "sum":
            return float(arr.sum())
        if self._reward_aggregation == "mean":
            return float(arr.mean())
        if self._reward_aggregation == "last":
            return float(arr[-1])
        if self._reward_aggregation == "binary_sum":
            return float((arr > self._reward_threshold).sum())
        raise ValueError(f"Unknown aggregation: {self._reward_aggregation}")

    # ── Properties ──

    @property
    def total_episodes(self) -> int:
        return len(self._episode_indices)

    @property
    def total_frames(self) -> int:
        return len(self._transformed)

    @property
    def episode_indices(self) -> list[int]:
        return list(self._episode_indices)

    # ── Episode access ──

    def get_episode_reward(self, episode_index: int) -> float:
        return self._episode_rewards[episode_index]

    def get_episode_success(self, episode_index: int) -> bool:
        return self.get_episode_reward(episode_index) > 0

    def get_episode_length(self, episode_index: int) -> int:
        return self._episode_length[episode_index]

    def get_episode_prompt(self, episode_index: int) -> str:
        return self._episode_prompts.get(episode_index, "")

    def get_reward_stats(self) -> dict[str, float]:
        rewards = [self._episode_rewards[e] for e in self._episode_indices]
        arr = np.array(rewards)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    # ── GRPO sampling ──

    def sample_episodes_for_grpo(
        self,
        n_groups: int = 1,
        n_samples_per_group: int = 8,
        rng: np.random.Generator | None = None,
    ) -> list[list[int]]:
        """Sample episode groups for GRPO training.

        Returns a list of episode groups, each containing episode indices.
        Episodes are sampled uniformly at random.
        """
        if rng is None:
            rng = np.random.default_rng()

        groups = []
        available = np.array(self._episode_indices)
        for _ in range(n_groups):
            if len(available) >= n_samples_per_group:
                chosen = rng.choice(available, size=n_samples_per_group, replace=False)
            else:
                chosen = rng.choice(available, size=n_samples_per_group, replace=True)
            groups.append(chosen.tolist())
        return groups

    def sample_batch_from_episodes(
        self,
        episode_indices: Sequence[int],
        frames_per_episode: int = 2,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """Sample a batch of frames from given episodes.

        Uses **transformed** dataset for model inputs (state, image, actions)
        and **raw** dataset for rewards (reward is dropped by RepackTransform).

        Returns:
            obs_batch: dict — batched model-ready tensors
            action_batch: np.ndarray (B, action_horizon, action_dim)
            reward_batch: np.ndarray (B,)
        """
        if rng is None:
            rng = np.random.default_rng()

        all_observations = []
        all_rewards = []

        for ep_idx in episode_indices:
            indices = self._episode_to_indices[ep_idx]
            n_frames = min(frames_per_episode, len(indices))

            if n_frames >= len(indices):
                chosen = list(indices)
            else:
                chosen_indices = rng.choice(len(indices), size=n_frames, replace=False)
                chosen = [indices[j] for j in sorted(chosen_indices)]

            for sample_idx in chosen:
                # Model inputs from transformed dataset
                all_observations.append(self._transformed[sample_idx])

                # Rewards from cached array (avoids per-sample image loading)
                all_rewards.append(float(self._reward_array[sample_idx]))

        # Collate observations — after transforms, only model keys remain:
        # state, image, image_mask, actions, tokenized_prompt, tokenized_prompt_mask
        obs_batch: dict[str, Any] = {}
        skip_keys = {"actions"}
        sample_keys = [k for k in all_observations[0].keys() if k not in skip_keys]

        for key in sample_keys:
            values = [obs[key] for obs in all_observations]
            if isinstance(values[0], dict):
                obs_batch[key] = {}
                for subkey in values[0]:
                    subvalues = [v[subkey] for v in values]
                    if hasattr(subvalues[0], "numpy"):
                        obs_batch[key][subkey] = torch.stack(subvalues)
                    else:
                        obs_batch[key][subkey] = np.stack(subvalues)
            elif isinstance(values[0], str):
                obs_batch[key] = values
            elif hasattr(values[0], "numpy"):
                obs_batch[key] = torch.stack(values)
            else:
                obs_batch[key] = np.stack(values)

        # Stack actions
        action_list = []
        for obs in all_observations:
            a = obs["actions"]
            if hasattr(a, "numpy"):
                a = a.numpy()
            action_list.append(np.asarray(a, dtype=np.float32))
        action_batch = np.stack(action_list)

        reward_batch = np.array(all_rewards, dtype=np.float32)

        return obs_batch, action_batch, reward_batch

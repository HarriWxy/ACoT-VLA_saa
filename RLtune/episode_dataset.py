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
from typing import Any, Literal, Sequence, SupportsIndex

import numpy as np
import torch

import openpi.models.model as _model

logger = logging.getLogger(__name__)


def _to_numpy_array(value: Any) -> Any:
    """Convert torch tensors or lists to numpy arrays for model-side collation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _normalize_image_layout(image: Any) -> Any:
    """Normalize image tensors to channels-last layout for the model input pipeline."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[1] in {1, 3} and image.shape[-1] not in {1, 3}:
            return image.permute(0, 2, 3, 1)
        if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[1] not in {1, 3} and image.shape[2] not in {1, 3}:
            return image.permute(1, 2, 0)
        return image

    if isinstance(image, np.ndarray):
        if image.ndim == 4 and image.shape[1] in {1, 3} and image.shape[-1] not in {1, 3}:
            return np.moveaxis(image, 1, -1)
        if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[1] not in {1, 3} and image.shape[2] not in {1, 3}:
            return np.moveaxis(image, 0, -1)

    return image


def _to_torch_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert array-like values to torch tensors safely, including read-only numpy arrays."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value

    if isinstance(value, np.ndarray):
        if not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=dtype)

    return torch.as_tensor(np.array(value, copy=True), dtype=dtype)


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

        hf = None
        if hasattr(raw_ds, "hf_dataset"):
            hf = raw_ds.hf_dataset

        if hf is not None and hasattr(hf, "column_names") and "episode_index" in hf.column_names and "reward" in hf.column_names:
            logger.info("Building episode index from hf_dataset...")
            ep_col = np.array(hf["episode_index"])
            reward_col = np.array(hf["reward"]).flatten()

            # Build episode index: episode_index -> list of sample indices
            self._episode_to_indices: dict[int, list[int]] = defaultdict(list)  # initialize a dictionary to map episode indices to sample indices
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
        else:
            logger.info("No hf_dataset episode metadata found; falling back to one-episode-per-sample indexing.")
            num_frames = len(self._transformed)
            self._episode_to_indices = {idx: [idx] for idx in range(num_frames)}
            self._episode_rewards = {idx: self._aggregate_reward([0.0]) for idx in range(num_frames)}
            self._episode_length = {idx: 1 for idx in range(num_frames)}
            self._episode_prompts = {idx: "" for idx in range(num_frames)}
            self._episode_indices = list(range(num_frames))
            self._reward_array = np.zeros(num_frames, dtype=np.float32)

        logger.info(
            f"EpisodeAwareDataset: {self.total_episodes} episodes, "
            f"{self.total_frames} frames, "
            f"reward_agg={reward_aggregation}"
        )

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        """Return one transformed frame for standard data-loader batching."""
        return self._transformed[index]

    def __len__(self) -> int:
        """Return the number of transformed frames available for batching."""
        return len(self._transformed)

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

        Batch size = len(episode_indices) * frames_per_episode.

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
                sample_obs = self._transformed[sample_idx]
                if isinstance(sample_obs, dict) and isinstance(sample_obs.get("image"), dict):
                    sample_obs = dict(sample_obs)
                    sample_obs["image"] = {
                        key: _normalize_image_layout(value) for key, value in sample_obs["image"].items()
                    }
                    image_masks = sample_obs.get("image_mask")
                    if isinstance(image_masks, dict):
                        sample_obs["image_mask"] = {
                            key: _to_torch_tensor(value, dtype=torch.bool) for key, value in image_masks.items()
                        }
                all_observations.append(sample_obs)

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
                    if all(v is None for v in subvalues):
                        obs_batch[key][subkey] = None
                    elif isinstance(subvalues[0], torch.Tensor):
                        stacked = torch.stack(
                            [v.to(dtype=torch.bool) if key.endswith("_mask") else v for v in subvalues],
                            dim=0,
                        )
                        if key.endswith("_mask") and stacked.ndim > 1 and stacked.shape[-1] == 1:
                            stacked = stacked.squeeze(-1)
                        obs_batch[key][subkey] = stacked
                    elif isinstance(subvalues[0], np.ndarray):
                        stacked = np.stack(subvalues)
                        if key.endswith("_mask"):
                            if stacked.ndim > 1 and stacked.shape[-1] == 1:
                                stacked = np.squeeze(stacked, axis=-1)
                            stacked = stacked.astype(bool)
                        obs_batch[key][subkey] = stacked
                    else:
                        obs_batch[key][subkey] = np.asarray(subvalues)
            elif isinstance(values[0], str):
                obs_batch[key] = values
            else:
                if all(v is None for v in values):
                    obs_batch[key] = None
                else:
                    values = [_to_numpy_array(v) for v in values]
                    if all(isinstance(v, np.ndarray) for v in values):
                        obs_batch[key] = np.stack(values)
                    else:
                        obs_batch[key] = np.asarray(values)

        # Stack actions
        action_list = []
        for obs in all_observations:
            a = obs["actions"]
            action_list.append(_to_numpy_array(a))
        action_batch = np.stack(action_list)

        reward_batch = np.array(all_rewards, dtype=np.float32)

        return _model.Observation.from_dict(obs_batch), action_batch, reward_batch

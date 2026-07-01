"""Reward manager for RL fine-tuning.

Implements reward computation strategies for VLA training.
Following SimpleVLA-RL's approach: simple binary (0/1) outcome rewards.
"""

from __future__ import annotations

import logging

import numpy as np

from .grpo_algo import TrajectoryData
from .rl_config import GRPOConfig

logger = logging.getLogger(__name__)


class RewardManager:
    """Manages reward computation for RL training.

    Following SimpleVLA-RL's philosophy:
    - Use simple binary (0/1) outcome rewards from environment success/failure.
    - No complex reward shaping needed.
    - GRPO's group normalization provides effective learning signal even with sparse rewards.
    - Reward coefficient amplifies the signal.

    Usage:
        reward_fn = RewardManager(config)
        rewards, metrics = reward_fn(rollout_batch)
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.reward_coef = config.reward_coef

    def compute_rewards(self, trajectories: list[TrajectoryData]) -> tuple[np.ndarray, dict[str, float]]:
        """Compute rewards for a batch of trajectories.

        Args:
            trajectories: List of trajectory data from rollouts.

        Returns:
            rewards: (batch_size,) array of scaled rewards.
            metrics: Dictionary of reward statistics.
        """
        # Binary outcome reward: 1 for success, 0 for failure
        raw_rewards = np.array([1.0 if t.success else 0.0 for t in trajectories])

        # Scale by reward coefficient
        scaled_rewards = raw_rewards * self.reward_coef

        # Compute metrics
        metrics = {
            "raw_reward_mean": float(np.mean(raw_rewards)),
            "raw_reward_sum": float(np.sum(raw_rewards)),
            "scaled_reward_mean": float(np.mean(scaled_rewards)),
            "success_rate": float(np.mean(raw_rewards)),
            "num_trajectories": float(len(trajectories)),
        }

        # Per-prompt accuracy
        prompt_indices = np.array([t.prompt_index for t in trajectories])
        unique_prompts = np.unique(prompt_indices)
        prompt_accs = []
        for pid in unique_prompts:
            mask = prompt_indices == pid
            prompt_accs.append(float(np.mean(raw_rewards[mask])))
        if prompt_accs:
            metrics["mean_prompt_accuracy"] = float(np.mean(prompt_accs))
            metrics["min_prompt_accuracy"] = float(np.min(prompt_accs))
            metrics["max_prompt_accuracy"] = float(np.max(prompt_accs))

        return scaled_rewards, metrics

    def __call__(self, trajectories: list[TrajectoryData]) -> tuple[np.ndarray, dict[str, float]]:
        """Alias for compute_rewards."""
        return self.compute_rewards(trajectories)


class DenseRewardManager:
    """Optional dense reward manager with intermediate rewards.

    Can be extended to provide shaped rewards for harder tasks.
    Currently implements the same binary outcome reward,
    but the interface allows for future extensions.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.reward_coef = config.reward_coef

    def compute_rewards(self, trajectories: list[TrajectoryData]) -> tuple[np.ndarray, dict[str, float]]:
        """Compute potentially dense rewards.

        For now, this is identical to RewardManager (binary outcome).
        Override this method to add dense reward components.
        """
        # Binary outcome reward
        raw_rewards = np.array([1.0 if t.success else 0.0 for t in trajectories])

        # Optional: add length bonus (shorter successful episodes get higher reward)
        # for i, traj in enumerate(trajectories):
        #     if traj.success and traj.episode_length > 0:
        #         # Normalize by max steps: shorter = better
        #         length_bonus = 1.0 - (traj.episode_length / self.config.max_episode_steps)
        #         raw_rewards[i] += 0.1 * length_bonus  # Small bonus

        scaled_rewards = raw_rewards * self.reward_coef

        metrics = {
            "raw_reward_mean": float(np.mean(raw_rewards)),
            "scaled_reward_mean": float(np.mean(scaled_rewards)),
            "success_rate": float(np.mean([t.success for t in trajectories])),
        }

        return scaled_rewards, metrics

    def __call__(self, trajectories: list[TrajectoryData]) -> tuple[np.ndarray, dict[str, float]]:
        return self.compute_rewards(trajectories)


def create_reward_manager(config: GRPOConfig, dense: bool = False) -> RewardManager | DenseRewardManager:
    """Factory function to create the appropriate reward manager."""
    if dense:
        return DenseRewardManager(config)
    return RewardManager(config)

"""RL fine-tuning package for ACoT-VLA.

Implements GRPO (Group Relative Policy Optimization) for flow-matching VLA models,
adapted from SimpleVLA-RL's approach.

Key modules:
- rl_config: Configuration dataclasses
- grpo_algo: GRPO algorithm (advantage estimation, policy loss)
- env_runner: Environment rollout collection (online RL)
- reward_manager: Reward computation
- episode_dataset: Episode-aware wrapper for standard LeRobot pipeline (offline RL)
- offline_dataset: Raw offline dataset loader (no transforms)
- train: Main training loop (JAX/Flax)
- train_pytorch: Main training loop (PyTorch with DDP, online)
- train_offline: Offline RL training loop (PyTorch with DDP)
"""

from .env_runner import EnvRunner
from .env_runner import ParallelEnvRunner
from .env_runner import compute_rollout_metrics
from .episode_dataset import EpisodeAwareDataset
from .grpo_algo import RolloutBatch
from .grpo_algo import TrajectoryData
from .grpo_algo import compute_advantages
from .grpo_algo import compute_grpo_advantage
from .grpo_algo import compute_rloo_advantage
from .grpo_algo import compute_weighted_flow_matching_loss
from .grpo_algo import filter_by_accuracy
from .offline_dataset import EpisodeData
from .offline_dataset import OfflineRLDataset
from .offline_dataset import create_offline_dataset
from .reward_manager import RewardManager
from .reward_manager import create_reward_manager
from .rl_config import EnvConfig
from .rl_config import GRPOConfig
from .train_offline_jax import OfflineJAXConfig

__all__ = [
    "EnvConfig",
    "EnvRunner",
    "EpisodeAwareDataset",
    "EpisodeData",
    "GRPOConfig",
    "OfflineRLDataset",
    "ParallelEnvRunner",
    "RewardManager",
    "RolloutBatch",
    "TrajectoryData",
    "compute_advantages",
    "compute_grpo_advantage",
    "compute_rloo_advantage",
    "compute_rollout_metrics",
    "compute_weighted_flow_matching_loss",
    "create_offline_dataset",
    "create_reward_manager",
    "filter_by_accuracy",
]

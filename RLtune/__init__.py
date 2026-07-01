"""RL fine-tuning package for ACoT-VLA.

Implements GRPO (Group Relative Policy Optimization) for flow-matching VLA models,
adapted from SimpleVLA-RL's approach for JAX/Flax.

Key modules:
- rl_config: Configuration dataclasses
- grpo_algo: GRPO algorithm (advantage estimation, policy loss)
- env_runner: Environment rollout collection
- reward_manager: Reward computation
- train: Main training loop
"""

from .env_runner import EnvRunner
from .env_runner import ParallelEnvRunner
from .env_runner import compute_rollout_metrics
from .grpo_algo import RolloutBatch
from .grpo_algo import TrajectoryData
from .grpo_algo import compute_advantages
from .grpo_algo import compute_grpo_advantage
from .grpo_algo import compute_rloo_advantage
from .grpo_algo import compute_weighted_flow_matching_loss
from .grpo_algo import filter_by_accuracy
from .reward_manager import RewardManager
from .reward_manager import create_reward_manager
from .rl_config import EnvConfig
from .rl_config import GRPOConfig

__all__ = [
    "EnvConfig",
    "EnvRunner",
    "GRPOConfig",
    "ParallelEnvRunner",
    "RewardManager",
    "RolloutBatch",
    "TrajectoryData",
    "compute_advantages",
    "compute_grpo_advantage",
    "compute_rloo_advantage",
    "compute_rollout_metrics",
    "compute_weighted_flow_matching_loss",
    "create_reward_manager",
    "filter_by_accuracy",
]

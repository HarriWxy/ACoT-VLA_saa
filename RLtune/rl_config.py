"""RL fine-tuning configuration for ACoT-VLA.

Adapted from SimpleVLA-RL's GRPO approach for JAX/Flax flow-matching models.
"""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) hyperparameters.

    Key design choices (following SimpleVLA-RL):
    - No Critic / Value network: advantages estimated via group-relative normalization.
    - Asymmetric PPO clipping: encourages positive updates more aggressively.
    - High sampling temperature for diverse exploration.
    - Accuracy filtering: only train on "medium difficulty" tasks.
    """

    # ── Core RL hyperparameters ──
    n_samples: int = 8                     # Number of rollouts per task (group size)
    clip_ratio_high: float = 0.28          # Upper PPO clip ratio (asymmetric)
    clip_ratio_low: float = 0.2            # Lower PPO clip ratio
    gamma: float = 1.0                     # Discount factor for returns
    gae_lambda: float = 0.95               # GAE lambda (used if adv_estimator='gae')
    entropy_coeff: float = 0.0             # Entropy regularization coefficient
    kl_coeff: float = 0.0                  # KL penalty coefficient (0 = disabled)
    reward_coef: float = 5.0               # Reward scaling coefficient

    # ── Advantage estimation ──
    adv_estimator: Literal["grpo", "rloo", "reinforce_plus_plus"] = "grpo"

    # ── Sampling / exploration ──
    sampling_temperature: float = 1.6      # High temp for diverse exploration
    num_denoise_steps: int = 10            # ODE steps for flow matching sampling

    # ── Accuracy filtering (Dynamic Sampling) ──
    filter_by_accuracy: bool = True        # Enable accuracy-based sample filtering
    accuracy_lower_bound: float = 0.1      # Min accuracy to include in training
    accuracy_upper_bound: float = 0.9      # Max accuracy to include in training

    # ── Training schedule ──
    total_epochs: int = 100                # Total RL training epochs
    num_train_steps_per_epoch: int = 50    # Steps per epoch
    learning_rate: float = 5e-6            # Learning rate (lower than SFT)
    warmup_steps: int = 100                # LR warmup steps
    grad_clip_norm: float = 1.0            # Gradient clipping norm
    weight_decay: float = 1e-4             # Weight decay

    # ── Mini-batch / memory ──
    mini_batch_size: int = 8               # Mini-batch size for policy updates
    gradient_accumulation_steps: int = 1   # Gradient accumulation steps
    traj_mini_batch_size: int = 16         # Max trajectory steps per micro-batch

    # ── Environment ──
    max_episode_steps: int = 250           # Max steps per episode
    replan_steps: int = 4                  # Replan every N steps
    env_id: str = "srb/sample_collection_visual"
    default_prompt: str = "collect the sample"

    # ── Checkpointing / logging ──
    save_interval: int = 5                 # Save checkpoint every N epochs
    log_interval: int = 1                  # Log metrics every N steps
    eval_interval: int = 5                 # Evaluate every N epochs
    num_eval_episodes: int = 20            # Number of eval episodes

    # ── Model ──
    config_name: str = "srb_train_gemma4"         # Base training config name
    checkpoint_dir: str | None = None      # Path to SFT checkpoint to start from
    action_horizon: int = 16               # Action chunk size
    action_dim: int = 7                    # Action dimensionality

    # ── Distributed ──
    num_workers: int = 4                   # Number of parallel environment workers

    # ── Reproducibility ──
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """Environment configuration for SRB rollout."""
    env_id: str = "srb/sample_collection_visual"
    prompt: str = "collect the sample"
    max_steps: int = 250
    replan_steps: int = 4
    seed: int = 0
    device: str = "cuda:0"
    headless: bool = True
    enable_cameras: bool = True

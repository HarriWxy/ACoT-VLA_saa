"""GRPO algorithm implementation for ACoT-VLA flow-matching models.

Adapted from SimpleVLA-RL's core_algos.py for JAX/Flax continuous action spaces.

Key difference from SimpleVLA-RL:
- SimpleVLA-RL uses token-level log-prob ratios (PPO clipping) because OpenVLA is autoregressive.
- ACoT-VLA uses flow matching (continuous ODE), so we weight the denoising loss by advantages.
- The "policy gradient" signal comes from scaling the MSE flow-matching loss by the advantage.

Mathematical formulation:
  For flow matching, the standard loss is: L = E[||v_θ(x_t, t) - u_t||²]
  With GRPO advantages, we weight this: L_rl = E[A_i · ||v_θ(x_t, t) - u_t||²]
  Where A_i is the group-normalized advantage for trajectory i.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------


def compute_grpo_advantage(
    rewards: np.ndarray,
    prompt_indices: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute GRPO (Group Relative Policy Optimization) advantages.

    For each prompt (task), we have `n_samples` rollouts with binary rewards.
    The advantage is the z-score normalized reward within each group:
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    Args:
        rewards: (batch_size,) array of episode rewards (0 or 1).
        prompt_indices: (batch_size,) array mapping each sample to its prompt/task.
        epsilon: Small constant for numerical stability.

    Returns:
        advantages: (batch_size,) array of normalized advantages.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    unique_prompts = np.unique(prompt_indices)

    for pid in unique_prompts:
        mask = prompt_indices == pid
        group_rewards = rewards[mask]
        if len(group_rewards) == 1:
            # Single sample: no normalization possible
            advantages[mask] = 0.0
        else:
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards)
            advantages[mask] = (group_rewards - mean_r) / (std_r + epsilon)

    return advantages


def compute_rloo_advantage(
    rewards: np.ndarray,
    prompt_indices: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Compute RLOO (Reinforce Leave-One-Out) advantages.

    For each sample, the baseline is the mean reward of the OTHER samples
    in the same group (leave-one-out):
        baseline_i = (sum(R_group) - R_i) / (n_samples - 1)
        A_i = R_i - baseline_i

    Args:
        rewards: (batch_size,) array of episode rewards.
        prompt_indices: (batch_size,) array mapping each sample to its prompt/task.
        n_samples: Number of samples per prompt.

    Returns:
        advantages: (batch_size,) array of RLOO advantages.
    """
    del n_samples  # The observed group size is authoritative after filtering.

    advantages = np.zeros_like(rewards, dtype=np.float32)
    unique_prompts = np.unique(prompt_indices)

    for pid in unique_prompts:
        mask = prompt_indices == pid
        group_rewards = rewards[mask]
        group_indices = np.flatnonzero(mask)
        group_sum = np.sum(group_rewards)
        group_size = len(group_rewards)
        for index, reward in zip(group_indices, group_rewards, strict=True):
            baseline = (group_sum - reward) / max(group_size - 1, 1)
            advantages[index] = reward - baseline

    return advantages


def compute_reinforce_pp_advantage(
    rewards: np.ndarray,
    prompt_indices: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute REINFORCE++ advantages (discounted returns, normalized).

    For outcome-only rewards (single reward at episode end),
    this is equivalent to the normalized reward.

    Args:
        rewards: (batch_size,) array of episode rewards.
        prompt_indices: (batch_size,) array mapping each sample to its prompt/task.
        gamma: Discount factor.

    Returns:
        advantages: (batch_size,) array of normalized advantages.
    """
    # For outcome rewards, returns = rewards
    # Normalize across the entire batch
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-6
    advantages = (rewards - mean_r) / std_r
    return advantages


def compute_advantages(
    rewards: np.ndarray,
    prompt_indices: np.ndarray,
    estimator: str = "grpo",
    n_samples: int = 8,
    gamma: float = 1.0,
) -> np.ndarray:
    """Dispatch advantage computation based on estimator type."""
    if estimator == "grpo":
        return compute_grpo_advantage(rewards, prompt_indices)
    if estimator == "rloo":
        return compute_rloo_advantage(rewards, prompt_indices, n_samples)
    if estimator == "reinforce_plus_plus":
        return compute_reinforce_pp_advantage(rewards, prompt_indices, gamma)
    raise ValueError(f"Unknown advantage estimator: {estimator}")


# ---------------------------------------------------------------------------
# Accuracy filtering (Dynamic Sampling)
# ---------------------------------------------------------------------------


def filter_by_accuracy(
    rewards: np.ndarray,
    prompt_indices: np.ndarray,
    lower_bound: float = 0.1,
    upper_bound: float = 0.9,
) -> tuple[np.ndarray, dict[str, float]]:
    """Filter samples based on per-prompt accuracy.

    Only keeps prompts where the accuracy is between lower_bound and upper_bound.
    - Prompts with accuracy > upper_bound are "too easy" (already learned).
    - Prompts with accuracy < lower_bound are "too hard" (no signal).

    Args:
        rewards: (batch_size,) array of episode rewards (0 or 1).
        prompt_indices: (batch_size,) array mapping each sample to its prompt/task.
        lower_bound: Minimum accuracy to include.
        upper_bound: Maximum accuracy to include.

    Returns:
        mask: (batch_size,) boolean array indicating which samples to keep.
        metrics: Dictionary with filtering statistics.
    """
    unique_prompts = np.unique(prompt_indices)
    prompt_acc = {}
    for pid in unique_prompts:
        mask = prompt_indices == pid
        prompt_acc[pid] = np.mean(rewards[mask])

    mask = np.zeros(len(rewards), dtype=bool)
    for pid, acc in prompt_acc.items():
        if lower_bound <= acc <= upper_bound:
            mask[prompt_indices == pid] = True

    total_prompts = len(unique_prompts)
    kept_prompts = sum(1 for acc in prompt_acc.values() if lower_bound <= acc <= upper_bound)

    metrics = {
        "total_prompts": float(total_prompts),
        "kept_prompts": float(kept_prompts),
        "filter_rate": 1.0 - kept_prompts / max(total_prompts, 1),
        "mean_accuracy": float(np.mean(list(prompt_acc.values()))) if prompt_acc else 0.0,
    }

    return mask, metrics


# ---------------------------------------------------------------------------
# Policy loss for flow-matching models
# ---------------------------------------------------------------------------


def compute_advantage_weights(
    advantages: jnp.ndarray,
    *,
    temperature: float = 1.0,
    max_weight: float = 20.0,
) -> jnp.ndarray:
    """Convert relative advantages into bounded, normalized BC weights."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1")

    max_log_weight = float(np.log(max_weight))
    log_weights = jnp.clip(advantages / temperature, -max_log_weight, max_log_weight)
    weights = jnp.exp(log_weights)
    return weights / jnp.mean(weights)


def compute_weighted_flow_matching_loss(
    per_sample_loss: jnp.ndarray,
    advantages: jnp.ndarray,
    eos_mask: jnp.ndarray | None = None,
    *,
    temperature: float = 1.0,
    max_weight: float = 20.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute a bounded advantage-weighted flow matching surrogate.

    Offline demonstrations do not provide the old-policy likelihood ratio
    required by PPO/GRPO. We therefore use non-negative, normalized weights so
    negative advantages reduce a demonstration's influence rather than making
    the non-negative flow-matching loss unbounded below.

    Args:
        per_sample_loss: (batch_size,) per-sample flow matching loss.
        advantages: (batch_size,) normalized advantages from GRPO.
        eos_mask: Optional mask for valid timesteps.

    Returns:
        loss: Scalar RL loss.
        metrics: Dictionary of training metrics.
    """
    del eos_mask
    weights = compute_advantage_weights(advantages, temperature=temperature, max_weight=max_weight)
    weighted_loss = jnp.mean(jax.lax.stop_gradient(weights) * per_sample_loss)

    metrics = {
        "rl_loss": weighted_loss,
        "mean_advantage": jnp.mean(advantages),
        "mean_advantage_weight": jnp.mean(weights),
        "max_advantage_weight": jnp.max(weights),
        "mean_per_sample_loss": jnp.mean(per_sample_loss),
        "positive_advantage_ratio": jnp.mean((advantages > 0).astype(jnp.float32)),
    }

    return weighted_loss, metrics


def compute_ppo_style_flow_matching_loss(
    old_loss: jnp.ndarray,
    new_loss: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.2,
) -> tuple[jnp.ndarray, dict[str, float]]:
    """Compute PPO-style clipped loss adapted for flow matching.

    Instead of log-prob ratios, we use loss ratios:
        ratio = old_loss / (new_loss + eps)
        L = max(-A*ratio, -A*clip(ratio, 1-low, 1+high))

    This is a heuristic adaptation — the ratio of losses approximates
    the ratio of likelihoods for Gaussian denoising.

    Args:
        old_loss: (batch_size,) per-sample loss from old policy.
        new_loss: (batch_size,) per-sample loss from new policy.
        advantages: (batch_size,) normalized advantages.
        clip_ratio_high: Upper clip bound.
        clip_ratio_low: Lower clip bound.

    Returns:
        loss: Scalar clipped policy loss.
        metrics: Dictionary of training metrics.
    """
    eps = 1e-8
    # Loss ratio: if new loss is lower, ratio > 1 (improvement)
    ratio = old_loss / (new_loss + eps)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    loss = jnp.mean(jnp.maximum(pg_losses1, pg_losses2))

    clip_frac = jnp.mean((pg_losses2 > pg_losses1).astype(jnp.float32))

    metrics = {
        "ppo_loss": float(loss),
        "clip_fraction": float(clip_frac),
        "mean_ratio": float(jnp.mean(ratio)),
        "mean_advantage": float(jnp.mean(advantages)),
    }

    return loss, metrics


# ---------------------------------------------------------------------------
# KL penalty (optional, for reference policy)
# ---------------------------------------------------------------------------


def compute_kl_penalty(
    current_params: Any,
    ref_params: Any,
    coef: float = 0.0,
) -> float:
    """Compute KL divergence penalty between current and reference policy.

    For flow matching models, we approximate KL via parameter-space L2 distance
    (a rough proxy when explicit log-likelihoods are unavailable).

    Args:
        current_params: Current model parameters (pytree).
        ref_params: Reference model parameters (pytree).
        coef: KL penalty coefficient.

    Returns:
        kl_penalty: Scalar KL penalty (0 if coef is 0).
    """
    if coef <= 0:
        return 0.0

    # Flatten parameters and compute L2 distance
    flat_current = jax.tree_util.tree_leaves(current_params)
    flat_ref = jax.tree_util.tree_leaves(ref_params)

    kl = 0.0
    for c, r in zip(flat_current, flat_ref):
        if hasattr(c, "shape") and hasattr(r, "shape"):
            kl += jnp.sum((c - r) ** 2)

    return coef * kl


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrajectoryData:
    """Data from a single rollout trajectory."""
    observation: Any          # Observation dict/array
    actions: Any              # Actions predicted by the model
    coarse_actions: Any | None = None  # For ACoT-VLA
    reward: float = 0.0       # Episode reward (0 or 1)
    success: bool = False     # Whether the task was completed
    episode_length: int = 0   # Number of steps in the episode
    prompt_index: int = 0     # Which task/prompt this belongs to
    per_step_loss: Any | None = None  # Per-step flow matching loss (for weighting)


@dataclasses.dataclass
class RolloutBatch:
    """Batch of trajectories for RL training."""
    trajectories: list[TrajectoryData]

    @property
    def rewards(self) -> np.ndarray:
        return np.array([t.reward for t in self.trajectories])

    @property
    def prompt_indices(self) -> np.ndarray:
        return np.array([t.prompt_index for t in self.trajectories])

    @property
    def successes(self) -> np.ndarray:
        return np.array([t.success for t in self.trajectories])

    def __len__(self) -> int:
        return len(self.trajectories)

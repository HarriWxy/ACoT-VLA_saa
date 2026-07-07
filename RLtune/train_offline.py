"""Offline RL fine-tuning using pre-collected demonstration data.

Implements GRPO (Group Relative Policy Optimization) for flow-matching VLA
models using offline data from LeRobot parquet files, replacing the online
EnvRunner rollout with direct episode sampling from the dataset.

Key differences from online RL (train_pytorch.py):
- No environment rollouts — data comes from pre-collected episodes
- Episode rewards extracted from parquet reward field
- Episode grouping for GRPO by sampling from offline dataset
- Uses standard data pipeline (create_torch_dataset + transform_dataset)
  to ensure consistent normalization / tokenization / padding with SFT

Usage:
    # Single GPU:
    python -m RLtune.train_offline

    # Multi-GPU (DDP):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        -m RLtune.train_offline

    # Or using the launch script:
    bash RLtune/run_offline_train.sh
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import platform
import sys
import time
from typing import Literal

# Ensure project root is in sys.path
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import tqdm

from openpi.models_pytorch.action_proj_head import ActionProjectionHead
from openpi.models_pytorch.action_proj_head import ResidualActionProjectionHead
from openpi.models_pytorch.action_proj_head import freeze_model_train_head_only
from openpi.models_pytorch.action_proj_head import get_param_groups as get_proj_param_groups
from openpi.models_pytorch.action_proj_head import load_projection_head
from openpi.models_pytorch.action_proj_head import save_projection_head
from openpi.models_pytorch.action_proj_head import unfreeze_model_finetune
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from RLtune.episode_dataset import EpisodeAwareDataset
from RLtune.grpo_algo import compute_advantages
from RLtune.grpo_algo import filter_by_accuracy
from RLtune.rl_config import ProjectionHeadConfig
from RLtune.train_pytorch import build_model
from RLtune.train_pytorch import cleanup_ddp
from RLtune.train_pytorch import compute_rl_loss
from RLtune.train_pytorch import get_model
from RLtune.train_pytorch import init_logging
from RLtune.train_pytorch import jax_tree_to_device
from RLtune.train_pytorch import load_rl_checkpoint
from RLtune.train_pytorch import make_lr_schedule
from RLtune.train_pytorch import save_rl_checkpoint
from RLtune.train_pytorch import set_seed
from RLtune.train_pytorch import setup_ddp

# ---------------------------------------------------------------------------
# Offline GRPO Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OfflineGRPOConfig:
    """Configuration for offline GRPO training.

    Extends the online GRPO setup with offline-specific parameters
    (data directory, reward aggregation strategy, etc.).
    """

    # ── Core RL hyperparameters ──
    n_samples: int = 8
    clip_ratio_high: float = 0.28
    clip_ratio_low: float = 0.2
    gamma: float = 1.0
    entropy_coeff: float = 0.0
    kl_coeff: float = 0.0
    reward_coef: float = 5.0

    # ── Advantage estimation ──
    adv_estimator: Literal["grpo", "rloo", "reinforce_plus_plus"] = "grpo"

    # ── Accuracy filtering ──
    filter_by_accuracy: bool = True
    accuracy_lower_bound: float = 0.1
    accuracy_upper_bound: float = 0.9

    # ── Training schedule ──
    total_epochs: int = 100
    num_train_steps_per_epoch: int = 50
    learning_rate: float = 5e-6
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4

    # ── Offline data settings ──
    data_dir: str = "dataset/srb_tracking"
    reward_aggregation: Literal["sum", "mean", "last", "binary_sum"] = "sum"
    reward_threshold: float = 0.0
    frames_per_episode: int = 2  # Frames sampled per episode per training step

    # ── Model / checkpoint ──
    config_name: str = "srb_train_gemma4"
    checkpoint_dir: str | None = None
    action_horizon: int = 16
    action_dim: int = 37

    # ── Projection Head ──
    projection_head: ProjectionHeadConfig = dataclasses.field(default_factory=ProjectionHeadConfig)

    # ── Logging / checkpointing ──
    save_interval: int = 5
    log_interval: int = 1

    # ── Reproducibility ──
    seed: int = 42


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_loop(rl_config: OfflineGRPOConfig):
    """Main offline RL training loop using GRPO on pre-collected data.

    Pipeline per epoch:
    1. Sample a group of episodes from the offline dataset
    2. Compute episode rewards and GRPO advantages
    3. Optionally filter by accuracy (medium difficulty)
    4. For each training step:
       a. Sample frames from episodes
       b. Apply SRBInputs transform → (observation, actions)
       c. Compute advantage-weighted flow matching loss
       d. Backward + optimizer step
    5. Log metrics and save checkpoints
    """
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(rl_config.seed, local_rank)

    # ── Base config ──
    base_config = _config.get_config(rl_config.config_name)
    if rl_config.checkpoint_dir:
        base_config = dataclasses.replace(
            base_config,
            checkpoint_dir=pathlib.Path(rl_config.checkpoint_dir),
        )

    rl_ckpt_dir = base_config.checkpoint_dir / "rl_grpo"
    resuming = rl_ckpt_dir.exists() and any(rl_ckpt_dir.glob("epoch_*"))

    writer = None
    if is_main:
        # Slightly modified tensorboard init for offline
        log_dir = base_config.checkpoint_dir / "rl_grpo_offline" / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        hparams_text = (
            f"## Offline RL Config\n```json\n{dataclasses.asdict(rl_config)}\n```\n\n"
            f"## Base Config\n{rl_config.config_name}"
        )
        writer.add_text("hyperparameters", hparams_text)
        logging.info(f"TensorBoard logs -> {log_dir}")

    # ── Load offline dataset via standard pipeline ──
    # Uses create_torch_dataset + transform_dataset for consistent
    # normalization / SRBInputs / tokenization / padding with SFT training
    data_config = base_config.data.create(base_config.assets_dirs, base_config.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, base_config.model.action_horizon, base_config.model)
    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=False)

    dataset = EpisodeAwareDataset(
        transformed_dataset,
        reward_aggregation=rl_config.reward_aggregation,
        reward_threshold=rl_config.reward_threshold,
    )

    if is_main:
        logging.info(f"Offline dataset: {dataset.total_episodes} episodes, {dataset.total_frames} frames")
        reward_stats = dataset.get_reward_stats()
        logging.info(f"Reward stats: {reward_stats}")

    # ── Build model ──
    model = build_model(base_config, device)

    # Load SFT weights if starting fresh
    if not resuming and base_config.pytorch_weight_path is not None:
        model_path = os.path.join(base_config.pytorch_weight_path, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(get_model(model), model_path)
            logging.info(f"Loaded SFT weights from {model_path}")

    # DDP wrapping
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # ── Projection head ──
    projection_head = None
    proj_cfg = rl_config.projection_head
    _base_model = get_model(model)
    model_action_dim = getattr(_base_model, "action_dim", None) or _base_model.config.action_dim

    if proj_cfg.enabled and rl_config.action_dim != model_action_dim:
        head_cls = ResidualActionProjectionHead if proj_cfg.head_type == "residual" else ActionProjectionHead
        projection_head = head_cls(
            actual_action_dim=rl_config.action_dim,
            model_action_dim=model_action_dim,
            hidden_dim=proj_cfg.hidden_dim,
            num_layers=proj_cfg.num_layers,
            activation=proj_cfg.activation,
            use_layer_norm=proj_cfg.use_layer_norm,
            residual_scale=proj_cfg.residual_scale,
        ).to(device)

        if proj_cfg.head_checkpoint_path and os.path.exists(proj_cfg.head_checkpoint_path):
            load_projection_head(projection_head, proj_cfg.head_checkpoint_path)

        if proj_cfg.training_stage == "stage1":
            freeze_model_train_head_only(get_model(model), projection_head)
        elif proj_cfg.training_stage == "stage2":
            unfreeze_model_finetune(
                get_model(model), projection_head, unfreeze_action_heads=True, unfreeze_backbone=False
            )
        elif proj_cfg.training_stage == "stage3":
            unfreeze_model_finetune(
                get_model(model), projection_head, unfreeze_action_heads=True, unfreeze_backbone=True
            )

        if is_main:
            total_p = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in projection_head.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(
                p.numel() for p in projection_head.parameters()
            )
            logging.info(
                f"Projection head: {proj_cfg.training_stage} | "
                f"Trainable: {trainable_p / 1e6:.2f}M / {total_p / 1e6:.2f}M"
            )

    # ── Optimizer ──
    if projection_head is not None:
        param_groups = get_proj_param_groups(
            get_model(model), projection_head, lr_model=proj_cfg.lr_model, lr_head=proj_cfg.lr_head
        )
        rl_lr = max(pg["lr"] for pg in param_groups)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=rl_config.weight_decay)
    else:
        rl_lr = rl_config.learning_rate
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=rl_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=rl_config.weight_decay
        )

    # LR schedule
    total_rl_steps = rl_config.total_epochs * rl_config.num_train_steps_per_epoch
    scheduler = make_lr_schedule(
        optimizer, warmup_steps=rl_config.warmup_steps, peak_lr=rl_lr, decay_steps=total_rl_steps, end_lr=rl_lr * 0.1
    )

    # Resume
    start_epoch = 0
    best_reward = -float("inf")
    global_step = 0
    if resuming:
        start_epoch, best_reward = load_rl_checkpoint(
            model, optimizer, rl_ckpt_dir, device, projection_head=projection_head
        )
        for _ in range(start_epoch * rl_config.num_train_steps_per_epoch):
            scheduler.step()
        global_step = start_epoch * rl_config.num_train_steps_per_epoch

    # RNG for episode sampling
    rng = np.random.default_rng(rl_config.seed + local_rank)

    # ── Training loop ──
    if is_main:
        logging.info(f"Running offline RL on: {platform.node()}")
        logging.info(f"Config: {dataclasses.asdict(rl_config)}")
        logging.info(f"Starting from epoch {start_epoch}")

    model.train()
    if projection_head is not None:
        projection_head.train()

    for epoch in range(start_epoch, rl_config.total_epochs):
        epoch_start = time.time()

        if is_main:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Epoch {epoch + 1}/{rl_config.total_epochs}")
            logging.info(f"{'=' * 60}")

        # ── Phase 1: Sample episode group for GRPO ──
        groups = dataset.sample_episodes_for_grpo(
            n_groups=1,
            n_samples_per_group=rl_config.n_samples,
            rng=rng,
        )
        group = groups[0]

        # ── Phase 2: Compute episode rewards ──
        rewards = np.array([dataset.get_episode_reward(ep) for ep in group], dtype=np.float32)
        prompt_indices = np.zeros(len(group), dtype=np.int64)  # All same task

        if is_main:
            logging.info(
                f"Episode rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
                f"min={rewards.min():.4f}, max={rewards.max():.4f}"
            )

        # ── Phase 3: Filter by accuracy (optional) ──
        if rl_config.filter_by_accuracy:
            binary_rewards = np.array([1.0 if dataset.get_episode_success(ep) else 0.0 for ep in group])
            mask, filter_metrics = filter_by_accuracy(
                binary_rewards,
                prompt_indices,
                lower_bound=rl_config.accuracy_lower_bound,
                upper_bound=rl_config.accuracy_upper_bound,
            )
            if is_main:
                logging.info(f"Filter: {filter_metrics}")

            if mask.sum() == 0:
                if is_main:
                    logging.warning("All episodes filtered out! Skipping epoch.")
                continue

            filtered_indices = np.where(mask)[0]
            group = [group[i] for i in filtered_indices]
            rewards = rewards[filtered_indices]
            prompt_indices = prompt_indices[filtered_indices]

        # ── Phase 4: Compute GRPO advantages ──
        advantages = compute_advantages(
            rewards,
            prompt_indices,
            estimator=rl_config.adv_estimator,
            n_samples=rl_config.n_samples,
            gamma=rl_config.gamma,
        )

        if is_main:
            logging.info(f"Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")

        # ── Phase 5: Policy update ──
        step_losses = []
        pbar = (
            tqdm.tqdm(range(rl_config.num_train_steps_per_epoch), desc=f"Epoch {epoch + 1}", disable=not is_main)
            if is_main
            else range(rl_config.num_train_steps_per_epoch)
        )

        for step in pbar:
            # Sample frames from episodes — observations are already fully
            # transformed (normalized, tokenized, padded) by the standard pipeline
            obs_batch, actions_np, _ = dataset.sample_batch_from_episodes(
                group,
                frames_per_episode=rl_config.frames_per_episode,
                rng=rng,
            )

            batch_size = actions_np.shape[0]

            # Move to device
            observation = jax_tree_to_device(obs_batch, device)
            actions = torch.from_numpy(actions_np).float().to(device)

            # Expand advantages to match frame-level batch
            adv_per_frame = np.repeat(advantages, rl_config.frames_per_episode)
            if len(adv_per_frame) < batch_size:
                repeats = (batch_size // len(adv_per_frame)) + 1
                adv_per_frame = np.tile(adv_per_frame, repeats)
            adv_tensor = torch.from_numpy(adv_per_frame[:batch_size]).float().to(device)

            # Project actions to model space if needed
            model_actions = actions
            if projection_head is not None:
                model_actions = projection_head.project_to_model(actions)

            # Forward + loss
            loss, step_metrics = compute_rl_loss(
                model,
                observation,
                model_actions,
                adv_tensor,
                clip_advantage=3.0,
            )

            # Backward
            loss.backward()

            all_params = list(model.parameters())
            if projection_head is not None:
                all_params += list(projection_head.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=rl_config.grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            step_losses.append(step_metrics)

            # Progress bar
            if is_main and isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "adv": f"{adv_tensor.mean().item():.3f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

            # Step-level logging
            if is_main and step % rl_config.log_interval == 0 and step > 0:
                writer.add_scalar("step/rl_loss", step_metrics.get("rl_loss", loss.item()), global_step)
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                writer.add_scalar("step/grad_norm", gn, global_step)
                writer.add_scalar("step/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                for key, value in step_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"step/{key}", value, global_step)

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        avg_metrics = {}
        if step_losses:
            avg_metrics = {k: float(np.mean([s[k] for s in step_losses if k in s])) for k in step_losses[0]}

        all_metrics = {
            **avg_metrics,
            "epoch": epoch + 1,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
            "episode_reward_mean": float(np.mean(rewards)),
            "episode_reward_std": float(np.std(rewards)),
            "n_episodes": float(len(group)),
        }

        if is_main:
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"epoch/{key}", value, epoch)
            logging.info(f"Epoch {epoch + 1} summary: {all_metrics}")

        # ── Save checkpoint ──
        if (epoch + 1) % rl_config.save_interval == 0:
            if is_main:
                logging.info(f"Saving checkpoint at epoch {epoch + 1}...")
            save_rl_checkpoint(
                model,
                optimizer,
                epoch + 1,
                global_step,
                rl_config,
                base_config,
                best_reward,
                is_main,
                projection_head=projection_head,
            )

        # Track best model by mean episode reward
        mean_ep_reward = float(np.mean(rewards))
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            if is_main:
                logging.info(f"New best reward: {best_reward:.4f}")
                best_dir = base_config.checkpoint_dir / "rl_grpo_offline" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_model(get_model(model), best_dir / "model.safetensors")
                if projection_head is not None:
                    save_projection_head(projection_head, best_dir / "projection_head.safetensors")

    # Final save
    if is_main:
        logging.info("Training complete. Saving final checkpoint...")
    save_rl_checkpoint(
        model,
        optimizer,
        rl_config.total_epochs,
        global_step,
        rl_config,
        base_config,
        best_reward,
        is_main,
        projection_head=projection_head,
    )

    if is_main:
        writer.add_scalar("best_reward", best_reward, global_step)
        writer.close()
        logging.info(f"Best reward achieved: {best_reward:.4f}")

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for offline RL training."""
    init_logging()
    import tyro

    rl_config = tyro.cli(OfflineGRPOConfig)
    train_loop(rl_config)


if __name__ == "__main__":
    main()

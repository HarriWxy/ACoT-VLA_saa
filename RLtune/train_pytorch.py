"""PyTorch RL fine-tuning training script for ACoT-VLA.

Implements GRPO (Group Relative Policy Optimization) for flow-matching VLA models
using PyTorch with DDP multi-GPU support.

Adapted from the JAX version (RLtune/train.py), using advantage-weighted flow
matching loss instead of PPO log-prob ratio clipping.

Training loop:
    for epoch in range(total_epochs):
        1. Collect rollouts: run policy in environment, get trajectories
        2. Compute rewards: binary success/failure from environment
        3. Filter by accuracy: only train on medium-difficulty tasks
        4. Compute advantages: GRPO group normalization
        5. Update policy: weighted flow matching loss on demonstration data
        6. Log metrics and save checkpoints

Usage:
    # Single GPU:
    python -m RLtune.train_pytorch --config_name srb_train \
        --checkpoint_dir path/to/sft_ckpt

    # Multi-GPU (DDP):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        -m RLtune.train_pytorch --config_name srb_train \
        --checkpoint_dir path/to/sft_ckpt

    # Or using the launch script:
    bash RLtune/run_rl_train_pytorch.sh
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import pathlib
import platform
import shutil
import sys
import time
from typing import Any

# Ensure project root is in sys.path so 'RLtune' can be imported
# regardless of whether we run as `python -m RLtune.train_pytorch`
# or `python RLtune/train_pytorch.py`.
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.acot_vla_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

from RLtune.env_runner import EnvRunner
from RLtune.env_runner import compute_rollout_metrics
from RLtune.grpo_algo import compute_advantages
from RLtune.grpo_algo import filter_by_accuracy
from RLtune.reward_manager import create_reward_manager
from RLtune.rl_config import GRPOConfig

# Projection head for action dim bridging
from openpi.models_pytorch.action_proj_head import (
    ResidualActionProjectionHead,
    ActionProjectionHead,
    freeze_model_train_head_only,
    unfreeze_model_finetune,
    get_param_groups as get_proj_param_groups,
    save_projection_head,
    load_projection_head,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp():
    """Initialize distributed training. (torch) """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def get_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get the underlying model, unwrapping DDP if needed."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def build_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    """Build the PyTorch model from config.

    Supports both PI0 and ACoT-VLA models. The model type is determined by
    checking if the config has ACoT-specific attributes (adopt_explicit_action_reasoner,
    adopt_implicit_action_reasoner, coarse_action_horizon, etc.).

    Args:
        config: Training configuration.
        device: Target device.

    Returns:
        Model instance (PI0Pytorch or ACOT_VLAPytorch).
    """
    # Detect ACoT-VLA config
    is_acot = hasattr(config.model, "adopt_explicit_action_reasoner") or \
              hasattr(config.model, "coarse_action_horizon") or \
              hasattr(config.model, "adopt_implicit_action_reasoner")

    if is_acot:
        # Build ACoT-VLA model
        from openpi.models_pytorch.acot_vla_pytorch import ACOTConfigPytorch, ACOT_VLAPytorch

        model_cfg = ACOTConfigPytorch(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            coarse_action_horizon=getattr(config.model, "coarse_action_horizon", 50),
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            coarse_action_expert_variant=getattr(config.model, "coarse_action_expert_variant", "gemma_300m"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            adopt_explicit_action_reasoner=getattr(config.model, "adopt_explicit_action_reasoner", False),
            adopt_implicit_action_reasoner=getattr(config.model, "adopt_implicit_action_reasoner", False),
            query_based_implicit_extractor=getattr(config.model, "query_based_implicit_extractor", False),
            attention_pooling_implicit_extractor=getattr(config.model, "attention_pooling_implicit_extractor", False),
            downsample_based_implicit_extractor=getattr(config.model, "downsample_based_implicit_extractor", False),
            use_one_step_inference=getattr(config.model, "use_one_step_inference", False),
            exploration_std=getattr(config.model, "exploration_std", 0.0),
            self_consistency_loss_scale=getattr(config.model, "self_consistency_loss_scale", 0.0),
            sc_midpoint_samples=getattr(config.model, "sc_midpoint_samples", 1),
        )

        model = ACOT_VLAPytorch(model_cfg).to(device)
        logging.info(f"Built ACoT-VLA model: {model_cfg}")

    else:
        # Build PI0 model
        if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
            model_cfg = openpi.models.pi0_config.Pi0Config(
                dtype=config.pytorch_training_precision,
                action_dim=config.model.action_dim,
                action_horizon=config.model.action_horizon,
                max_token_len=config.model.max_token_len,
                paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
                action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
                pi05=getattr(config.model, "pi05", False),
            )
        else:
            model_cfg = config.model
            object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")

    return model


# ---------------------------------------------------------------------------
# TensorBoard helpers
# ---------------------------------------------------------------------------


def init_tensorboard(
    rl_config: GRPOConfig,
    base_config: _config.TrainConfig,
    *,
    resuming: bool,
) -> SummaryWriter:
    """Initialize TensorBoard logging for RL training.

    Returns a SummaryWriter instance. Logs are written to
    ``<checkpoint_dir>/rl_grpo/tensorboard/``.
    """
    log_dir = base_config.checkpoint_dir / "rl_grpo" / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # Log hyperparameters as text
    hparams_text = (
        f"## RL Config\n```json\n{dataclasses.asdict(rl_config)}\n```\n\n"
        f"## Base Config\n```json\n{dataclasses.asdict(base_config)}\n```"
    )
    writer.add_text("hyperparameters", hparams_text)

    logging.info(f"TensorBoard logs -> {log_dir}")
    return writer


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def save_rl_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    rl_config: GRPOConfig,
    base_config: _config.TrainConfig,
    best_success_rate: float,
    is_main: bool,
    data_config=None,
    projection_head: ActionProjectionHead | None = None,
):
    """Save RL fine-tuning checkpoint."""
    if not is_main:
        return

    ckpt_dir = base_config.checkpoint_dir / "rl_grpo" / f"epoch_{epoch}"
    tmp_dir = base_config.checkpoint_dir / "rl_grpo" / f"tmp_epoch_{epoch}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    model_to_save = get_model(model)
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")

    # Save projection head if present
    if projection_head is not None:
        save_projection_head(projection_head, tmp_dir / "projection_head.safetensors")

    # Save optimizer state
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

    # Save metadata
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "best_success_rate": best_success_rate,
        "rl_config": dataclasses.asdict(rl_config),
        "timestamp": time.time(),
    }
    torch.save(metadata, tmp_dir / "metadata.pt")

    # Save norm stats if available
    if data_config is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_dir / "assets" / data_config.asset_id, norm_stats)

    # Atomic move
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)

    logging.info(f"Saved RL checkpoint at epoch {epoch} -> {ckpt_dir}")


def load_rl_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    projection_head: ActionProjectionHead | None = None,
) -> tuple[int, float]:
    """Load RL checkpoint and return (epoch, best_success_rate)."""
    if not checkpoint_dir.exists():
        return 0, 0.0

    # Find the latest epoch checkpoint
    epoch_dirs = [
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith("epoch_")
    ]
    if not epoch_dirs:
        return 0, 0.0

    latest_dir = max(epoch_dirs, key=lambda d: int(d.name.split("_")[1]))

    # Load model
    model_path = latest_dir / "model.safetensors"
    if model_path.exists():
        model_to_load = get_model(model)
        safetensors.torch.load_model(model_to_load, model_path, device=str(device))
        logging.info(f"Loaded model from {model_path}")

    # Load projection head if present
    if projection_head is not None:
        head_path = latest_dir / "projection_head.safetensors"
        if head_path.exists():
            load_projection_head(projection_head, head_path)
            logging.info(f"Loaded projection head from {head_path}")

    # Load optimizer
    optimizer_path = latest_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
        logging.info(f"Loaded optimizer from {optimizer_path}")

    # Load metadata
    metadata_path = latest_dir / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location=device, weights_only=False)
        epoch = metadata.get("epoch", 0)
        best_success_rate = metadata.get("best_success_rate", 0.0)
        logging.info(f"Resumed from epoch {epoch}, best_success_rate={best_success_rate:.4f}")
        return epoch, best_success_rate

    return 0, 0.0


# ---------------------------------------------------------------------------
# RL Training Step
# ---------------------------------------------------------------------------


def compute_rl_loss(
    model: torch.nn.Module,
    observation: Any,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    coarse_actions: torch.Tensor | None = None,
    clip_advantage: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantage-weighted flow matching loss for RL fine-tuning.

    The standard flow matching loss is:
        L_fm = ||v_θ(x_t, t) - u_t||²

    With GRPO advantages, we weight each sample's loss:
        L_rl = mean(A_i * L_fm_i)

    This encourages the model to:
    - Better fit trajectories that led to success (positive advantage).
    - Move away from trajectories that led to failure (negative advantage).

    Args:
        model: The model (PI0Pytorch or ACOT_VLAPytorch).
        observation: Observation dict from data loader.
        actions: Target actions tensor [B, action_horizon, action_dim].
        advantages: Per-sample advantages [B].
        coarse_actions: Coarse target actions [B, coarse_action_horizon, action_dim].
            Required for ACoT-VLA with explicit action reasoner.
        clip_advantage: Clipping range for advantages.

    Returns:
        loss: Scalar RL loss.
        metrics: Dictionary of training metrics.
    """
    # Check if this is an ACoT-VLA model
    from openpi.models_pytorch.acot_vla_pytorch import ACOT_VLAPytorch

    is_acot = isinstance(model, ACOT_VLAPytorch) or \
              (hasattr(model, "module") and isinstance(model.module, ACOT_VLAPytorch))

    if is_acot:
        # ACoT-VLA forward pass returns scalar loss (sum of coarse + fine)
        loss = model(observation, actions, coarse_actions)

        # Add self-consistency loss if configured (OFP path compression)
        _base = get_model(model)
        sc_scale = getattr(_base, 'self_consistency_loss_scale', 0.0)
        sc_samples = getattr(_base, 'sc_midpoint_samples', 1)
        if sc_scale > 0 and coarse_actions is not None and _base.training:
            sc_loss = _base.compute_self_consistency_loss(
                observation, actions, coarse_actions,
                num_midpoint_samples=sc_samples,
            )
            loss = loss + sc_scale * sc_loss

        # For ACoT-VLA, we weight the loss by mean advantage
        # (since we can't get per-sample loss without modifying the forward pass)
        clipped_adv = advantages.clamp(-clip_advantage, clip_advantage)
        mean_adv = clipped_adv.mean()

        # Scale loss by advantage sign to encourage/discourage
        if mean_adv > 0:
            rl_loss = loss * mean_adv
        else:
            rl_loss = loss * (1.0 + mean_adv)  # Reduce loss when negative advantage

        metrics = {
            "rl_loss": rl_loss.item(),
            "base_loss": loss.item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "positive_advantage_ratio": (advantages > 0).float().mean().item(),
        }
        if sc_scale > 0:
            metrics["sc_loss"] = sc_loss.item() if isinstance(sc_loss, torch.Tensor) else sc_loss

    else:
        # PI0 forward pass: returns per-element MSE loss [B, action_horizon, action_dim] # obs:dict  actions: tensor
        per_element_loss = model(observation, actions)

        # Reduce to per-sample loss [B]
        if per_element_loss.ndim > 1:
            per_sample_loss = per_element_loss.mean(dim=list(range(1, per_element_loss.ndim)))
        else:
            per_sample_loss = per_element_loss

        # Clip advantages for stability
        clipped_adv = advantages.clamp(-clip_advantage, clip_advantage)

        # Weight loss by advantages
        rl_loss = (clipped_adv * per_sample_loss).mean()

        metrics = {
            "rl_loss": rl_loss.item(),
            "mean_per_sample_loss": per_sample_loss.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
            "positive_advantage_ratio": (advantages > 0).float().mean().item(),
        }

    return rl_loss, metrics


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    end_lr: float,
):
    """Create a cosine decay LR schedule with warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return (init_lr + (peak_lr - init_lr) * step / warmup_steps) / peak_lr
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return (end_lr + (peak_lr - end_lr) * cos) / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_loop(rl_config: GRPOConfig):
    """Main RL training loop using PyTorch DDP.

    Orchestrates the full GRPO training pipeline:
    1. Initialize model from SFT checkpoint
    2. Start policy server for rollouts
    3. For each epoch:
       a. Collect rollouts in environment
       b. Compute rewards and advantages
       c. Optionally filter by accuracy
       d. Update policy with advantage-weighted loss
       e. Log metrics and save checkpoints
    """
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(rl_config.seed, local_rank)

    # Get base training config
    base_config = _config.get_config(rl_config.config_name)  # srb_train

    # Override checkpoint dir if specified
    if rl_config.checkpoint_dir:
        base_config = dataclasses.replace(
            base_config,
            checkpoint_dir=pathlib.Path(rl_config.checkpoint_dir),
        )

    # Initialize checkpoint directory
    rl_ckpt_dir = base_config.checkpoint_dir / "rl_grpo"
    resuming = rl_ckpt_dir.exists() and any(rl_ckpt_dir.glob("epoch_*"))

    writer = None
    if is_main:
        writer = init_tensorboard(rl_config, base_config, resuming=resuming)

    # Build data loader for offline RL (using SFT demonstration data)
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    loader = _data_loader.create_data_loader(
        base_config,
        framework="pytorch",
        shuffle=True,
        skip_norm_stats=True,
    )

    # Build model
    model = build_model(base_config, device)

    # Load SFT weights if starting fresh
    if not resuming and base_config.pytorch_weight_path is not None:
        model_path = os.path.join(base_config.pytorch_weight_path, "model.safetensors")
        if os.path.exists(model_path):
            safetensors.torch.load_model(get_model(model), model_path)
            logging.info(f"Loaded SFT weights from {model_path}")

    # Setup DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # ── Projection head setup ──
    projection_head = None
    proj_cfg = rl_config.projection_head
    _base_model = get_model(model)
    model_action_dim = getattr(_base_model, 'action_dim', None) or _base_model.config.action_dim  # typically 32

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

        # Load pretrained head weights if available
        if proj_cfg.head_checkpoint_path and os.path.exists(proj_cfg.head_checkpoint_path):
            load_projection_head(projection_head, proj_cfg.head_checkpoint_path)

        # Apply freeze strategy based on training stage
        if proj_cfg.training_stage == "stage1":
            freeze_model_train_head_only(get_model(model), projection_head)
        elif proj_cfg.training_stage == "stage2":
            unfreeze_model_finetune(
                get_model(model), projection_head,
                unfreeze_action_heads=True, unfreeze_backbone=False,
            )
        elif proj_cfg.training_stage == "stage3":
            unfreeze_model_finetune(
                get_model(model), projection_head,
                unfreeze_action_heads=True, unfreeze_backbone=True,
            )

        if is_main:
            total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in projection_head.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in projection_head.parameters())
            logging.info(
                f"Projection head: {proj_cfg.training_stage} mode | "
                f"Trainable: {trainable/1e6:.2f}M / {total_params/1e6:.2f}M total"
            )

    # ── Build optimizer ──
    if projection_head is not None:
        # Use param groups with different LR for head vs model
        param_groups = get_proj_param_groups(
            get_model(model), projection_head,
            lr_model=proj_cfg.lr_model,
            lr_head=proj_cfg.lr_head,
        )
        # Override model LR based on training stage
        if proj_cfg.training_stage == "stage1":
            # Stage 1: model is frozen, only head trains
            pass  # get_proj_param_groups already handles this
        elif proj_cfg.training_stage == "stage2":
            pass  # model LR already set
        elif proj_cfg.training_stage == "stage3":
            # Add backbone group with lower LR
            for g in param_groups:
                if g["name"] == "model":
                    # Split into backbone vs action heads
                    backbone_params = []
                    action_head_params = []
                    action_head_names = [
                        "action_in_proj", "action_out_proj",
                        "time_mlp_in", "time_mlp_out",
                        "state_proj", "action_time_mlp_in", "action_time_mlp_out",
                    ]
                    for p in g["params"]:
                        # Check param name to decide group
                        # We can't easily get the name here, so just use the model LR
                        action_head_params.append(p)
                    # For stage3, all model params use model LR
                    g["lr"] = proj_cfg.lr_model

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=rl_config.weight_decay,
        )
        # Use the highest LR for the schedule
        rl_lr = max(pg["lr"] for pg in param_groups)
    else:
        rl_lr = rl_config.learning_rate
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=rl_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=rl_config.weight_decay,
        )

    # LR schedule
    total_rl_steps = rl_config.total_epochs * rl_config.num_train_steps_per_epoch
    scheduler = make_lr_schedule(
        optimizer,
        warmup_steps=rl_config.warmup_steps,
        peak_lr=rl_lr,
        decay_steps=total_rl_steps,
        end_lr=rl_lr * 0.1,
    )

    # Load RL checkpoint if resuming
    start_epoch = 0
    best_success_rate = 0.0
    global_step = 0
    if resuming:
        start_epoch, best_success_rate = load_rl_checkpoint(
            model, optimizer, rl_ckpt_dir, device,
            projection_head=projection_head,
        )
        # Advance scheduler to the correct step
        for _ in range(start_epoch * rl_config.num_train_steps_per_epoch):
            scheduler.step()
        global_step = start_epoch * rl_config.num_train_steps_per_epoch

    # Initialize RL components
    reward_fn = create_reward_manager(rl_config)
    env_runner = EnvRunner(rl_config)

    # Data iterator for offline RL
    data_iter = iter(loader)

    # ── Main training loop ──
    if is_main:
        logging.info(f"Running RL fine-tuning on: {platform.node()}")
        logging.info(f"RL Config: {dataclasses.asdict(rl_config)}")
        logging.info(f"Starting RL training for {rl_config.total_epochs} epochs from epoch {start_epoch}")

    model.train()
    if projection_head is not None:
        projection_head.train()
    start_time = time.time()

    for epoch in range(start_epoch, rl_config.total_epochs):
        epoch_start = time.time()

        if is_main:
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch + 1}/{rl_config.total_epochs}")
            logging.info(f"{'='*60}")

        # Set epoch for DDP sampler
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)

        # ── Phase 1: Collect rollouts ──
        if is_main:
            logging.info("Phase 1: Collecting rollouts...")

        trajectories = env_runner.collect_rollouts()

        # Compute rollout metrics
        rollout_metrics = compute_rollout_metrics(trajectories)
        if is_main:
            logging.info(f"Rollout metrics: {rollout_metrics}")

        # ── Phase 2: Compute rewards and advantages ──
        if is_main:
            logging.info("Phase 2: Computing rewards and advantages...")

        rewards, reward_metrics = reward_fn(trajectories)
        prompt_indices = np.array([t.prompt_index for t in trajectories])

        # ── Phase 3: Filter by accuracy (optional) ──
        if rl_config.filter_by_accuracy:
            raw_rewards = np.array([t.reward for t in trajectories])
            mask, filter_metrics = filter_by_accuracy(
                raw_rewards,
                prompt_indices,
                lower_bound=rl_config.accuracy_lower_bound,
                upper_bound=rl_config.accuracy_upper_bound,
            )
            if is_main:
                logging.info(f"Filter metrics: {filter_metrics}")

            if mask.sum() == 0:
                if is_main:
                    logging.warning("All samples filtered out! Skipping this epoch.")
                continue

            filtered_indices = np.where(mask)[0]
            filtered_rewards = rewards[filtered_indices]
            filtered_prompt_indices = prompt_indices[filtered_indices]
        else:
            filtered_rewards = rewards
            filtered_prompt_indices = prompt_indices

        # ── Phase 4: Compute GRPO advantages ──
        if is_main:
            logging.info("Phase 3: Computing GRPO advantages...")

        advantages = compute_advantages(
            filtered_rewards,
            filtered_prompt_indices,
            estimator=rl_config.adv_estimator,
            n_samples=rl_config.n_samples,
            gamma=rl_config.gamma,
        )

        # ── Phase 5: Policy update ──
        if is_main:
            logging.info("Phase 4: Updating policy...")

        step_losses = []
        pbar = (
            tqdm.tqdm(
                range(rl_config.num_train_steps_per_epoch),
                desc=f"Epoch {epoch+1}",
                disable=not is_main,
            )
            if is_main
            else range(rl_config.num_train_steps_per_epoch)
        )

        for step in pbar:
            # Get next batch from data loader
            try:
                observation, actions = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                observation, actions = next(data_iter)

            # Move to device
            observation = jax_tree_to_device(observation, device)
            actions = actions.to(torch.float32).to(device)

            # Extract coarse_actions from observation if available (ACoT-VLA)
            coarse_actions = None
            if hasattr(observation, 'coarse_actions'):
                coarse_actions = observation.coarse_actions.to(torch.float32).to(device)
            elif isinstance(observation, dict) and "coarse_actions" in observation:
                coarse_actions = observation["coarse_actions"].to(torch.float32).to(device)

            # Convert advantages to tensor and broadcast to batch size
            batch_size = actions.shape[0]
            adv_tensor = torch.from_numpy(advantages[:batch_size]).float().to(device)
            if len(adv_tensor) < batch_size:
                # Tile advantages if batch is larger than available advantages
                repeats = (batch_size // len(adv_tensor)) + 1
                adv_tensor = adv_tensor.repeat(repeats)[:batch_size]

            # Project actions to model space if using projection head
            model_actions = actions
            if projection_head is not None:
                model_actions = projection_head.project_to_model(actions)

            # Forward + loss
            loss, step_metrics = compute_rl_loss(
                model, observation, model_actions, adv_tensor,
                coarse_actions=coarse_actions,
                clip_advantage=3.0,
            )

            # Backward
            loss.backward()

            # Gradient clipping (include projection head params)
            all_params = list(model.parameters())
            if projection_head is not None:
                all_params += list(projection_head.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, max_norm=rl_config.grad_clip_norm
            )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            step_losses.append(step_metrics)

            # Update progress bar
            if is_main and isinstance(pbar, tqdm.tqdm):
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "adv": f"{adv_tensor.mean().item():.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

            # Log step metrics to TensorBoard
            if is_main and step % rl_config.log_interval == 0 and step > 0:
                avg_loss = np.mean([s["rl_loss"] for s in step_losses[-rl_config.log_interval:]])
                logging.info(
                    f"  Step {step}: rl_loss={avg_loss:.4f}, "
                    f"grad_norm={grad_norm:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                # Write step-level scalars to TensorBoard
                writer.add_scalar("step/rl_loss", avg_loss, global_step)
                writer.add_scalar("step/grad_norm", grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, global_step)
                writer.add_scalar("step/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                # Log projection head LR if using param groups
                if projection_head is not None and len(optimizer.param_groups) > 1:
                    for pg in optimizer.param_groups:
                        if pg.get("name") == "projection_head":
                            writer.add_scalar("step/lr_head", pg["lr"], global_step)
                for key, value in step_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"step/{key}", value, global_step)

        # ── Logging ──
        epoch_time = time.time() - epoch_start

        # Aggregate step metrics
        if step_losses:
            avg_metrics = {
                k: np.mean([s[k] for s in step_losses])
                for k in step_losses[0].keys()
            }
        else:
            avg_metrics = {}

        all_metrics = {
            **rollout_metrics,
            **reward_metrics,
            **avg_metrics,
            "epoch": epoch + 1,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        # Add advantage statistics
        all_metrics["advantage_mean"] = float(np.mean(advantages))
        all_metrics["advantage_std"] = float(np.std(advantages))

        if is_main:
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"epoch/{key}", value, epoch)
                elif isinstance(value, str):
                    writer.add_text(f"epoch/{key}", value, epoch)
            logging.info(f"Epoch {epoch + 1} summary: {all_metrics}")

        # ── Save checkpoint ──
        if (epoch + 1) % rl_config.save_interval == 0:
            if is_main:
                logging.info(f"Saving checkpoint at epoch {epoch + 1}...")
            save_rl_checkpoint(
                model, optimizer, epoch + 1, global_step,
                rl_config, base_config, best_success_rate,
                is_main, data_config, # 
                projection_head=projection_head,
            )

        # ── Track best model ──
        if rollout_metrics.get("success_rate", 0.0) > best_success_rate:
            best_success_rate = rollout_metrics["success_rate"]
            if is_main:
                logging.info(f"New best success rate: {best_success_rate:.4f}")
                # Save best model separately
                best_dir = base_config.checkpoint_dir / "rl_grpo" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_model(get_model(model), best_dir / "model.safetensors")
                if projection_head is not None:
                    save_projection_head(projection_head, best_dir / "projection_head.safetensors")

    # Final save
    if is_main:
        logging.info("Training complete. Saving final checkpoint...")
    save_rl_checkpoint(
        model, optimizer, rl_config.total_epochs, global_step,
        rl_config, base_config, best_success_rate,
        is_main,
        projection_head=projection_head,
    )

    if is_main:
        writer.add_scalar("best_success_rate", best_success_rate, global_step)
        writer.close()
        logging.info(f"Best success rate achieved: {best_success_rate:.4f}")

    cleanup_ddp()


# ---------------------------------------------------------------------------
# Utility: convert JAX-style nested structure to PyTorch device
# ---------------------------------------------------------------------------


def jax_tree_to_device(tree: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested structure to the given device.

    Handles the observation structure returned by the PyTorch data loader,
    which may be a nested dict or a dataclass with tensor fields.
    """
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    elif isinstance(tree, dict):
        return {k: jax_tree_to_device(v, device) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        moved = [jax_tree_to_device(v, device) for v in tree]
        return type(tree)(moved) if isinstance(tree, tuple) else moved
    elif hasattr(tree, "__dataclass_fields__"):
        # Handle dataclass-like objects (e.g., Observation)
        field_values = {}
        for field_name in tree.__dataclass_fields__:
            value = getattr(tree, field_name)
            field_values[field_name] = jax_tree_to_device(value, device)
        return type(tree)(**field_values)
    elif hasattr(tree, "to"):
        # Generic tensor-like object with .to() method
        return tree.to(device)
    else:
        return tree


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for PyTorch RL training."""
    init_logging()

    import tyro
    rl_config = tyro.cli(GRPOConfig)
    train_loop(rl_config)


if __name__ == "__main__":
    main()

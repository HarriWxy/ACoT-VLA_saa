"""Action Projection Head for mapping between model internal dim and actual action dim.

When the actual action dimension exceeds the model's internal action_dim (typically 32),
this module provides a learnable projection that bridges the gap.

Architecture:
    actual_action_dim ──(down/up proj)──> model_action_dim (32) ──(model)──> model_action_dim (32) ──(up/down proj)──> actual_action_dim

Two modes:
    - "bottleneck": actual_dim > model_dim. Compress actions into the model's latent space.
    - "expand": actual_dim < model_dim. Expand model output to actual action space (alternative to zero-padding).

Usage:
    head = ActionProjectionHead(actual_action_dim=64, model_action_dim=32)

    # Training: project target actions to model space, compute loss in model space
    target_actions_proj = head.project_to_model(target_actions)  # [B, H, 64] -> [B, H, 32]
    loss = model(observation, target_actions_proj)

    # Inference: project model output back to actual action space
    model_actions = model.sample_actions(observation)  # [B, H, 32]
    actual_actions = head.project_to_actual(model_actions)  # [B, H, 64]
"""

from __future__ import annotations

import pathlib
import logging

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActionProjectionHead(nn.Module):
    """Learnable projection between actual action dim and model internal action dim.

    Supports both directions:
    - project_to_model: actual_dim → model_dim (for training targets)
    - project_to_actual: model_dim → actual_dim (for inference output)

    The projection is a small MLP with residual connections for stability.
    """

    def __init__(
        self,
        actual_action_dim: int,
        model_action_dim: int = 32,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: str = "silu",
        use_layer_norm: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.actual_action_dim = actual_action_dim
        self.model_action_dim = model_action_dim
        self.residual_scale = residual_scale

        if hidden_dim is None:
            # Hidden dim is the larger of the two, capped at 128
            hidden_dim = min(max(actual_action_dim, model_action_dim) * 2, 128)

        # Determine mode
        if actual_action_dim > model_action_dim:
            self.mode = "bottleneck"
            logger.info(
                f"ActionProjectionHead: bottleneck mode ({actual_action_dim} → {model_action_dim}). "
                f"Actions will be compressed into the model's latent space."
            )
        elif actual_action_dim < model_action_dim:
            self.mode = "expand"
            logger.info(
                f"ActionProjectionHead: expand mode ({actual_action_dim} → {model_action_dim}). "
                f"Actions will be expanded (replaces zero-padding)."
            )
        else:
            self.mode = "identity"
            logger.info("ActionProjectionHead: identity mode (dims match). No projection needed.")
            return

        # Build projection networks
        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]

        # actual_dim → model_dim (for training targets)
        layers_to_model = []
        in_dim = actual_action_dim
        for i in range(num_layers - 1):
            layers_to_model.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            if use_layer_norm:
                layers_to_model.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers_to_model.append(nn.Linear(in_dim, model_action_dim))
        self.proj_to_model = nn.Sequential(*layers_to_model)

        # model_dim → actual_dim (for inference output)
        layers_to_actual = []
        in_dim = model_action_dim
        for i in range(num_layers - 1):
            layers_to_actual.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            if use_layer_norm:
                layers_to_actual.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers_to_actual.append(nn.Linear(in_dim, actual_action_dim))
        self.proj_to_actual = nn.Sequential(*layers_to_actual)

        # Initialize with small weights for stable training start
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with small values for stability."""
        for module in [self.proj_to_model, self.proj_to_actual]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def project_to_model(self, actions: torch.Tensor) -> torch.Tensor:
        """Project actual actions to model's internal action space.

        Args:
            actions: [B, H, actual_action_dim]

        Returns:
            projected: [B, H, model_action_dim]
        """
        if self.mode == "identity":
            return actions
        return self.proj_to_model(actions)

    def project_to_actual(self, model_actions: torch.Tensor) -> torch.Tensor:
        """Project model output back to actual action space.

        Args:
            model_actions: [B, H, model_action_dim]

        Returns:
            projected: [B, H, actual_action_dim]
        """
        if self.mode == "identity":
            return model_actions
        return self.proj_to_actual(model_actions)

    def forward(self, actions: torch.Tensor, direction: str = "to_model") -> torch.Tensor:
        """Forward pass.

        Args:
            actions: Input action tensor.
            direction: "to_model" for actual→model, "to_actual" for model→actual.

        Returns:
            Projected actions.
        """
        if direction == "to_model":
            return self.project_to_model(actions)
        elif direction == "to_actual":
            return self.project_to_actual(actions)
        else:
            raise ValueError(f"Unknown direction: {direction}")


class ResidualActionProjectionHead(ActionProjectionHead):
    """Projection head with residual connections.

    For bottleneck mode: learns a residual on top of a linear compression.
    For expand mode: learns a residual on top of a linear expansion.

    This is more stable than pure MLP projection because the residual
    connection provides a good initialization (near-identity mapping).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "identity":
            return

        # Linear shortcut for residual
        self.linear_to_model = nn.Linear(self.actual_action_dim, self.model_action_dim)
        self.linear_to_actual = nn.Linear(self.model_action_dim, self.actual_action_dim)

        # Initialize linear shortcuts as the primary mapping
        nn.init.xavier_uniform_(self.linear_to_model.weight)
        nn.init.zeros_(self.linear_to_model.bias)
        nn.init.xavier_uniform_(self.linear_to_actual.weight)
        nn.init.zeros_(self.linear_to_actual.bias)

        # Re-initialize MLP residuals to near-zero
        for module in [self.proj_to_model, self.proj_to_actual]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def project_to_model(self, actions: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return actions
        # Linear shortcut + small MLP residual
        return self.linear_to_model(actions) + self.residual_scale * self.proj_to_model(actions)

    def project_to_actual(self, model_actions: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return model_actions
        # Linear shortcut + small MLP residual
        return self.linear_to_actual(model_actions) + self.residual_scale * self.proj_to_actual(model_actions)


def wrap_model_with_projection(
    model: torch.nn.Module,
    actual_action_dim: int,
    model_action_dim: int = 32,
    head_type: str = "residual",
    **kwargs,
) -> tuple[torch.nn.Module, ActionProjectionHead]:
    """Create a projection head and return it alongside the model.

    This is a convenience function that doesn't modify the model itself.
    The caller is responsible for using the projection head in the training
    and inference loops.

    Args:
        model: The PI0/ACoT model (unchanged).
        actual_action_dim: The actual action dimension of the environment.
        model_action_dim: The model's internal action dimension (default 32).
        head_type: "mlp" or "residual".
        **kwargs: Additional kwargs for the projection head.

    Returns:
        (model, projection_head) tuple.
    """
    if actual_action_dim == model_action_dim:
        logger.info("Action dims match, no projection head needed.")
        return model, None

    cls = ResidualActionProjectionHead if head_type == "residual" else ActionProjectionHead
    head = cls(
        actual_action_dim=actual_action_dim,
        model_action_dim=model_action_dim,
        **kwargs,
    )
    return model, head


# ---------------------------------------------------------------------------
# Freeze utilities
# ---------------------------------------------------------------------------


def freeze_model_train_head_only(
    model: torch.nn.Module,
    projection_head: ActionProjectionHead,
) -> int:
    """Freeze the entire main model, only keep projection head trainable.

    This is the recommended first stage of training:
        Stage 1: freeze_model_train_head_only() → train projection head only
        Stage 2: unfreeze_model_finetune() → fine-tune the whole model with small LR

    Args:
        model: The PI0/ACoT model to freeze.
        projection_head: The projection head to keep trainable.

    Returns:
        Number of trainable parameters.
    """
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # Ensure projection head is trainable
    for param in projection_head.parameters():
        param.requires_grad_(True)

    trainable = sum(p.numel() for p in projection_head.parameters())
    total_model = sum(p.numel() for p in model.parameters())
    total_head = sum(p.numel() for p in projection_head.parameters())

    logger.info(
        f"Frozen model ({total_model / 1e6:.1f}M params). "
        f"Trainable projection head: {trainable / 1e3:.1f}K params "
        f"({100 * trainable / (total_model + total_head):.4f}% of total)."
    )
    return trainable


def unfreeze_model_finetune(
    model: torch.nn.Module,
    projection_head: ActionProjectionHead | None = None,
    unfreeze_action_heads: bool = True,
    unfreeze_backbone: bool = False,
) -> dict[str, int]:
    """Unfreeze model parameters for fine-tuning (Stage 2).

    Args:
        model: The PI0/ACoT model.
        projection_head: Optional projection head (stays trainable).
        unfreeze_action_heads: If True, unfreeze action_in_proj, action_out_proj, etc.
        unfreeze_backbone: If True, unfreeze the entire backbone (VLM + expert).

    Returns:
        Dict with trainable param counts for each component.
    """
    stats = {}

    if unfreeze_backbone:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad_(True)
        stats["backbone"] = sum(p.numel() for p in model.parameters())
        logger.info("Unfroze entire backbone.")
    elif unfreeze_action_heads:
        # Only unfreeze action-related layers
        action_head_names = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        action_count = 0
        for name, param in model.named_parameters():
            if any(head_name in name for head_name in action_head_names):
                param.requires_grad_(True)
                action_count += param.numel()
            else:
                param.requires_grad_(False)
        stats["action_heads"] = action_count
        logger.info(f"Unfroze action heads: {action_count / 1e3:.1f}K params.")

    if projection_head is not None:
        for param in projection_head.parameters():
            param.requires_grad_(True)
        stats["projection_head"] = sum(p.numel() for p in projection_head.parameters())

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if projection_head is not None:
        total_trainable += sum(p.numel() for p in projection_head.parameters())
    stats["total_trainable"] = total_trainable

    logger.info(f"Total trainable: {total_trainable / 1e6:.2f}M params.")
    return stats


def get_param_groups(
    model: torch.nn.Module,
    projection_head: ActionProjectionHead | None = None,
    lr_model: float = 5e-6,
    lr_head: float = 1e-4,
) -> list[dict]:
    """Create optimizer parameter groups with different learning rates.

    Typically:
    - Projection head: higher LR (learning from scratch)
    - Model action heads: medium LR (fine-tuning)
    - Model backbone: low LR or frozen

    Args:
        model: The PI0/ACoT model.
        projection_head: Optional projection head.
        lr_model: Learning rate for model trainable params.
        lr_head: Learning rate for projection head.

    Returns:
        List of param group dicts for torch.optim.AdamW.
    """
    param_groups = []

    # Model trainable params
    model_trainable = [p for p in model.parameters() if p.requires_grad]
    if model_trainable:
        param_groups.append({
            "params": model_trainable,
            "lr": lr_model,
            "name": "model",
        })

    # Projection head params (always trainable if provided)
    if projection_head is not None:
        head_params = [p for p in projection_head.parameters() if p.requires_grad]
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr_head,
                "name": "projection_head",
            })

    for g in param_groups:
        count = sum(p.numel() for p in g["params"])
        logger.info(f"  Param group '{g['name']}': {count / 1e3:.1f}K params, lr={g['lr']}")

    return param_groups


def save_projection_head(
    head: ActionProjectionHead,
    path: str | pathlib.Path,
    model: torch.nn.Module | None = None,
):
    """Save projection head (and optionally model action heads) to disk.

    Args:
        head: The projection head to save.
        path: Output path for the safetensors file.
        model: Optional model to save action heads from.
    """
    import pathlib
    import safetensors.torch

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {}
    # Save projection head
    for k, v in head.state_dict().items():
        state[f"projection_head.{k}"] = v

    # Optionally save model action heads
    if model is not None:
        action_head_names = [
            "action_in_proj", "action_out_proj",
            "time_mlp_in", "time_mlp_out",
            "state_proj", "action_time_mlp_in", "action_time_mlp_out",
        ]
        for name, param in model.named_parameters():
            if any(h in name for h in action_head_names):
                state[f"model.{name}"] = param.data

    safetensors.torch.save_file(state, str(path))
    logger.info(f"Saved projection head to {path} ({len(state)} tensors)")


def load_projection_head(
    head: ActionProjectionHead,
    path: str | pathlib.Path,
    model: torch.nn.Module | None = None,
):
    """Load projection head (and optionally model action heads) from disk.

    Args:
        head: The projection head to load into.
        path: Path to the safetensors file.
        model: Optional model to load action heads into.
    """
    import pathlib
    import safetensors.torch

    path = pathlib.Path(path)
    state = safetensors.torch.load_file(str(path))

    # Load projection head
    head_state = {}
    for k, v in state.items():
        if k.startswith("projection_head."):
            head_state[k[len("projection_head."):]] = v
    head.load_state_dict(head_state)
    logger.info(f"Loaded projection head from {path}")

    # Optionally load model action heads
    if model is not None:
        model_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                model_state[k[len("model."):]] = v
        if model_state:
            model.load_state_dict(model_state, strict=False)
            logger.info(f"Loaded {len(model_state)} model action head tensors")

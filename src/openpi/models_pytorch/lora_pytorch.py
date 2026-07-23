"""PyTorch LoRA implementation for openpi models.

Provides custom LoRA Linear layers that can be injected into HuggingFace
transformer models (Gemma4, PaliGemma) for parameter-efficient fine-tuning.

Design mirrors the JAX LoRA in `openpi.models.lora`:
- LoRA A initialized with normal(stddev=0.01)
- LoRA B initialized with zeros
- Scaling: alpha / rank (standard) or alpha / sqrt(rank) (RSLoRA)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """LoRA configuration for PyTorch layers."""

    rank: int
    alpha: float = 1.0
    dropout: float = 0.0
    rslora: bool = False

    @property
    def scaling(self) -> float:
        if self.rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters.

    Wraps a frozen base linear layer and adds trainable LoRA A/B matrices.
    Output = base_linear(x) + (x @ A^T @ B^T) * scaling
    """

    def __init__(self, base_linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base_linear = base_linear
        self.config = config

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

        # LoRA A: (rank, in_features) — initialized with normal
        self.lora_A = nn.Parameter(torch.empty(config.rank, in_features))
        nn.init.normal_(self.lora_A, std=0.01)

        # LoRA B: (out_features, rank) — initialized with zeros
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        result = self.base_linear(x)

        # LoRA forward (trainable) — cast to input dtype to handle bf16/fp16 inputs
        lora_dtype = x.dtype
        lora_out = self.dropout(x) @ self.lora_A.T.to(lora_dtype)  # (..., rank)
        lora_out = lora_out @ self.lora_B.T.to(lora_dtype)          # (..., out_features)
        result = result + lora_out * self.config.scaling

        return result

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base linear for inference (no overhead)."""
        merged_weight = self.base_linear.weight.data + (self.lora_B @ self.lora_A) * self.config.scaling
        self.base_linear.weight.data.copy_(merged_weight)
        return self.base_linear


def inject_lora_linear(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: list[str] | None = None,
) -> list[str]:
    """Inject LoRA adapters into matching nn.Linear modules of a model.

    Args:
        model: The model to inject LoRA into.
        config: LoRA configuration.
        target_modules: List of module name patterns to match. If None, targets
            all nn.Linear layers. Supports substring matching.

    Returns:
        List of module names that were wrapped with LoRA.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    injected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name.split(".")[-1] for target in target_modules):
            continue

        # Get parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with LoRA version
        lora_layer = LoRALinear(module, config)
        setattr(parent, parts[-1], lora_layer)
        injected.append(name)

    return injected


def freeze_non_lora_params(model: nn.Module) -> int:
    """Freeze all parameters except LoRA adapters.

    Args:
        model: The model with LoRA adapters injected.

    Returns:
        Number of trainable parameters.
    """
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable_count += param.numel()
        else:
            param.requires_grad_(False)
    return trainable_count


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA parameters for saving (adapter weights)."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}

"""Gemma4 + Expert Model for PI0 robotics policy.

Uses Gemma 4 native multimodal architecture:
- Gemma4VisionModel + Gemma4MultimodalEmbedder for visual processing
  (replaces PaliGemma's SigLIP vision tower)
- Gemma4TextModel for VLM language model
- Gemma4TextModel as action expert (shares attention with VLM)

Key differences from Gemma 2 based gemma_pytorch.py:
- 4 LayerNorms per layer (input, post_attn, pre_ffn, post_ffn) + layer_scalar
- Q/K/V norms (RMSNorm on Q/K, norm-without-scale on V)
- Dual RoPE: sliding (theta=10k, full rotation), full (theta=1M, partial=0.25)
- Dual attention masks: full causal vs sliding window
- No _gated_residual — simple residual addition
- PLE (per-layer embeddings) — disabled for initial integration
- adarms support for PI05 mode via post-init layer modification
"""

from __future__ import annotations

import contextlib
import logging
import types
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoImageProcessor
from transformers import Gemma4ForCausalLM
from transformers import Gemma4ForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma4.modeling_gemma4 import repeat_kv


def _get_weight(module: nn.Module) -> torch.Tensor:
    """获取模块权重, 兼容 LoRALinear (权重在 base_linear.weight)。"""
    return module.base_linear.weight if hasattr(module, "base_linear") else module.weight


def _scaled_dot_product_attention(query, key, value, attention_mask):
    """Run SDPA while preserving the fused CUDA kernel for grouped queries."""
    query_heads = query.shape[1]
    key_heads = key.shape[1]
    if query_heads % key_heads != 0:
        raise ValueError(f"Query heads ({query_heads}) must be divisible by key heads ({key_heads})")

    if query_heads != key_heads:
        # On the tested CUDA backend, enable_gqa forces SDPA onto a less memory-
        # efficient path. Expand once here so the regular call can use Flash
        # or memory-efficient attention.
        key = repeat_kv(key, query_heads // key_heads)
        value = repeat_kv(value, query_heads // key_heads)

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=1.0,
    )


def _make_layer_types(num_layers: int, pattern: int = 6) -> list[str]:
    """Generate Gemma4 layer_types with last layer forced to full_attention.

    Args:
        num_layers: Total number of decoder layers.
        pattern: Sliding window pattern period (default 6 → 5:1 sliding:full).
    """
    layer_types = [
        "sliding_attention" if ((i + 1) % pattern) != 0 else "full_attention"
        for i in range(num_layers)
    ]
    # Ensure last layer is full_attention (required by Gemma4)
    layer_types[-1] = "full_attention"
    return layer_types


def _enable_adarms_on_layer(layer, cond_dim: int):
    """Add adarms (adaptive RMSNorm) support to a Gemma4 decoder layer.

    Modifies the layer's norm layers in-place to support conditioning,
    and replaces the forward method to pass adarms_cond through.
    The layer structure remains identical for weight loading compatibility.
    """
    for norm_name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        norm = getattr(layer, norm_name)
        dim = norm.weight.shape[0]
        norm.cond_dim = cond_dim
        norm.adarms_dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(norm.adarms_dense.weight)
        nn.init.zeros_(norm.adarms_dense.bias)
        norm.adarms_enabled = True
        # Replace with adarms-aware forward
        norm.forward = types.MethodType(_adarms_norm_forward, norm)


def _adarms_norm_forward(self, hidden_states, cond=None):
    """Forward pass for Gemma4RMSNorm with optional adarms conditioning."""
    dtype = hidden_states.dtype
    # RMSNorm normalization (same as Gemma4RMSNorm._norm)
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    normed = hidden_states * torch.pow(mean_squared, -0.5)

    if cond is None or not getattr(self, "adarms_enabled", False):
        # Standard path
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(hidden_states)

    # Adaptive path: scale, shift, gate
    modulation = self.adarms_dense(cond.to(dtype=self.adarms_dense.weight.dtype))
    if hidden_states.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normed = normed * (1 + scale.float()) + shift.float()
    return normed.to(dtype), gate.to(dtype)


def _gated_residual(x, y, gate):
    """Gated residual: x + y * gate if gate is provided, else x + y."""
    if gate is None:
        return x + y
    return x + y * gate


def _patch_gemma4_decoder_layer_adarms():
    """Allow Gemma4 decoder layers to consume an optional adarms condition."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    if getattr(Gemma4TextDecoderLayer, "_openpi_adarms_patched", False):
        return

    def _apply_norm(norm_fn, hidden_states, cond):
        if cond is None:
            out = norm_fn(hidden_states)
        else:
            out = norm_fn(hidden_states, cond=cond)
        return out[0] if isinstance(out, tuple) else out

    def _patched_forward(
        self,
        hidden_states,
        per_layer_input=None,
        shared_kv_states=None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        cond = kwargs.pop("cond", None)
        if cond is None and "adarms_cond" in kwargs:
            cond = kwargs.pop("adarms_cond")

        residual = hidden_states
        hidden_states = _apply_norm(self.input_layernorm, hidden_states, cond)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = _apply_norm(self.post_attention_layernorm, hidden_states, cond)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = _apply_norm(self.pre_feedforward_layernorm, hidden_states, cond)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = _apply_norm(self.post_feedforward_layernorm_1, hidden_states, cond)

            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = _apply_norm(self.pre_feedforward_layernorm_2, hidden_states_flat, cond)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = _apply_norm(self.post_feedforward_layernorm_2, hidden_states_2, cond)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = _apply_norm(self.post_feedforward_layernorm, hidden_states, cond)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = _apply_norm(self.post_per_layer_input_norm, hidden_states, cond)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states

    Gemma4TextDecoderLayer.forward = _patched_forward
    Gemma4TextDecoderLayer._openpi_adarms_patched = True  # noqa: SLF001


_patch_gemma4_decoder_layer_adarms()


class Gemma4WithExpertModel(nn.Module):
    """Gemma-4 native multimodal + action expert model.

    Architecture:
    - VLM: Gemma-4 text decoder (native, no PaliGemma dependency)
    - Vision: Gemma-4 VisionModel + MultimodalEmbedder (replaces SigLIP)
    - Action Expert: Gemma-4 text decoder (smaller, shares joint attention with VLM)

    Joint attention: Q/K/V from both models are concatenated along the
    sequence dimension, with RoPE applied per-model based on layer_type,
    then a single attention operation is applied.
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        gemma4_model_path: str = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        elif isinstance(use_adarms, bool):
            use_adarms = [use_adarms, use_adarms]
        super().__init__()

        self.use_adarms = use_adarms

        # --- Load native Gemma-4 vision components from pretrained checkpoint ---
        if gemma4_model_path is None:
            raise ValueError(
                "gemma4_model_path is required. Pass the path to the Gemma-4 "
                "checkpoint (e.g. './models/gemma-4-E2B') to load the native "
                "vision tower and multimodal embedder."
            )

        logging.info(f"Loading Gemma-4 vision components from {gemma4_model_path}...")
        full_model = Gemma4ForConditionalGeneration.from_pretrained(
            gemma4_model_path, dtype=torch.bfloat16
        )
        # Extract native vision components (correct path: model.model.*)
        self.vision_tower = full_model.model.vision_tower       # Gemma4VisionModel
        self.embed_vision = full_model.model.embed_vision       # Gemma4MultimodalEmbedder (768→1536)
        self.image_processor = AutoImageProcessor.from_pretrained(gemma4_model_path)
        # Store soft tokens count (typically 256 for 224x224 images with patch=16)
        self._num_soft_tokens = full_model.config.vision_soft_tokens_per_image  # 256

        # Extract language model embedding from the pretrained model
        pretrained_embed = full_model.model.language_model.embed_tokens
        logging.info(f"  Extracted vision_tower, embed_vision, image_processor (soft_tokens={self._num_soft_tokens})")

        del full_model  # Free the full model (keep only vision + embed)

        # --- VLM text decoder ---
        vlm_text_config = CONFIG_MAPPING["gemma4_text"](
            hidden_size=vlm_config.width,
            intermediate_size=vlm_config.mlp_dim,
            num_attention_heads=vlm_config.num_heads,
            num_hidden_layers=vlm_config.depth,
            num_key_value_heads=vlm_config.num_kv_heads,
            head_dim=vlm_config.head_dim,
            global_head_dim=vlm_config.head_dim,
            vocab_size=262144,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            hidden_size_per_layer_input=0,  # Disable PLE for initial integration
            layer_types=_make_layer_types(vlm_config.depth),
        )
        self.gemma4_vlm = Gemma4ForCausalLM(vlm_text_config)
        # Use pretrained embedding from Gemma-4 checkpoint
        self.gemma4_vlm.model.embed_tokens = pretrained_embed
        logging.info("  Created VLM text decoder with pretrained embedding")

        # --- Action expert ---
        action_expert_config_hf = CONFIG_MAPPING["gemma4_text"](
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            head_dim=action_expert_config.head_dim,
            global_head_dim=action_expert_config.head_dim,
            vocab_size=262144,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            hidden_size_per_layer_input=0,  # Disable PLE for action expert
            layer_types=_make_layer_types(action_expert_config.depth),
        )
        self.gemma4_expert = Gemma4ForCausalLM(config=action_expert_config_hf)
        self.gemma4_expert.model.embed_tokens = None  # Use VLM embeddings

        # --- adarms support for PI05 ---
        self.adarms_enabled = any(use_adarms)
        if self.adarms_enabled:
            self._setup_adarms(vlm_config, action_expert_config, use_adarms)

        self._to_bfloat16_for_selected_params(precision)

    def _setup_adarms(self, vlm_config, action_expert_config, use_adarms):
        """Add adarms (adaptive RMSNorm) support to decoder layers."""
        if use_adarms[0]:
            for layer in self.gemma4_vlm.model.layers:
                _enable_adarms_on_layer(layer, vlm_config.width)
        if use_adarms[1]:
            for layer in self.gemma4_expert.model.layers:
                _enable_adarms_on_layer(layer, action_expert_config.width)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """Public alias for _to_bfloat16_for_selected_params (compatibility with policy_config.py)."""
        self._to_bfloat16_for_selected_params(precision)

    def _to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.embeddings.patch_embedding.weight",
            "vision_tower.embeddings.patch_embedding.bias",
            "vision_tower.embeddings.position_embedding.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "model.norm.weight",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def inject_lora(self, vlm_lora_config=None, expert_lora_config=None):
        """Inject LoRA adapters into VLM and/or action expert.

        Args:
            vlm_lora_config: LoRA config for the VLM language model. None = no LoRA.
            expert_lora_config: LoRA config for the action expert. None = no LoRA.

        Returns:
            List of injected module names.
        """
        from openpi.models_pytorch.lora_pytorch import LoRAConfig, inject_lora_linear

        injected = []
        if vlm_lora_config is not None:
            lora_cfg = LoRAConfig(
                rank=vlm_lora_config.rank,
                alpha=vlm_lora_config.alpha,
                rslora=getattr(vlm_lora_config, "rslora", False),
            )
            injected.extend(inject_lora_linear(self.gemma4_vlm, lora_cfg))

        if expert_lora_config is not None:
            lora_cfg = LoRAConfig(
                rank=expert_lora_config.rank,
                alpha=expert_lora_config.alpha,
                rslora=getattr(expert_lora_config, "rslora", False),
            )
            injected.extend(inject_lora_linear(self.gemma4_expert, lora_cfg))

        logging.info(f"Injected LoRA into {len(injected)} modules")
        return injected

    def embed_image(self, images) -> torch.Tensor:
        """Embed images using Gemma-4 native vision encoder.

        Args:
            images: PIL Image(s) or a tensor of shape (B, C, H, W). Floating-point
                tensors may use [-1, 1], [0, 1], or [0, 255] values.

        Returns:
            Image embeddings of shape (B, num_tokens, vlm_hidden_size).
        """
        if isinstance(images, torch.Tensor):
            if images.dim() != 4:
                raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {images.shape}")
            processor_images = images.detach().to(dtype=torch.float32)
            if images.is_floating_point():
                image_min = processor_images.amin()
                if image_min < 0:
                    processor_images = (processor_images + 1.0) * 127.5
                elif processor_images.amax() <= 1.0:
                    processor_images = processor_images * 255.0
            processor_batch_size = images.shape[0]
        else:
            # Already PIL Image(s)
            processor_images = images if isinstance(images, list) else [images]
            processor_batch_size = len(processor_images)

        # Process through image processor → pixel_values (B, N, 768), position_ids (B, N, 2)
        inputs = self.image_processor(images=processor_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.vision_tower.device, dtype=self.vision_tower.dtype)
        position_ids = inputs["image_position_ids"].to(device=self.vision_tower.device)

        # Vision tower: (B, N, 768) → output tokens
        with torch.no_grad():
            vision_out = self.vision_tower(
                pixel_values=pixel_values,
                pixel_position_ids=position_ids,
            )
        batch_size = processor_batch_size
        # The vision tower output may not be batched: handle (N, D) vs (B, N, D)
        vision_features = vision_out.last_hidden_state
        if vision_features.dim() == 2:
            # (N, D) → (B, N, D)
            vision_features = vision_features.unsqueeze(0).expand(batch_size, -1, -1)
        elif vision_features.dim() == 3 and vision_features.shape[0] != batch_size:
            # (B*N, D) → (B, N, D)
            num_tokens = vision_features.shape[0] // batch_size
            vision_features = vision_features.reshape(batch_size, num_tokens, -1)

        # Project to VLM hidden size: (B, 256, 768) → (B, 256, 1536)
        image_embeddings = self.embed_vision(inputs_embeds=vision_features)
        return image_embeddings

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.gemma4_vlm.model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            # VLM-only forward (prefix caching)
            model_kwargs = {
                "inputs_embeds": inputs_embeds[0],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
            if adarms_cond is not None and adarms_cond[0] is not None:
                model_kwargs["adarms_cond"] = adarms_cond[0]
            prefix_output = self.gemma4_vlm.model(**model_kwargs)
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Action expert-only forward (suffix with cached prefix)
            model_kwargs = {
                "inputs_embeds": inputs_embeds[1],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
            if adarms_cond is not None and adarms_cond[1] is not None:
                model_kwargs["adarms_cond"] = adarms_cond[1]
            suffix_output = self.gemma4_expert.model(**model_kwargs)
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            # Joint attention forward
            prefix_output, suffix_output, prefix_past_key_values = self._forward_joint(
                attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond
            )

        return [prefix_output, suffix_output], prefix_past_key_values

    def _forward_joint(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        adarms_cond,
    ):
        """Joint attention forward pass for both VLM and action expert."""
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

        vlm_model = self.gemma4_vlm.model
        expert_model = self.gemma4_expert.model

        num_layers = min(len(vlm_model.layers), len(expert_model.layers))

        # Handle attention mask: accept either single tensor or dict of masks
        if isinstance(attention_mask, dict):
            full_mask = attention_mask.get("full_attention")
            sliding_mask = attention_mask.get("sliding_attention", full_mask)
        else:
            full_mask = attention_mask
            sliding_mask = attention_mask

        # SDPA requires an additive mask to match the query dtype. Preserve the
        # shared-mask alias so short sequences still retain only one mask tensor.
        masks_are_shared = sliding_mask is full_mask
        attention_dtype = inputs_embeds[0].dtype
        if full_mask is not None and full_mask.dtype not in (torch.bool, attention_dtype):
            full_mask = full_mask.to(dtype=attention_dtype)
        if masks_are_shared:
            sliding_mask = full_mask
        elif sliding_mask is not None and sliding_mask.dtype not in (torch.bool, attention_dtype):
            sliding_mask = sliding_mask.to(dtype=attention_dtype)

        # Check gradient checkpointing
        use_gradient_checkpointing = (
            hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training
        ) or (
            hasattr(expert_model, "gradient_checkpointing")
            and expert_model.gradient_checkpointing
            and self.training
        )

        # Pre-compute position embeddings for each layer_type
        # Use vlm_hidden for dtype/device reference (position_ids cover full combined sequence)
        hidden_states_dummy = inputs_embeds[0]
        unique_layer_types = set(vlm_model.config.layer_types[:num_layers])
        position_embeddings = {}
        for layer_type in unique_layer_types:
            position_embeddings[layer_type] = vlm_model.rotary_emb(
                hidden_states_dummy, position_ids, layer_type
            )

        # In offline RL the VLM can be fully frozen while the action expert remains trainable.
        # Prefix tokens cannot attend to suffix tokens, so the VLM branch can be evaluated
        # without autograd and the suffix branch can attend to its detached K/V states.
        # vlm_is_frozen is an invariant flag (only checks param state, not inputs) so it
        # remains correct during gradient-checkpoint recomputation where
        # inputs_embeds.requires_grad flips to True.
        vlm_is_frozen = not any(param.requires_grad for param in vlm_model.parameters())
        frozen_vlm_path = vlm_is_frozen and (
            adarms_cond[0] is None or not adarms_cond[0].requires_grad
        )
        # When VLM is frozen, wrap all VLM computation in torch.no_grad() to prevent
        # autograd from saving intermediate activations (LoRA A/B matmuls, MLP, layernorms).
        # This is the primary memory saving during backward checkpoint recomputation.
        _vlm_ctx = torch.no_grad if vlm_is_frozen else contextlib.nullcontext

        def compute_layer_complete(
            layer_idx, vlm_hidden, expert_hidden, full_mask, sliding_mask, position_embeddings, adarms_cond
        ):
            """Process one decoder layer with joint attention for both models."""
            vlm_layer = vlm_model.layers[layer_idx]
            expert_layer = expert_model.layers[layer_idx]
            layer_type = vlm_model.config.layer_types[layer_idx]
            mask = sliding_mask if layer_type == "sliding_attention" else full_mask
            cos, sin = position_embeddings[layer_type]

            # --- Input LayerNorm + Q/K/V projection ---
            # VLM branch: no_grad when frozen to avoid storing LoRA intermediates
            vlm_cond = adarms_cond[0] if adarms_cond is not None else None
            expert_cond = adarms_cond[1] if adarms_cond is not None else None
            with _vlm_ctx():
                vlm_ln_out = (
                    vlm_layer.input_layernorm(vlm_hidden, cond=vlm_cond)
                    if vlm_cond is not None
                    else vlm_layer.input_layernorm(vlm_hidden)
                )
                vlm_h, vlm_gate = (vlm_ln_out if isinstance(vlm_ln_out, tuple) else (vlm_ln_out, None))
                input_shape_vlm = vlm_h.shape[:-1]
                hidden_shape_vlm = (*input_shape_vlm, -1, vlm_layer.self_attn.head_dim)
                vlm_q = vlm_layer.self_attn.q_norm(vlm_layer.self_attn.q_proj(vlm_h).view(hidden_shape_vlm))
                vlm_k = vlm_layer.self_attn.k_norm(vlm_layer.self_attn.k_proj(vlm_h).view(hidden_shape_vlm))
                vlm_v = vlm_layer.self_attn.v_norm(vlm_layer.self_attn.v_proj(vlm_h).view(hidden_shape_vlm))
                # del vlm_h  # input_layernorm output consumed by Q/K/V projections

            # Expert branch: always with grad
            expert_ln_out = (
                expert_layer.input_layernorm(expert_hidden, cond=expert_cond)
                if expert_cond is not None
                else expert_layer.input_layernorm(expert_hidden)
            )
            expert_h, expert_gate = (expert_ln_out if isinstance(expert_ln_out, tuple) else (expert_ln_out, None))
            input_shape_expert = expert_h.shape[:-1]
            hidden_shape_expert = (*input_shape_expert, -1, expert_layer.self_attn.head_dim)
            expert_q = expert_layer.self_attn.q_norm(expert_layer.self_attn.q_proj(expert_h).view(hidden_shape_expert))
            expert_k = expert_layer.self_attn.k_norm(expert_layer.self_attn.k_proj(expert_h).view(hidden_shape_expert))
            expert_v = expert_layer.self_attn.v_norm(expert_layer.self_attn.v_proj(expert_h).view(hidden_shape_expert))

            vlm_seq_len = vlm_hidden.shape[1]
            expert_seq_len = expert_hidden.shape[1]
            if vlm_q.shape[2] != expert_q.shape[2]:
                raise ValueError(
                    "Gemma4 VLM and action expert must use the same number of query heads "
                    f"for joint attention, got {vlm_q.shape[2]} and {expert_q.shape[2]}"
                )

            # Apply RoPE before transposing to [batch, heads, seq, head_dim].
            # Slicing the shared position embedding avoids concatenating the two
            # hidden-state streams before attention.
            vlm_cos, vlm_sin = cos[:, :vlm_seq_len], sin[:, :vlm_seq_len]
            expert_cos, expert_sin = cos[:, vlm_seq_len:], sin[:, vlm_seq_len:]
            vlm_query_states = apply_rotary_pos_emb(
                vlm_q, vlm_cos, vlm_sin, unsqueeze_dim=2
            ).transpose(1, 2)
            vlm_key_states = apply_rotary_pos_emb(
                vlm_k, vlm_cos, vlm_sin, unsqueeze_dim=2
            ).transpose(1, 2)
            vlm_value_states = vlm_v.transpose(1, 2)
            expert_query_states = apply_rotary_pos_emb(
                expert_q, expert_cos, expert_sin, unsqueeze_dim=2
            ).transpose(1, 2)
            expert_key_states = apply_rotary_pos_emb(
                expert_k, expert_cos, expert_sin, unsqueeze_dim=2
            ).transpose(1, 2)
            expert_value_states = expert_v.transpose(1, 2)

            # Align KV heads between the two streams only when their configs differ;
            # the SDPA helper handles the final query-to-KV expansion.
            vlm_kv_heads = vlm_key_states.shape[1]
            expert_kv_heads = expert_key_states.shape[1]
            if vlm_kv_heads != expert_kv_heads:
                target_kv_heads = max(vlm_kv_heads, expert_kv_heads)
                if target_kv_heads % vlm_kv_heads != 0 or target_kv_heads % expert_kv_heads != 0:
                    raise ValueError(
                        "Gemma4 VLM and action expert KV-head counts must divide a common target, "
                        f"got {vlm_kv_heads} and {expert_kv_heads}"
                    )
                if vlm_kv_heads != target_kv_heads:
                    vlm_key_states = repeat_kv(vlm_key_states, target_kv_heads // vlm_kv_heads)
                    vlm_value_states = repeat_kv(vlm_value_states, target_kv_heads // vlm_kv_heads)
                if expert_kv_heads != target_kv_heads:
                    expert_key_states = repeat_kv(expert_key_states, target_kv_heads // expert_kv_heads)
                    expert_value_states = repeat_kv(expert_value_states, target_kv_heads // expert_kv_heads)

            if frozen_vlm_path:
                with torch.no_grad():
                    vlm_att_output = _scaled_dot_product_attention(
                        vlm_query_states,
                        vlm_key_states,
                        vlm_value_states,
                        None if mask is None else mask[:, :, :vlm_seq_len, :vlm_seq_len],
                    )

                expert_key_states = torch.cat([vlm_key_states.detach(), expert_key_states], dim=2)
                expert_value_states = torch.cat([vlm_value_states.detach(), expert_value_states], dim=2)
                expert_att_output = _scaled_dot_product_attention(
                    expert_query_states,
                    expert_key_states,
                    expert_value_states,
                    None if mask is None else mask[:, :, vlm_seq_len:, :],
                )
            else:
                # Concatenate only after RoPE and KV alignment. This preserves
                # the joint-attention semantics while avoiding needless copies.
                query_states = torch.cat([vlm_query_states, expert_query_states], dim=2)
                key_states = torch.cat([vlm_key_states, expert_key_states], dim=2)
                value_states = torch.cat([vlm_value_states, expert_value_states], dim=2)
                att_output = _scaled_dot_product_attention(query_states, key_states, value_states, mask)
                vlm_att_output, expert_att_output = att_output.split((vlm_seq_len, expert_seq_len), dim=2)

            # Keep the two output streams separate; concatenating them and then
            # splitting again creates a large transient tensor during backward.
            head_dim = vlm_layer.self_attn.head_dim
            num_q_heads = vlm_query_states.shape[1]
            vlm_att_out = vlm_att_output.transpose(1, 2).reshape(
                vlm_att_output.shape[0], vlm_seq_len, num_q_heads * head_dim
            )
            expert_att_out = expert_att_output.transpose(1, 2).reshape(
                expert_att_output.shape[0], expert_seq_len, num_q_heads * head_dim
            )

            # --- VLM: o_proj + post_attn_norm + residual + MLP (no_grad if frozen) ---
            with _vlm_ctx():
                if vlm_att_out.dtype != _get_weight(vlm_layer.self_attn.o_proj).dtype:
                    vlm_att_out = vlm_att_out.to(_get_weight(vlm_layer.self_attn.o_proj).dtype)
                if frozen_vlm_path:
                    vlm_att_out = vlm_att_out.detach()
                vlm_att_out = vlm_layer.self_attn.o_proj(vlm_att_out)
                vlm_post_attn = (
                    vlm_layer.post_attention_layernorm(vlm_att_out, cond=vlm_cond)
                    if vlm_cond is not None
                    else vlm_layer.post_attention_layernorm(vlm_att_out)
                )
                vlm_post_attn, vlm_post_gate = (
                    vlm_post_attn if isinstance(vlm_post_attn, tuple) else (vlm_post_attn, None)
                )
                vlm_hidden = _gated_residual(vlm_hidden, vlm_post_attn, vlm_gate or vlm_post_gate)

                residual = vlm_hidden
                vlm_ff = (
                    vlm_layer.pre_feedforward_layernorm(vlm_hidden, cond=vlm_cond)
                    if vlm_cond is not None
                    else vlm_layer.pre_feedforward_layernorm(vlm_hidden)
                )
                vlm_ff = vlm_ff[0] if isinstance(vlm_ff, tuple) else vlm_ff
                vlm_ff = vlm_layer.mlp(vlm_ff)
                vlm_ff = (
                    vlm_layer.post_feedforward_layernorm(vlm_ff, cond=vlm_cond)
                    if vlm_cond is not None
                    else vlm_layer.post_feedforward_layernorm(vlm_ff)
                )
                vlm_ff = vlm_ff[0] if isinstance(vlm_ff, tuple) else vlm_ff
                vlm_hidden = residual + vlm_ff

                vlm_hidden = vlm_hidden * vlm_layer.layer_scalar

            # --- Expert: o_proj + post_attn_norm + residual + MLP ---
            if expert_att_out.dtype != _get_weight(expert_layer.self_attn.o_proj).dtype:
                expert_att_out = expert_att_out.to(_get_weight(expert_layer.self_attn.o_proj).dtype)
            expert_att_out = expert_layer.self_attn.o_proj(expert_att_out)
            expert_post_attn = (
                expert_layer.post_attention_layernorm(expert_att_out, cond=expert_cond)
                if expert_cond is not None
                else expert_layer.post_attention_layernorm(expert_att_out)
            )
            expert_post_attn, expert_post_gate = (
                expert_post_attn if isinstance(expert_post_attn, tuple) else (expert_post_attn, None)
            )
            expert_hidden = _gated_residual(expert_hidden, expert_post_attn, expert_gate or expert_post_gate)

            residual = expert_hidden
            expert_ff = (
                expert_layer.pre_feedforward_layernorm(expert_hidden, cond=expert_cond)
                if expert_cond is not None
                else expert_layer.pre_feedforward_layernorm(expert_hidden)
            )
            expert_ff = expert_ff[0] if isinstance(expert_ff, tuple) else expert_ff
            expert_ff = expert_layer.mlp(expert_ff)
            expert_ff = (
                expert_layer.post_feedforward_layernorm(expert_ff, cond=expert_cond)
                if expert_cond is not None
                else expert_layer.post_feedforward_layernorm(expert_ff)
            )
            expert_ff = expert_ff[0] if isinstance(expert_ff, tuple) else expert_ff
            expert_hidden = residual + expert_ff

            expert_hidden = expert_hidden * expert_layer.layer_scalar

            return vlm_hidden, expert_hidden

        # Process all layers
        vlm_hidden = inputs_embeds[0]
        expert_hidden = inputs_embeds[1]

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                vlm_hidden, expert_hidden = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    vlm_hidden,
                    expert_hidden,
                    full_mask,
                    sliding_mask,
                    position_embeddings,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                vlm_hidden, expert_hidden = compute_layer_complete(
                    layer_idx, vlm_hidden, expert_hidden, full_mask, sliding_mask, position_embeddings, adarms_cond
                )

        # Final norm
        prefix_output = vlm_model.norm(vlm_hidden)
        suffix_output = expert_model.norm(expert_hidden)

        return prefix_output, suffix_output, None

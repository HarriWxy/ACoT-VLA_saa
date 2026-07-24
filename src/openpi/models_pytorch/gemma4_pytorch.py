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
from transformers import AutoImageProcessor
from transformers import Gemma4ForCausalLM
from transformers import Gemma4ForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING


def _get_weight(module: nn.Module) -> torch.Tensor:
    """获取模块权重, 兼容 LoRALinear (权重在 base_linear.weight)。"""
    return module.base_linear.weight if hasattr(module, "base_linear") else module.weight
from transformers.models.gemma4.modeling_gemma4 import repeat_kv


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
    modulation = self.adarms_dense(cond)
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
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "model.norm",
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
            prefix_output = self.gemma4_vlm.model(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Action expert-only forward (suffix with cached prefix)
            suffix_output = self.gemma4_expert.model(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
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
            with _vlm_ctx():
                vlm_ln_out = vlm_layer.input_layernorm(vlm_hidden)
                vlm_h, vlm_gate = (vlm_ln_out if isinstance(vlm_ln_out, tuple) else (vlm_ln_out, None))
                input_shape_vlm = vlm_h.shape[:-1]
                hidden_shape_vlm = (*input_shape_vlm, -1, vlm_layer.self_attn.head_dim)
                vlm_q = vlm_layer.self_attn.q_norm(vlm_layer.self_attn.q_proj(vlm_h).view(hidden_shape_vlm))
                vlm_k = vlm_layer.self_attn.k_norm(vlm_layer.self_attn.k_proj(vlm_h).view(hidden_shape_vlm))
                vlm_v = vlm_layer.self_attn.v_norm(vlm_layer.self_attn.v_proj(vlm_h).view(hidden_shape_vlm))
                del vlm_h  # input_layernorm output consumed by Q/K/V projections

            # Expert branch: always with grad
            expert_ln_out = expert_layer.input_layernorm(expert_hidden)
            expert_h, expert_gate = (expert_ln_out if isinstance(expert_ln_out, tuple) else (expert_ln_out, None))
            input_shape_expert = expert_h.shape[:-1]
            hidden_shape_expert = (*input_shape_expert, -1, expert_layer.self_attn.head_dim)
            expert_q = expert_layer.self_attn.q_norm(expert_layer.self_attn.q_proj(expert_h).view(hidden_shape_expert))
            expert_k = expert_layer.self_attn.k_norm(expert_layer.self_attn.k_proj(expert_h).view(hidden_shape_expert))
            expert_v = expert_layer.self_attn.v_norm(expert_layer.self_attn.v_proj(expert_h).view(hidden_shape_expert))
            del expert_h  # input_layernorm output consumed by Q/K/V projections

            # Expand KV heads to match if models have different num_kv_heads
            # repeat_kv expects [batch, num_kv_heads, seq, head_dim]
            vlm_kv_heads = vlm_k.shape[2]
            expert_kv_heads = expert_k.shape[2]
            if vlm_kv_heads != expert_kv_heads:
                target_kv_heads = max(vlm_kv_heads, expert_kv_heads)
                if vlm_kv_heads < target_kv_heads:
                    vlm_k = repeat_kv(vlm_k.permute(0, 2, 1, 3), target_kv_heads // vlm_kv_heads).permute(0, 2, 1, 3)
                    vlm_v = repeat_kv(vlm_v.permute(0, 2, 1, 3), target_kv_heads // vlm_kv_heads).permute(0, 2, 1, 3)
                if expert_kv_heads < target_kv_heads:
                    expert_k = repeat_kv(expert_k.permute(0, 2, 1, 3), target_kv_heads // expert_kv_heads).permute(0, 2, 1, 3)
                    expert_v = repeat_kv(expert_v.permute(0, 2, 1, 3), target_kv_heads // expert_kv_heads).permute(0, 2, 1, 3)

            vlm_seq_len = vlm_hidden.shape[1]
            if frozen_vlm_path:
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

                num_q_heads = vlm_query_states.shape[1]
                if vlm_key_states.shape[1] != num_q_heads:
                    vlm_kv_repeat = num_q_heads // vlm_key_states.shape[1]
                    vlm_key_states = repeat_kv(vlm_key_states, vlm_kv_repeat)
                    vlm_value_states = repeat_kv(vlm_value_states, vlm_kv_repeat)
                if expert_key_states.shape[1] != num_q_heads:
                    expert_kv_repeat = num_q_heads // expert_key_states.shape[1]
                    expert_key_states = repeat_kv(expert_key_states, expert_kv_repeat)
                    expert_value_states = repeat_kv(expert_value_states, expert_kv_repeat)

                with torch.no_grad():
                    vlm_att_output = torch.nn.functional.scaled_dot_product_attention(
                        vlm_query_states,
                        vlm_key_states,
                        vlm_value_states,
                        attn_mask=None if mask is None else mask[:, :, :vlm_seq_len, :vlm_seq_len],
                        dropout_p=0.0,
                        scale=1.0,
                    )
                del vlm_query_states  # VLM Q no longer needed

                expert_key_states = torch.cat([vlm_key_states.detach(), expert_key_states], dim=2)
                del vlm_key_states  # Detached copy consumed by concat
                expert_value_states = torch.cat([vlm_value_states.detach(), expert_value_states], dim=2)
                del vlm_value_states  # Detached copy consumed by concat
                expert_att_output = torch.nn.functional.scaled_dot_product_attention(
                    expert_query_states,
                    expert_key_states,
                    expert_value_states,
                    attn_mask=None if mask is None else mask[:, :, vlm_seq_len:, :],
                    dropout_p=0.0,
                    scale=1.0,
                )
                del expert_key_states, expert_value_states  # Consumed by SDPA
                att_output = torch.cat([vlm_att_output, expert_att_output], dim=2)
                del vlm_att_output, expert_att_output  # Consumed by concat
                batch_size = expert_query_states.shape[0]
            else:
                # Concatenate along sequence dimension, then apply RoPE
                # (position_ids cover the full combined sequence)
                query_states = torch.cat([vlm_q, expert_q], dim=1)  # [batch, seq, heads, head_dim]
                key_states = torch.cat([vlm_k, expert_k], dim=1)
                value_states = torch.cat([vlm_v, expert_v], dim=1)

                query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                # GQA: expand KV heads to match Q heads if needed
                # After transpose: [batch, num_heads, seq, head_dim]
                # num_attention_heads (Q=8) may differ from num_kv_heads (K/V=4)
                num_q_heads = query_states.shape[1]
                num_kv_heads = key_states.shape[1]
                if num_kv_heads != num_q_heads:
                    n_rep = num_q_heads // num_kv_heads
                    key_states = repeat_kv(key_states, n_rep)
                    value_states = repeat_kv(value_states, n_rep)

                batch_size = query_states.shape[0]

                # On supported CUDA backends, SDPA selects a fused memory-efficient
                # kernel for this dense prefix-LM mask and avoids materializing the
                # [B, H, S, S] attention-weight tensor.
                # scale=1.0 preserves the pre-existing joint-attention calculation.
                att_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=mask,
                    dropout_p=0.0,
                    scale=1.0,
                )
                del query_states, key_states, value_states  # Consumed by SDPA
            att_output = att_output.transpose(1, 2).contiguous()

            head_dim = vlm_layer.self_attn.head_dim
            att_output = att_output.reshape(batch_size, -1, num_q_heads * head_dim)

            # Split output back to each model
            vlm_seq_len = vlm_hidden.shape[1]
            vlm_att_out = att_output[:, :vlm_seq_len]
            expert_att_out = att_output[:, vlm_seq_len:]
            del att_output  # Free full [B, total_seq, dim] tensor immediately

            # --- VLM: o_proj + post_attn_norm + residual + MLP (no_grad if frozen) ---
            with _vlm_ctx():
                # Cache weight ref to avoid repeated _get_weight() lookups
                o_proj_weight = _get_weight(vlm_layer.self_attn.o_proj)
                if vlm_att_out.dtype != o_proj_weight.dtype:
                    vlm_att_out = vlm_att_out.to(o_proj_weight.dtype)
                if frozen_vlm_path:
                    vlm_att_out = vlm_att_out.detach()
                vlm_att_out = vlm_layer.self_attn.o_proj(vlm_att_out)
                vlm_post_attn = vlm_layer.post_attention_layernorm(vlm_att_out)
                del vlm_att_out  # o_proj output no longer needed
                vlm_post_attn, vlm_post_gate = (
                    vlm_post_attn if isinstance(vlm_post_attn, tuple) else (vlm_post_attn, None)
                )
                vlm_hidden = _gated_residual(vlm_hidden, vlm_post_attn, vlm_gate or vlm_post_gate)
                del vlm_post_attn  # Consumed by residual connection

                residual = vlm_hidden
                vlm_ff = vlm_layer.pre_feedforward_layernorm(vlm_hidden)
                del vlm_hidden  # Free old hidden; residual still holds reference
                vlm_ff = vlm_layer.mlp(vlm_ff)
                vlm_ff = vlm_layer.post_feedforward_layernorm(vlm_ff)
                vlm_hidden = residual + vlm_ff
                del residual, vlm_ff  # Both consumed by addition

                vlm_hidden = vlm_hidden * vlm_layer.layer_scalar

            # --- Expert: o_proj + post_attn_norm + residual + MLP ---
            expert_o_proj_weight = _get_weight(expert_layer.self_attn.o_proj)
            if expert_att_out.dtype != expert_o_proj_weight.dtype:
                expert_att_out = expert_att_out.to(expert_o_proj_weight.dtype)
            expert_att_out = expert_layer.self_attn.o_proj(expert_att_out)
            expert_post_attn = expert_layer.post_attention_layernorm(expert_att_out)
            del expert_att_out  # o_proj output no longer needed
            expert_post_attn, expert_post_gate = (
                expert_post_attn if isinstance(expert_post_attn, tuple) else (expert_post_attn, None)
            )
            expert_hidden = _gated_residual(expert_hidden, expert_post_attn, expert_gate or expert_post_gate)
            del expert_post_attn  # Consumed by residual connection

            residual = expert_hidden
            expert_ff = expert_layer.pre_feedforward_layernorm(expert_hidden)
            del expert_hidden  # Free old hidden; residual still holds reference
            expert_ff = expert_layer.mlp(expert_ff)
            expert_ff = expert_layer.post_feedforward_layernorm(expert_ff)
            expert_hidden = residual + expert_ff
            del residual, expert_ff  # Both consumed by addition

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

        # If VLM has more layers than expert, continue processing VLM-only layers.
        # This ensures the full VLM depth is utilized even when expert is smaller.
        vlm_only_start = num_layers
        vlm_only_end = len(vlm_model.layers)
        if vlm_only_start < vlm_only_end:
            vlm_seq_len = vlm_hidden.shape[1]

            for layer_idx in range(vlm_only_start, vlm_only_end):
                vlm_layer = vlm_model.layers[layer_idx]
                layer_type = vlm_model.config.layer_types[layer_idx]
                mask = sliding_mask if layer_type == "sliding_attention" else full_mask
                # Slice mask to VLM-only sequence length
                if mask is not None:
                    mask = mask[:, :, :vlm_seq_len, :vlm_seq_len]
                cos, sin = position_embeddings[layer_type]
                cos, sin = cos[:, :vlm_seq_len], sin[:, :vlm_seq_len]

                # Bind loop variables via default args to avoid closure capture issues
                def _vlm_only_layer(vlm_hidden, _vlm_layer=vlm_layer, _cos=cos, _sin=sin, _mask=mask):
                    vlm_ln_out = _vlm_layer.input_layernorm(vlm_hidden)
                    vlm_h, vlm_gate = (vlm_ln_out if isinstance(vlm_ln_out, tuple) else (vlm_ln_out, None))
                    input_shape_vlm = vlm_h.shape[:-1]
                    hidden_shape_vlm = (*input_shape_vlm, -1, _vlm_layer.self_attn.head_dim)
                    vlm_q = _vlm_layer.self_attn.q_norm(_vlm_layer.self_attn.q_proj(vlm_h).view(hidden_shape_vlm))
                    vlm_k = _vlm_layer.self_attn.k_norm(_vlm_layer.self_attn.k_proj(vlm_h).view(hidden_shape_vlm))
                    vlm_v = _vlm_layer.self_attn.v_norm(_vlm_layer.self_attn.v_proj(vlm_h).view(hidden_shape_vlm))
                    del vlm_h

                    # Apply RoPE
                    vlm_q = apply_rotary_pos_emb(vlm_q, _cos, _sin, unsqueeze_dim=2).transpose(1, 2)
                    vlm_k = apply_rotary_pos_emb(vlm_k, _cos, _sin, unsqueeze_dim=2).transpose(1, 2)
                    vlm_v = vlm_v.transpose(1, 2)

                    # GQA expand
                    num_q_heads = vlm_q.shape[2]
                    if vlm_k.shape[2] != num_q_heads:
                        n_rep = num_q_heads // vlm_k.shape[2]
                        vlm_k = repeat_kv(vlm_k, n_rep)
                        vlm_v = repeat_kv(vlm_v, n_rep)

                    att_output = torch.nn.functional.scaled_dot_product_attention(
                        vlm_q, vlm_k, vlm_v,
                        attn_mask=_mask, dropout_p=0.0, scale=1.0,
                    )
                    del vlm_q, vlm_k, vlm_v

                    att_output = att_output.transpose(1, 2).contiguous()
                    head_dim = _vlm_layer.self_attn.head_dim
                    att_output = att_output.reshape(att_output.shape[0], -1, num_q_heads * head_dim)

                    # o_proj + post_attn_norm + residual + MLP
                    o_proj_weight = _get_weight(_vlm_layer.self_attn.o_proj)
                    if att_output.dtype != o_proj_weight.dtype:
                        att_output = att_output.to(o_proj_weight.dtype)
                    att_output = _vlm_layer.self_attn.o_proj(att_output)
                    vlm_post_attn = _vlm_layer.post_attention_layernorm(att_output)
                    del att_output
                    vlm_post_attn, vlm_post_gate = (
                        vlm_post_attn if isinstance(vlm_post_attn, tuple) else (vlm_post_attn, None)
                    )
                    vlm_hidden = _gated_residual(vlm_hidden, vlm_post_attn, vlm_gate or vlm_post_gate)
                    del vlm_post_attn

                    residual = vlm_hidden
                    vlm_ff = _vlm_layer.pre_feedforward_layernorm(vlm_hidden)
                    del vlm_hidden
                    vlm_ff = _vlm_layer.mlp(vlm_ff)
                    vlm_ff = _vlm_layer.post_feedforward_layernorm(vlm_ff)
                    vlm_hidden = residual + vlm_ff
                    del residual, vlm_ff

                    vlm_hidden = vlm_hidden * _vlm_layer.layer_scalar
                    return vlm_hidden

                if use_gradient_checkpointing:
                    vlm_hidden = torch.utils.checkpoint.checkpoint(
                        _vlm_only_layer, vlm_hidden,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                else:
                    vlm_hidden = _vlm_only_layer(vlm_hidden)

        # Final norm
        prefix_output = vlm_model.norm(vlm_hidden)
        suffix_output = expert_model.norm(expert_hidden)

        return prefix_output, suffix_output, None

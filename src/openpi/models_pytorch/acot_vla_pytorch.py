"""ACoT-VLA PyTorch implementation.

This module implements the ACoT-VLA (Action Chain-of-Thought Vision-Language-Action)
model in PyTorch, mirroring the JAX/Flax implementation in `openpi/models/acot_vla.py`.

Architecture overview:
    ACoT-VLA extends PI0 with a dual-expert action reasoning system:
    - Coarse Action Reasoner (explicit): generates coarse-grained action plans
    - Fine Action Expert: produces precise actions conditioned on reasoning
    - Implicit Action Reasoner: extracts latent action cues from VLM features

Key components:
    - PaliGemmaWithDualExpertModel: three-expert transformer (VLM + 2 action experts)
    - LearnableQueryExtractor / AttentionPoolingExtractor / DownsampleExtractor:
      implicit action reasoning modules
    - UnifiedAttentionModule: cross-attention fusion for explicit/implicit reasoning
    - ACOT_VLAPytorch: main model with flow matching training and inference
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Literal, Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger("ACoT_VLA_Pytorch")

# ---------------------------------------------------------------------------
# Shared utilities (reuse from pi0_pytorch)
# ---------------------------------------------------------------------------

from openpi.models_pytorch.pi0_pytorch import (
    create_sinusoidal_pos_embedding,
    get_safe_dtype,
    make_att_2d_masks,
    sample_beta,
)


# ---------------------------------------------------------------------------
# Helper modules (from JAX acot_vla.py)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Three-layer MLP with optional activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, activate: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activate = activate

    def forward(self, x: Tensor) -> Tensor:
        if self.activate:
            return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x)))))
        return self.fc3(self.fc2(self.fc1(x)))


class LearnableQueryExtractor(nn.Module):
    """Learnable query-based implicit action reasoner.

    Extracts action-relevant features from VLM key-value cache using
    learnable query vectors and multi-head cross-attention.

    For each transformer layer l, computes:
        Q_l = learnable_queries  (num_queries, dim)
        K_l, V_l = kv_cache[:, l, :, :]
        output_l = CrossAttention(Q_l, K_l, V_l)
    """

    def __init__(
        self,
        num_queries: int,
        dim: int,
        output_dim: int,
        depth: int,
        heads: int = 8,
        head_dim: int = 256,
        group_size: int = 3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_groups = depth // group_size

        # Learnable query parameters for each layer
        self.query_params = nn.ParameterList([
            nn.Parameter(torch.randn(num_queries, dim) * 0.02)
            for _ in range(self.depth)
        ])

        # Grouped projection layers (shared within each group)
        self.q_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(heads * head_dim, output_dim)
            for _ in range(self.num_groups)
        ])

    def forward(self, K: Tensor, V: Tensor) -> Tensor:
        """
        Args:
            K: Key tensor [B, L, T, D] from VLM KV cache
            V: Value tensor [B, L, T, D] from VLM KV cache

        Returns:
            Extracted features [B, L, output_dim]
        """
        B, L, T, D = K.shape
        outputs = []

        for l in range(L):
            g = l // self.group_size
            Q_l = self.query_params[l].unsqueeze(0)  # (1, Q, D)
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]

            Q_proj = self.q_proj[g](Q_l).reshape(1, self.num_queries, self.heads, self.head_dim).permute(0, 2, 1, 3)
            K_proj = self.k_proj[g](K_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
            V_proj = self.v_proj[g](V_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

            attn = torch.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            pooled = torch.einsum("bhqk,bhkd->bhqd", attn, V_proj)  # (B, H, Q, Hd)

            pooled = pooled.mean(dim=2)  # (B, H, Hd)
            pooled = pooled.permute(0, 2, 1).reshape(B, self.heads * self.head_dim)
            pooled = self.out_proj[g](pooled)  # (B, output_dim)
            outputs.append(pooled)

        return torch.stack(outputs, dim=1)  # (B, L, output_dim)


class AttentionPoolingExtractor(nn.Module):
    """Attention pooling-based implicit action reasoner.

    Uses the key tensor itself as the query source (mean-pooled),
    avoiding the need for separate learnable queries.
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        heads: int = 8,
        head_dim: int = 256,
        group_size: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.num_groups = depth // group_size

        self.k_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.q_proj = nn.ModuleList([
            nn.Linear(dim, heads * head_dim)
            for _ in range(self.num_groups)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(heads * head_dim, output_dim)
            for _ in range(self.num_groups)
        ])

    def forward(self, K: Tensor, V: Tensor) -> Tensor:
        """
        Args:
            K: [B, L, T, D]
            V: [B, L, T, D]
        Returns:
            [B, L, output_dim]
        """
        B, L, T, D = K.shape
        outputs = []

        for l in range(L):
            g = l // self.group_size
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]
            # Query from mean-pooled keys
            Q_l = K_l.mean(dim=1, keepdim=True)  # (B, 1, D)

            Q_proj = self.q_proj[g](Q_l).reshape(B, 1, self.heads, self.head_dim).permute(0, 2, 1, 3)
            K_proj = self.k_proj[g](K_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
            V_proj = self.v_proj[g](V_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

            attn = torch.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            pooled = torch.einsum("bhqk,bhkd->bhqd", attn, V_proj)  # (B, H, 1, Hd)

            pooled = pooled.permute(0, 2, 1, 3).reshape(B, self.heads * self.head_dim)
            pooled = self.out_proj[g](pooled)
            outputs.append(pooled)

        return torch.stack(outputs, dim=1)


class DownsampleExtractor(nn.Module):
    """Downsample-based implicit action reasoner.

    Projects K/V to a lower-dimensional space before attention,
    reducing computational cost for deep transformer models.
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        downsample_dim: int = 512,
        group_size: int = 3,
        num_queries: int = 1,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.downsample_dim = downsample_dim
        self.group_size = group_size
        self.num_groups = depth // group_size
        self.num_queries = num_queries
        self.heads = heads
        self.head_dim = downsample_dim // heads

        self.query_params = nn.ParameterList([
            nn.Parameter(torch.randn(num_queries, dim) * 0.02)
            for _ in range(self.depth)
        ])

        self.q_proj = nn.ModuleList([
            nn.Linear(dim, downsample_dim)
            for _ in range(self.num_groups)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(dim, downsample_dim)
            for _ in range(self.num_groups)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim, downsample_dim)
            for _ in range(self.num_groups)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(downsample_dim, output_dim)
            for _ in range(self.num_groups)
        ])

    def forward(self, K: Tensor, V: Tensor) -> Tensor:
        B, L, T, D = K.shape
        outputs = []

        for l in range(L):
            g = l // self.group_size
            K_l, V_l = K[:, l, :, :], V[:, l, :, :]

            Q_l = self.query_params[l].unsqueeze(0)  # (1, Q, D)
            Q_proj = self.q_proj[g](Q_l).reshape(1, self.num_queries, self.heads, self.head_dim).permute(0, 2, 1, 3)
            K_proj = self.k_proj[g](K_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
            V_proj = self.v_proj[g](V_l).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)

            Q_proj = Q_proj.expand(B, -1, -1, -1)  # Tile for batch

            attn = torch.einsum("bhqd,bhkd->bhqk", Q_proj, K_proj) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)

            pooled = torch.einsum("bhqk,bhkd->bhqd", attn, V_proj)
            pooled = pooled.mean(dim=2) if self.num_queries > 1 else pooled.squeeze(dim=2)

            pooled = pooled.permute(0, 2, 1).reshape(B, self.downsample_dim)
            feat = self.out_proj[g](pooled)
            outputs.append(feat)

        return torch.stack(outputs, dim=1)  # (B, L, output_dim)


class UnifiedAttentionModule(nn.Module):
    """Cross-attention module for fusing action reasoning signals.

    Used to align explicit/implicit action reasoning with the expert's
    action token representations.
    """

    def __init__(
        self,
        in_dim_1: int,
        in_dim_2: int,
        out_dim: int,
        apply_sigmoid: bool = False,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.q_proj = nn.Linear(in_dim_1, hidden_dim)
        self.kv_proj = nn.Linear(in_dim_2, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.apply_sigmoid = apply_sigmoid

    def forward(self, feat_1: Tensor, feat_2: Tensor) -> Tensor:
        """
        Args:
            feat_1: Query features [B, T1, D1]
            feat_2: Key/Value features [B, T2, D2]

        Returns:
            Fused features [B, T1, out_dim]
        """
        B, T1, _ = feat_1.shape
        T2 = feat_2.shape[1]

        Q = self.q_proj(feat_1).reshape(B, T1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        KV = self.kv_proj(feat_2)
        K, V = KV.chunk(2, dim=-1)
        K = K.reshape(B, T2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, T2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, V)

        out = out.permute(0, 2, 1, 3).reshape(B, T1, self.num_heads * self.head_dim)
        output = self.out_proj(out)

        if self.apply_sigmoid:
            return torch.sigmoid(output)
        return output


# ---------------------------------------------------------------------------
# Three-expert PaliGemma model for ACoT-VLA
# ---------------------------------------------------------------------------


class PaliGemmaWithDualExpertModel(nn.Module):
    """PaliGemma with two action experts (coarse + fine) for ACoT-VLA.

    This extends PaliGemmaWithExpertModel to support three experts:
    1. PaliGemma VLM (vision-language model)
    2. Coarse action expert (for explicit action reasoning)
    3. Fine action expert (for final action prediction)

    The three experts share attention computation within each transformer layer,
    following the JAX gemma.Module architecture.
    """

    def __init__(
        self,
        vlm_config,
        coarse_action_expert_config,
        action_expert_config,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()
        if use_adarms is None:
            use_adarms = [False, False, False]

        # Build PaliGemma VLM
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        # Build coarse action expert
        coarse_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=coarse_action_expert_config.head_dim,
            hidden_size=coarse_action_expert_config.width,
            intermediate_size=coarse_action_expert_config.mlp_dim,
            num_attention_heads=coarse_action_expert_config.num_heads,
            num_hidden_layers=coarse_action_expert_config.depth,
            num_key_value_heads=coarse_action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=coarse_action_expert_config.width if use_adarms[1] else None,
        )

        # Build fine action expert
        fine_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[2],
            adarms_cond_dim=action_expert_config.width if use_adarms[2] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.coarse_expert = GemmaForCausalLM(config=coarse_expert_config_hf)
        self.coarse_expert.model.embed_tokens = None
        self.fine_expert = GemmaForCausalLM(config=fine_expert_config_hf)
        self.fine_expert.model.embed_tokens = None

        self._apply_precision(precision)
        self._debug_gc_printed = False

    def _apply_precision(self, precision: str):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: Tensor) -> Tensor:
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[Tensor | None] | None = None,
    ) -> tuple[list[Tensor | None], list[torch.FloatTensor] | None]:
        """Forward pass with up to three experts.

        Args:
            attention_mask: 4D attention mask.
            position_ids: Position indices.
            past_key_values: KV cache from prefix forward.
            inputs_embeds: List of 3 embedding tensors [vlm, coarse, fine].
                Each can be None if that expert should not be run.
            use_cache: Whether to return KV cache.
            adarms_cond: List of 3 conditioning tensors for adaRMS.

        Returns:
            outputs_embeds: List of 3 output tensors [vlm, coarse, fine].
            past_key_values: KV cache (from VLM prefix pass only).
        """
        if adarms_cond is None:
            adarms_cond = [None, None, None]
        if inputs_embeds is None:
            inputs_embeds = [None, None, None]

        # Determine which experts to run
        run_vlm = inputs_embeds[0] is not None
        run_coarse = inputs_embeds[1] is not None
        run_fine = inputs_embeds[2] is not None

        # Build list of active models and their inputs
        models = [self.paligemma.language_model, self.coarse_expert.model, self.fine_expert.model]
        active_indices = [i for i, flag in enumerate([run_vlm, run_coarse, run_fine]) if flag]

        # --- Prefix-only pass (VLM with cache) ---
        if run_vlm and not run_coarse and not run_fine:
            output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            result = [output.last_hidden_state, None, None]
            return result, output.past_key_values

        # --- Suffix-only pass (one or two experts with KV cache) ---
        if not run_vlm and (run_coarse or run_fine):
            results = [None, None, None]
            kv_out = None

            for idx in active_indices:
                model = models[idx]
                output = model.forward(
                    inputs_embeds=inputs_embeds[idx],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[idx],
                )
                results[idx] = output.last_hidden_state

            return results, kv_out

        # --- Joint pass (VLM + one or two experts, shared attention) ---
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        # Check gradient checkpointing
        use_gradient_checkpointing = (
            self.training
            and any(
                hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing
                for m in [self.coarse_expert.model, self.fine_expert.model]
            )
        )

        def compute_layer_complete(layer_idx, inputs_embeds_list, attn_mask, pos_ids, adarms_cond_list):
            """Process one transformer layer for all active experts."""
            query_states = []
            key_states = []
            value_states = []
            gates = []

            for i in active_indices:
                layer = models[i].layers[layer_idx]
                hidden_states = inputs_embeds_list[i]
                hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond_list[i])
                gates.append(gate)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                query_states.append(q)
                key_states.append(k)
                value_states.append(v)

            # Concatenate across experts
            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            # Apply RoPE
            dummy_tensor = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype,
            )
            cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, pos_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            batch_size = query_states.shape[0]
            ref_layer = models[active_indices[0]].layers[layer_idx]
            scaling = ref_layer.self_attn.scaling
            head_dim = ref_layer.self_attn.head_dim

            # Joint attention
            att_output, _ = modeling_gemma.eager_attention_forward(
                ref_layer.self_attn, query_states, key_states, value_states,
                attn_mask, scaling,
            )
            att_output = att_output.reshape(batch_size, -1, len(active_indices) * 8 * head_dim)

            # Split and process each expert's output
            outputs = list(inputs_embeds_list)  # Copy
            start_pos = 0
            for j, i in enumerate(active_indices):
                layer = models[i].layers[layer_idx]
                seq_len = inputs_embeds_list[i].shape[1]
                end_pos = start_pos + seq_len

                out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
                out_emb = modeling_gemma._gated_residual(inputs_embeds_list[i], out_emb, gates[j])
                after_residual = out_emb.clone()

                out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond_list[i])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(after_residual, out_emb, gate)

                outputs[i] = out_emb
                start_pos = end_pos

            return tuple(outputs)

        # Run all layers
        inputs_embeds_list = list(inputs_embeds)
        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                result = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx, inputs_embeds_list, attention_mask, position_ids, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                result = compute_layer_complete(
                    layer_idx, inputs_embeds_list, attention_mask, position_ids, adarms_cond,
                )
            inputs_embeds_list = list(result)

        # Final norms
        outputs_embeds = [None, None, None]
        for i in active_indices:
            out_emb, _ = models[i].norm(inputs_embeds_list[i], cond=adarms_cond[i])
            outputs_embeds[i] = out_emb

        return outputs_embeds, None


# ---------------------------------------------------------------------------
# ACoT-VLA Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ACOTConfigPytorch:
    """ACoT-VLA configuration for PyTorch model."""
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    coarse_action_expert_variant: str = "gemma_300m"
    action_expert_variant: str = "gemma_300m"

    action_dim: int = 32
    coarse_action_horizon: int = 50
    action_horizon: int = 30
    max_token_len: int = 200
    pi05: bool = True

    adopt_explicit_action_reasoner: bool = False
    adopt_implicit_action_reasoner: bool = False
    query_based_implicit_extractor: bool = False
    attention_pooling_implicit_extractor: bool = False
    downsample_based_implicit_extractor: bool = False

    # One-step flow inference (OFP) parameters
    use_one_step_inference: bool = False
    exploration_std: float = 0.0
    self_consistency_loss_scale: float = 0.0
    sc_midpoint_samples: int = 1

    pytorch_compile_mode: str | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)


# ---------------------------------------------------------------------------
# Main ACoT-VLA PyTorch Model
# ---------------------------------------------------------------------------


class ACOT_VLAPytorch(nn.Module):
    """ACoT-VLA PyTorch model with dual-expert action reasoning.

    Architecture:
        1. PaliGemma VLM encodes images + language → prefix features
        2. Coarse Action Reasoner generates coarse action plan (explicit reasoning)
        3. Implicit Action Reasoner extracts latent cues from VLM features
        4. Fine Action Expert produces precise actions conditioned on reasoning
        5. Flow matching loss for training; Euler ODE sampling for inference
    """

    def __init__(self, config: ACOTConfigPytorch):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        # Get expert configs
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        coarse_action_expert_config = _gemma.get_config(config.coarse_action_expert_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Three-expert model
        self.dual_expert_model = PaliGemmaWithDualExpertModel(
            paligemma_config,
            coarse_action_expert_config,
            action_expert_config,
            use_adarms=[False, True, True] if self.pi05 else [False, False, False],
            precision=config.dtype,
        )

        # Action projection layers
        self.coarse_action_in_proj = nn.Linear(config.action_dim, coarse_action_expert_config.width)
        self.coarse_action_out_proj = nn.Linear(coarse_action_expert_config.width, config.action_dim)
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # Time/state MLPs
        if self.pi05:
            self.coarse_time_mlp_in = nn.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.coarse_time_mlp_out = nn.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.coarse_action_time_mlp_in = nn.Linear(2 * coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.coarse_action_time_mlp_out = nn.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Store dimensions for later use
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.coarse_action_horizon = config.coarse_action_horizon

        # --- Explicit action reasoner ---
        self.adopt_explicit_action_reasoner = config.adopt_explicit_action_reasoner
        if self.adopt_explicit_action_reasoner:
            self.explicit_action_reasoner = UnifiedAttentionModule(
                in_dim_1=action_expert_config.width,
                in_dim_2=coarse_action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
            )

        # --- Implicit action reasoner ---
        self.adopt_implicit_action_reasoner = config.adopt_implicit_action_reasoner
        self.query_based_implicit_extractor = config.query_based_implicit_extractor
        self.attention_pooling_implicit_extractor = config.attention_pooling_implicit_extractor
        self.downsample_based_implicit_extractor = config.downsample_based_implicit_extractor

        if self.adopt_implicit_action_reasoner:
            if self.query_based_implicit_extractor:
                self.implicit_action_reasoner = LearnableQueryExtractor(
                    num_queries=8,
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                    group_size=3,
                )
            elif self.attention_pooling_implicit_extractor:
                self.implicit_action_reasoner = AttentionPoolingExtractor(
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                    group_size=3,
                )
            elif self.downsample_based_implicit_extractor:
                self.implicit_action_reasoner = DownsampleExtractor(
                    num_queries=1,
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    downsample_dim=paligemma_config.head_dim // 2,
                    heads=paligemma_config.num_heads,
                    group_size=3,
                )
            else:
                raise ValueError("At least one extractor type must be selected when adopt_implicit_action_reasoner is True.")

            self.implicit_action_reasoner_interact = UnifiedAttentionModule(
                in_dim_1=action_expert_config.width,
                in_dim_2=action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
            )

        # --- Fusion modules ---
        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            self.explicit_action_reason_proj = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.implicit_action_reason_proj = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_reasoning_fusion = UnifiedAttentionModule(
                in_dim_1=2 * action_expert_config.width,
                in_dim_2=2 * action_expert_config.width,
                out_dim=action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
                num_heads=4,
            )
        elif self.adopt_explicit_action_reasoner or self.adopt_implicit_action_reasoner:
            self.action_reasoning_fusion = MLP(
                input_dim=2 * action_expert_config.width,
                hidden_dim=action_expert_config.width,
                output_dim=action_expert_config.width,
                activate=False,
            )

        # Gradient checkpointing
        self.gradient_checkpointing_enabled = False

        # One-step flow inference (OFP) config
        self.use_one_step_inference = config.use_one_step_inference
        self.exploration_std = config.exploration_std
        self.self_consistency_loss_scale = config.self_consistency_loss_scale
        self.sc_midpoint_samples = config.sc_midpoint_samples

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
            self.sample_actions_one_step = torch.compile(self.sample_actions_one_step, mode=config.pytorch_compile_mode)

        logger.info(f"ACoT-VLA PyTorch initialized: pi05={self.pi05}, "
                     f"explicit_reasoner={self.adopt_explicit_action_reasoner}, "
                     f"implicit_reasoner={self.adopt_implicit_action_reasoner}, "
                     f"one_step_inference={self.use_one_step_inference}")

    # -----------------------------------------------------------------------
    # Gradient checkpointing
    # -----------------------------------------------------------------------

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        for model in [self.dual_expert_model.paligemma.language_model,
                       self.dual_expert_model.coarse_expert.model,
                       self.dual_expert_model.fine_expert.model]:
            if hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for ACoT-VLA")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        for model in [self.dual_expert_model.paligemma.language_model,
                       self.dual_expert_model.coarse_expert.model,
                       self.dual_expert_model.fine_expert.model]:
            if hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for ACoT-VLA")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # -----------------------------------------------------------------------
    # Attention mask helpers
    # -----------------------------------------------------------------------

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # -----------------------------------------------------------------------
    # Embedding functions
    # -----------------------------------------------------------------------

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images and language tokens for the VLM prefix."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.dual_expert_model.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self._apply_checkpoint(self.dual_expert_model.embed_language_tokens, lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        suf_type: str = "reasoner",
        explicit_action_reason: Tensor | None = None,
        implicit_action_reason: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Embed state, noisy actions, and timestep for an action expert.

        Args:
            state: Robot state [B, action_dim].
            noisy_actions: Noisy actions [B, horizon, action_dim].
            timestep: Diffusion timestep [B].
            suf_type: "reasoner" for coarse expert, "expert" for fine expert.
            explicit_action_reason: Coarse actions for explicit reasoning [B, coarse_horizon, action_dim].
            implicit_action_reason: Implicit features [B, L, width].

        Returns:
            embs, pad_masks, att_masks, adarms_cond
        """
        embs = []
        pad_masks = []
        att_masks = []
        bsize = state.shape[0]
        device = state.device

        # State token (only for pi0 mode)
        if not self.pi05:
            state_emb = self._apply_checkpoint(self.state_proj, state.float())
            embs.append(state_emb[:, None, :])
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Action tokens + time embedding
        if suf_type == "reasoner":
            action_tokens = self.coarse_action_in_proj(noisy_actions)
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.coarse_action_in_proj.out_features,
                min_period=4e-3, max_period=4.0, device=device,
            )
            horizon = self.coarse_action_horizon

            if self.pi05:
                time_emb = F.silu(self.coarse_time_mlp_out(F.silu(self.coarse_time_mlp_in(time_emb))))
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
            else:
                time_tokens = time_emb[:, None, :].expand_as(action_tokens)
                action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
                action_time_tokens = F.silu(self.coarse_action_time_mlp_in(action_time_tokens))
                action_time_tokens = self.coarse_action_time_mlp_out(action_time_tokens)
                action_expert_tokens = action_time_tokens
                adarms_cond = None

        elif suf_type == "expert":
            action_tokens = self.action_in_proj(noisy_actions)
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.action_in_proj.out_features,
                min_period=4e-3, max_period=4.0, device=device,
            )
            horizon = self.action_horizon

            if self.pi05:
                time_emb = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
                action_expert_tokens = action_tokens
                adarms_cond = time_emb
            else:
                time_tokens = time_emb[:, None, :].expand_as(action_tokens)
                action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
                action_time_tokens = F.silu(self.action_time_mlp_in(action_time_tokens))
                action_time_tokens = self.action_time_mlp_out(action_time_tokens)
                action_expert_tokens = action_time_tokens
                adarms_cond = None

            # Apply action reasoning fusion
            if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
                explicit_tokens = self.coarse_action_in_proj(explicit_action_reason)
                aligned_explicit = self.explicit_action_reasoner(action_expert_tokens, explicit_tokens)

                aligned_implicit = self.implicit_action_reasoner_interact(action_expert_tokens, implicit_action_reason)

                explicit_fused = torch.cat([action_expert_tokens, aligned_explicit], dim=-1)
                explicit_fused = self.explicit_action_reason_proj(explicit_fused)

                implicit_fused = torch.cat([action_expert_tokens, aligned_implicit], dim=-1)
                implicit_fused = self.implicit_action_reason_proj(implicit_fused)

                combined = torch.cat([explicit_fused, implicit_fused], dim=-1)
                action_expert_tokens = self.action_reasoning_fusion(combined, combined)

            elif self.adopt_explicit_action_reasoner:
                explicit_tokens = self.coarse_action_in_proj(explicit_action_reason)
                aligned_explicit = self.explicit_action_reasoner(action_expert_tokens, explicit_tokens)
                combined = torch.cat([action_expert_tokens, aligned_explicit], dim=-1)
                action_expert_tokens = self.action_reasoning_fusion(combined)

            elif self.adopt_implicit_action_reasoner:
                aligned_implicit = self.implicit_action_reasoner_interact(action_expert_tokens, implicit_action_reason)
                combined = torch.cat([action_expert_tokens, aligned_implicit], dim=-1)
                action_expert_tokens = self.action_reasoning_fusion(combined)

        else:
            raise ValueError(f"Unknown suffix type: {suf_type}")

        embs.append(action_expert_tokens)
        pad_masks.append(torch.ones(bsize, horizon, dtype=torch.bool, device=device))
        att_masks += [1] + [0] * (horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.float32, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # -----------------------------------------------------------------------
    # Training forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        observation,
        actions: Tensor,
        coarse_actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Training forward pass with dual flow matching loss.

        Args:
            observation: Observation dict from data loader.
            actions: Fine-grained target actions [B, action_horizon, action_dim].
            coarse_actions: Coarse target actions [B, coarse_action_horizon, action_dim].
            noise: Optional noise tensor.
            time: Optional timestep tensor.

        Returns:
            Scalar loss (sum of coarse + fine flow matching losses).
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        device = actions.device

        if noise is None:
            coarse_noise = self.sample_noise(coarse_actions.shape, device)
            expert_noise = self.sample_noise(actions.shape, device)
        else:
            coarse_noise, expert_noise = noise.chunk(2, dim=-2)

        if time is None:
            time = self.sample_time(actions.shape[0], device)

        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_ref_t = time_expanded * coarse_noise + (1 - time_expanded) * coarse_actions
        u_ref_t = coarse_noise - coarse_actions

        x_expert_t = time_expanded * expert_noise + (1 - time_expanded) * actions
        u_expert_t = expert_noise - actions

        # === Prefix forward (VLM) ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Cast to model dtype
        first_layer = self.dual_expert_model.paligemma.language_model.layers[0]
        if first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        [prefix_out, _, _], _ = self.dual_expert_model.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=False,
        )

        # === Coarse action reasoner forward ===
        explicit_action_reason = None
        if self.adopt_explicit_action_reasoner:
            suffix_coarse_embs, suffix_coarse_pad, suffix_coarse_att, adarms_coarse = self.embed_suffix(
                state, x_ref_t, time, suf_type="reasoner",
            )

            if first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_coarse_embs = suffix_coarse_embs.to(dtype=torch.bfloat16)

            pad_masks_coarse = torch.cat([prefix_pad_masks, suffix_coarse_pad], dim=1)
            att_masks_coarse = torch.cat([prefix_att_masks, suffix_coarse_att], dim=1)
            att_2d_coarse = make_att_2d_masks(pad_masks_coarse, att_masks_coarse)
            position_ids_coarse = torch.cumsum(pad_masks_coarse, dim=1) - 1
            att_4d_coarse = self._prepare_attention_masks_4d(att_2d_coarse)

            [_, suffix_coarse_out, _], _ = self.dual_expert_model.forward(
                attention_mask=att_4d_coarse,
                position_ids=position_ids_coarse,
                inputs_embeds=[prefix_embs, suffix_coarse_embs, None],
                use_cache=False,
                adarms_cond=[None, adarms_coarse, None],
            )

            v_ref_t = self.coarse_action_out_proj(suffix_coarse_out[:, -self.coarse_action_horizon:])
            # Teacher forcing: use ground truth coarse actions as explicit reason
            explicit_action_reason = coarse_actions

        # === Implicit action reasoner ===
        implicit_action_reason = None
        if self.adopt_implicit_action_reasoner:
            # Extract features from VLM prefix KV cache
            # We use the prefix output as a proxy for KV cache features
            # In JAX, this comes from the actual KV cache; here we use the hidden states
            # Reshape to [B, L, T, D] format expected by extractors
            # For simplicity, we use prefix_out reshaped as (B, depth, seq_len, head_dim)
            depth = self.dual_expert_model.paligemma.config.text_config.num_hidden_layers
            head_dim = self.dual_expert_model.paligemma.config.text_config.head_dim
            seq_len = prefix_out.shape[1]

            # Approximate: use prefix_out as both K and V
            # Shape: [B, depth, seq_len, head_dim] — simplified from actual KV cache
            prefix_kv = prefix_out.unsqueeze(1).expand(-1, depth, -1, -1)
            if prefix_kv.shape[-1] != head_dim:
                # Project to head_dim if needed
                prefix_kv = prefix_kv[..., :head_dim]

            implicit_action_reason = self.implicit_action_reasoner(prefix_kv, prefix_kv)

        # === Fine action expert forward ===
        suffix_expert_embs, suffix_expert_pad, suffix_expert_att, adarms_expert = self.embed_suffix(
            state, x_expert_t, time,
            suf_type="expert",
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
        )

        if first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_expert_embs = suffix_expert_embs.to(dtype=torch.bfloat16)

        pad_masks_expert = torch.cat([prefix_pad_masks, suffix_expert_pad], dim=1)
        att_masks_expert = torch.cat([prefix_att_masks, suffix_expert_att], dim=1)
        att_2d_expert = make_att_2d_masks(pad_masks_expert, att_masks_expert)
        position_ids_expert = torch.cumsum(pad_masks_expert, dim=1) - 1
        att_4d_expert = self._prepare_attention_masks_4d(att_2d_expert)

        [_, _, suffix_expert_out], _ = self.dual_expert_model.forward(
            attention_mask=att_4d_expert,
            position_ids=position_ids_expert,
            inputs_embeds=[prefix_embs, None, suffix_expert_embs],
            use_cache=False,
            adarms_cond=[None, None, adarms_expert],
        )

        v_expert_t = self.action_out_proj(suffix_expert_out[:, -self.action_horizon:])

        # === Compute loss ===
        if self.adopt_explicit_action_reasoner:
            # Dual loss: coarse + fine
            loss_coarse = F.mse_loss(u_ref_t, v_ref_t)
            loss_fine = F.mse_loss(u_expert_t, v_expert_t)
            return loss_coarse + loss_fine
        else:
            # Single loss: fine only
            return F.mse_loss(u_expert_t, v_expert_t)

    # -----------------------------------------------------------------------
    # Inference: action sampling via flow matching ODE
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        device: torch.device,
        observation,
        noise: Tensor | None = None,
        ref_noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> dict[str, Tensor]:
        """Sample actions via Euler ODE integration.

        Args:
            device: Target device.
            observation: Observation dict.
            noise: Optional noise for fine expert.
            ref_noise: Optional noise for coarse reasoner.
            num_steps: Number of ODE steps.

        Returns:
            Dict with "actions" key, and optionally "coarse_actions".
        """
        bsize = observation.state.shape[0]

        if ref_noise is None:
            ref_noise = self.sample_noise((bsize, self.coarse_action_horizon, self.action_dim), device)
        if noise is None:
            noise = self.sample_noise((bsize, self.action_horizon, self.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # === Prefix forward ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        [prefix_out, _, _], _ = self.dual_expert_model.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=False,
        )

        # === Implicit reasoning (computed once) ===
        implicit_action_reason = None
        if self.adopt_implicit_action_reasoner:
            depth = self.dual_expert_model.paligemma.config.text_config.num_hidden_layers
            head_dim = self.dual_expert_model.paligemma.config.text_config.head_dim
            prefix_kv = prefix_out.unsqueeze(1).expand(-1, depth, -1, -1)
            if prefix_kv.shape[-1] != head_dim:
                prefix_kv = prefix_kv[..., :head_dim]
            implicit_action_reason = self.implicit_action_reasoner(prefix_kv, prefix_kv)

        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

        # === Coarse action reasoner ODE ===
        explicit_action_reason = None
        if self.adopt_explicit_action_reasoner:
            x_t = ref_noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)

            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                    state, x_t, expanded_time, suf_type="reasoner",
                )

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1
                att_4d = self._prepare_attention_masks_4d(att_2d)

                [_, suffix_out, _], _ = self.dual_expert_model.forward(
                    attention_mask=att_4d,
                    position_ids=position_ids,
                    inputs_embeds=[prefix_embs, suffix_embs, None],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond, None],
                )

                v_t = self.coarse_action_out_proj(suffix_out[:, -self.coarse_action_horizon:])
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor

            explicit_action_reason = x_t

        # === Fine action expert ODE ===
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time, suf_type="expert",
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
            )

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(att_2d)

            [_, _, suffix_out], _ = self.dual_expert_model.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None, adarms_cond],
            )

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            x_t = x_t + dt_tensor * v_t
            time += dt_tensor

        result = {"actions": x_t}
        if self.adopt_explicit_action_reasoner:
            result["coarse_actions"] = explicit_action_reason
        return result

    # -----------------------------------------------------------------------
    # One-step flow inference (OFP, NFE=1)
    # Ref: PolicyFlowOneStep.compute_one_step_velocity / draw_actions
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def compute_one_step_velocity(
        self,
        device: torch.device,
        observation,
        noise: Tensor | None = None,
        ref_noise: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute velocity at t=0 for single-step flow inference (OFP, NFE=1).

        One-step action:  a = z_0 + u(z_0, t=0 | o)

        Args:
            device: Target device.
            observation: Observation dict.
            noise: Optional noise for fine expert [B, action_horizon, action_dim].
            ref_noise: Optional noise for coarse reasoner [B, coarse_action_horizon, action_dim].

        Returns:
            (vel_dict, std_dict) where
                vel_dict has "velocity" (fine) and optionally "coarse_velocity"
                std_dict has "std" (fine) and optionally "coarse_std"
        """
        bsize = observation.state.shape[0]

        if ref_noise is None:
            ref_noise = self.sample_noise((bsize, self.coarse_action_horizon, self.action_dim), device)
        if noise is None:
            noise = self.sample_noise((bsize, self.action_horizon, self.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # t=0 for all samples
        t_zero = torch.zeros(bsize, dtype=torch.float32, device=device)

        # === Prefix forward (VLM) ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        [prefix_out, _, _], _ = self.dual_expert_model.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=False,
        )

        # === Implicit reasoning (computed once) ===
        implicit_action_reason = None
        if self.adopt_implicit_action_reasoner:
            depth = self.dual_expert_model.paligemma.config.text_config.num_hidden_layers
            head_dim = self.dual_expert_model.paligemma.config.text_config.head_dim
            prefix_kv = prefix_out.unsqueeze(1).expand(-1, depth, -1, -1)
            if prefix_kv.shape[-1] != head_dim:
                prefix_kv = prefix_kv[..., :head_dim]
            implicit_action_reason = self.implicit_action_reasoner(prefix_kv, prefix_kv)

        vel_dict: dict[str, Tensor] = {}
        std_dict: dict[str, Tensor] = {}

        # === Coarse action reasoner: velocity at t=0 ===
        coarse_velocity = None
        if self.adopt_explicit_action_reasoner:
            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
                state, ref_noise, t_zero, suf_type="reasoner",
            )
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(att_2d)

            [_, suffix_out, _], _ = self.dual_expert_model.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, suffix_embs, None],
                use_cache=False,
                adarms_cond=[None, adarms_cond, None],
            )
            coarse_velocity = self.coarse_action_out_proj(suffix_out[:, -self.coarse_action_horizon:])
            vel_dict["coarse_velocity"] = coarse_velocity
            std_dict["coarse_std"] = torch.ones_like(coarse_velocity)

        # === Fine action expert: velocity at t=0 ===
        # For one-step, use coarse output as explicit reason (if available)
        explicit_action_reason = coarse_velocity if self.adopt_explicit_action_reasoner else None

        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(
            state, noise, t_zero, suf_type="expert",
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d)

        [_, _, suffix_out], _ = self.dual_expert_model.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs, None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None, adarms_cond],
        )
        fine_velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        vel_dict["velocity"] = fine_velocity
        std_dict["std"] = torch.ones_like(fine_velocity)

        return vel_dict, std_dict

    @torch.no_grad()
    def sample_actions_one_step(
        self,
        device: torch.device,
        observation,
        noise: Tensor | None = None,
        ref_noise: Tensor | None = None,
        exploration_std: float = 0.0,
    ) -> dict[str, Tensor]:
        """Sample actions via one-step flow (OFP, NFE=1).

        Implements:  action = z_0 + u(z_0, t=0 | o)

        This is significantly faster than multi-step Euler ODE while still
        leveraging the learned velocity field. Inspired by PolicyFlowOneStep.

        Args:
            device: Target device.
            observation: Observation dict.
            noise: Optional noise for fine expert [B, action_horizon, action_dim].
            ref_noise: Optional noise for coarse reasoner [B, coarse_action_horizon, action_dim].
            exploration_std: Std of additional Gaussian exploration noise.
                If 0, uses pure deterministic one-step prediction.

        Returns:
            Dict with "actions" key, and optionally "coarse_actions".
        """
        bsize = observation.state.shape[0]

        if ref_noise is None:
            ref_noise = self.sample_noise((bsize, self.coarse_action_horizon, self.action_dim), device)
        if noise is None:
            noise = self.sample_noise((bsize, self.action_horizon, self.action_dim), device)

        # Compute velocity at t=0
        vel_dict, std_dict = self.compute_one_step_velocity(
            device=device,
            observation=observation,
            noise=noise,
            ref_noise=ref_noise,
        )

        # One-step: action = z_0 + velocity(z_0, t=0 | o)
        coarse_actions = None
        if self.adopt_explicit_action_reasoner and "coarse_velocity" in vel_dict:
            coarse_actions = ref_noise + vel_dict["coarse_velocity"]

        actions = noise + vel_dict["velocity"]

        # Optional exploration noise
        if exploration_std > 0:
            delta_dist = torch.distributions.Normal(
                torch.zeros_like(actions),
                torch.ones_like(actions) * exploration_std,
            )
            actions = actions + delta_dist.sample()
            if coarse_actions is not None:
                coarse_actions = coarse_actions + delta_dist.sample()[:, :self.coarse_action_horizon]

        result = {"actions": actions}
        if coarse_actions is not None:
            result["coarse_actions"] = coarse_actions
        return result

    # -----------------------------------------------------------------------
    # Self-consistency loss (OFP path compression)
    # Ref: PolicyFlowOneStep._compute_self_consistency_loss
    # -----------------------------------------------------------------------

    def compute_self_consistency_loss(
        self,
        observation,
        actions_target: Tensor,
        coarse_actions_target: Tensor,
        num_midpoint_samples: int = 1,
    ) -> Tensor:
        """Self-consistency loss for one-step flow training (OFP).

        Enforces that the velocity field is consistent along the flow path:
            For any t in (0, 1), the predicted action from x_t should match
            the predicted action from x_0 (or x_1).

        Loss = || (x_t + t * u(x_t, t|o)) - (x_0 + u(x_0, 0|o)) ||^2

        This encourages the velocity field to define a "straight" path,
        improving one-step prediction quality.

        Args:
            observation: Observation dict.
            actions_target: Fine target actions [B, action_horizon, action_dim].
            coarse_actions_target: Coarse target actions [B, coarse_action_horizon, action_dim].
            num_midpoint_samples: Number of random t values to sample.

        Returns:
            Scalar self-consistency loss.
        """
        device = actions_target.device
        bsize = actions_target.shape[0]

        # Sample noise (shared x_0, x_1)
        coarse_noise = self.sample_noise(coarse_actions_target.shape, device)
        expert_noise = self.sample_noise(actions_target.shape, device)

        # Preprocess observation
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # === Prefix forward (shared) ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        first_layer = self.dual_expert_model.paligemma.language_model.layers[0]
        is_bf16 = first_layer.self_attn.q_proj.weight.dtype == torch.bfloat16

        [prefix_out, _, _], _ = self.dual_expert_model.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=False,
        )

        # === Implicit reasoning ===
        implicit_action_reason = None
        if self.adopt_implicit_action_reasoner:
            depth = self.dual_expert_model.paligemma.config.text_config.num_hidden_layers
            head_dim = self.dual_expert_model.paligemma.config.text_config.head_dim
            prefix_kv = prefix_out.unsqueeze(1).expand(-1, depth, -1, -1)
            if prefix_kv.shape[-1] != head_dim:
                prefix_kv = prefix_kv[..., :head_dim]
            implicit_action_reason = self.implicit_action_reasoner(prefix_kv, prefix_kv)

        sc_loss = torch.tensor(0.0, device=device)

        for _ in range(num_midpoint_samples):
            # Sample random t in (0, 1)
            t_mid = torch.rand(bsize, device=device) * 0.8 + 0.1  # avoid extremes

            # Interpolate x_t = t * x_1 + (1 - t) * x_0  (flow matching convention: x_0=noise, x_1=target)
            t_expanded = t_mid[:, None, None]
            x_coarse_t = t_expanded * coarse_noise + (1 - t_expanded) * coarse_actions_target
            x_expert_t = t_expanded * expert_noise + (1 - t_expanded) * actions_target

            t_zero = torch.zeros(bsize, dtype=torch.float32, device=device)

            # --- Velocity at t=0 (one-step prediction) ---
            # Coarse at t=0
            coarse_velocity_at_0 = None
            if self.adopt_explicit_action_reasoner:
                suffix_embs_0, suffix_pad_0, suffix_att_0, adarms_0 = self.embed_suffix(
                    state, coarse_noise, t_zero, suf_type="reasoner",
                )
                pad_masks_0 = torch.cat([prefix_pad_masks, suffix_pad_0], dim=1)
                att_masks_0 = torch.cat([prefix_att_masks, suffix_att_0], dim=1)
                att_2d_0 = make_att_2d_masks(pad_masks_0, att_masks_0)
                position_ids_0 = torch.cumsum(pad_masks_0, dim=1) - 1
                att_4d_0 = self._prepare_attention_masks_4d(att_2d_0)

                if is_bf16:
                    suffix_embs_0 = suffix_embs_0.to(dtype=torch.bfloat16)

                [_, suffix_out_0, _], _ = self.dual_expert_model.forward(
                    attention_mask=att_4d_0,
                    position_ids=position_ids_0,
                    inputs_embeds=[prefix_embs, suffix_embs_0, None],
                    use_cache=False,
                    adarms_cond=[None, adarms_0, None],
                )
                coarse_velocity_at_0 = self.coarse_action_out_proj(suffix_out_0[:, -self.coarse_action_horizon:])

            # Fine at t=0
            explicit_reason_0 = coarse_velocity_at_0 if self.adopt_explicit_action_reasoner else None
            suffix_embs_f0, suffix_pad_f0, suffix_att_f0, adarms_f0 = self.embed_suffix(
                state, expert_noise, t_zero, suf_type="expert",
                explicit_action_reason=explicit_reason_0,
                implicit_action_reason=implicit_action_reason,
            )
            pad_masks_f0 = torch.cat([prefix_pad_masks, suffix_pad_f0], dim=1)
            att_masks_f0 = torch.cat([prefix_att_masks, suffix_att_f0], dim=1)
            att_2d_f0 = make_att_2d_masks(pad_masks_f0, att_masks_f0)
            position_ids_f0 = torch.cumsum(pad_masks_f0, dim=1) - 1
            att_4d_f0 = self._prepare_attention_masks_4d(att_2d_f0)

            if is_bf16:
                suffix_embs_f0 = suffix_embs_f0.to(dtype=torch.bfloat16)

            [_, _, suffix_out_f0], _ = self.dual_expert_model.forward(
                attention_mask=att_4d_f0,
                position_ids=position_ids_f0,
                inputs_embeds=[prefix_embs, None, suffix_embs_f0],
                use_cache=False,
                adarms_cond=[None, None, adarms_f0],
            )
            fine_velocity_at_0 = self.action_out_proj(suffix_out_f0[:, -self.action_horizon:])

            # --- Velocity at t_mid ---
            # Coarse at t_mid
            coarse_velocity_at_t = None
            if self.adopt_explicit_action_reasoner:
                suffix_embs_t, suffix_pad_t, suffix_att_t, adarms_t = self.embed_suffix(
                    state, x_coarse_t, t_mid, suf_type="reasoner",
                )
                pad_masks_t = torch.cat([prefix_pad_masks, suffix_pad_t], dim=1)
                att_masks_t = torch.cat([prefix_att_masks, suffix_att_t], dim=1)
                att_2d_t = make_att_2d_masks(pad_masks_t, att_masks_t)
                position_ids_t = torch.cumsum(pad_masks_t, dim=1) - 1
                att_4d_t = self._prepare_attention_masks_4d(att_2d_t)

                if is_bf16:
                    suffix_embs_t = suffix_embs_t.to(dtype=torch.bfloat16)

                [_, suffix_out_t, _], _ = self.dual_expert_model.forward(
                    attention_mask=att_4d_t,
                    position_ids=position_ids_t,
                    inputs_embeds=[prefix_embs, suffix_embs_t, None],
                    use_cache=False,
                    adarms_cond=[None, adarms_t, None],
                )
                coarse_velocity_at_t = self.coarse_action_out_proj(suffix_out_t[:, -self.coarse_action_horizon:])

            # Fine at t_mid
            explicit_reason_t = coarse_velocity_at_t if self.adopt_explicit_action_reasoner else None
            suffix_embs_ft, suffix_pad_ft, suffix_att_ft, adarms_ft = self.embed_suffix(
                state, x_expert_t, t_mid, suf_type="expert",
                explicit_action_reason=explicit_reason_t,
                implicit_action_reason=implicit_action_reason,
            )
            pad_masks_ft = torch.cat([prefix_pad_masks, suffix_pad_ft], dim=1)
            att_masks_ft = torch.cat([prefix_att_masks, suffix_att_ft], dim=1)
            att_2d_ft = make_att_2d_masks(pad_masks_ft, att_masks_ft)
            position_ids_ft = torch.cumsum(pad_masks_ft, dim=1) - 1
            att_4d_ft = self._prepare_attention_masks_4d(att_2d_ft)

            if is_bf16:
                suffix_embs_ft = suffix_embs_ft.to(dtype=torch.bfloat16)

            [_, _, suffix_out_ft], _ = self.dual_expert_model.forward(
                attention_mask=att_4d_ft,
                position_ids=position_ids_ft,
                inputs_embeds=[prefix_embs, None, suffix_embs_ft],
                use_cache=False,
                adarms_cond=[None, None, adarms_ft],
            )
            fine_velocity_at_t = self.action_out_proj(suffix_out_ft[:, -self.action_horizon:])

            # --- Self-consistency constraint ---
            # Predicted action from t=0: a_0 = x_0 + u(x_0, 0|o)
            # Predicted action from t_mid: a_t = x_t + t_mid * u(x_t, t|o)
            # SC loss: ||a_t - a_0||^2
            pred_action_from_0 = expert_noise + fine_velocity_at_0
            pred_action_from_t = x_expert_t + t_mid[:, None, None] * fine_velocity_at_t
            sc_loss = sc_loss + F.mse_loss(pred_action_from_t, pred_action_from_0.detach())

            if self.adopt_explicit_action_reasoner and coarse_velocity_at_0 is not None:
                pred_coarse_from_0 = coarse_noise + coarse_velocity_at_0
                pred_coarse_from_t = x_coarse_t + t_mid[:, None, None] * coarse_velocity_at_t
                sc_loss = sc_loss + F.mse_loss(pred_coarse_from_t, pred_coarse_from_0.detach())

        return sc_loss / num_midpoint_samples

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------

    def count_trainable_params(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

import logging
from typing import Literal, Optional

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


# ---------------------------------------------------------------------------
# Compatibility shim: transformers ≥5.x removed the top-level `.language_model`
# and `.vision_tower` properties from `PaliGemmaForConditionalGeneration`.
# The code throughout this module (and `pi0_pytorch.py`) still uses the old
# paths, so we re-inject lightweight @property accessors that delegate to
# `self.model.<attr>`.
# ---------------------------------------------------------------------------
def _patch_paligemma_properties() -> None:
    """Add `.language_model` / `.vision_tower` / `.multi_modal_projector` properties
    to `PaliGemmaForConditionalGeneration` if they are missing."""
    if getattr(PaliGemmaForConditionalGeneration, "_openpi_props_patched", False):
        return

    if not hasattr(PaliGemmaForConditionalGeneration, "language_model"):
        PaliGemmaForConditionalGeneration.language_model = property(
            lambda self: self.model.language_model
        )
    if not hasattr(PaliGemmaForConditionalGeneration, "vision_tower"):
        PaliGemmaForConditionalGeneration.vision_tower = property(
            lambda self: self.model.vision_tower
        )
    if not hasattr(PaliGemmaForConditionalGeneration, "multi_modal_projector"):
        PaliGemmaForConditionalGeneration.multi_modal_projector = property(
            lambda self: self.model.multi_modal_projector
        )

    PaliGemmaForConditionalGeneration._openpi_props_patched = True  # noqa: SLF001


_patch_paligemma_properties()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compatibility shim: monkey-patch GemmaRMSNorm to support adaRMS
# (adaptive RMSNorm with `cond` parameter).
#
# The `transformers_replace` patches for Gemma2 add a `cond_dim` parameter to
# GemmaRMSNorm and a `cond` kwarg to its forward() method.  Transformers ≥5.x
# does not have these, so we patch the class in-place.
#
# Two things need to happen:
#   1. extend __init__ to accept `cond_dim` (needed during model construction)
#   2. extend forward() to accept an optional `cond` kwarg and return a
#      (hidden_states, gate) tuple (needed at runtime by gemma_pytorch.py)
# ---------------------------------------------------------------------------
_GemmaRMSNorm = modeling_gemma.GemmaRMSNorm
_ORIG_RMSNORM_INIT = _GemmaRMSNorm.__init__
_ORIG_RMSNORM_FORWARD = _GemmaRMSNorm.__forward__ if hasattr(_GemmaRMSNorm, "__forward__") else _GemmaRMSNorm.forward


def _patched_rmsnorm_init(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
    """Extended __init__ that optionally creates an adaptive dense layer."""
    _ORIG_RMSNORM_INIT(self, dim, eps)
    self.cond_dim = cond_dim
    if cond_dim is not None:
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
    else:
        self.dense = None


def _patched_rmsnorm_forward(self, x, cond=None):
    """Extended forward that supports an optional `cond` (adaRMS conditioning).

    Returns a tensor in the standard path (matching upstream transformers) and
    returns a `(hidden_states, gate)` tuple only when adaRMS conditioning is used.
    """
    dtype = x.dtype
    # RMSNorm computation (in float32 for numerical stability)
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + self.eps)

    if cond is None or self.dense is None:
        # Standard RMSNorm path: preserve the upstream API.
        out = normed * (1.0 + self.weight.float())
        return out.to(dtype)

    # adaRMS path: generate scale / shift / gate from the conditioning vector
    modulation = self.dense(cond.to(next(self.parameters()).dtype))
    if x.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
    out = normed * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


# Only patch if the native GemmaRMSNorm doesn't already support cond
if not hasattr(_GemmaRMSNorm, "dense"):
    _GemmaRMSNorm.__init__ = _patched_rmsnorm_init
    _GemmaRMSNorm.forward = _patched_rmsnorm_forward
    # Also patch the decoder-layer class so layers are created with cond_dim
    _GemmaDecoderLayer = modeling_gemma.GemmaDecoderLayer
    _ORIG_DECODER_INIT = _GemmaDecoderLayer.__init__

    def _patched_decoder_init(self, config, layer_idx):
        _ORIG_DECODER_INIT(self, config, layer_idx)
        # Re-create layernorms with cond_dim if use_adarms is enabled
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        if cond_dim is not None:
            self.input_layernorm = _GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
            self.post_attention_layernorm = _GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

    _GemmaDecoderLayer.__init__ = _patched_decoder_init

    # Patch GemmaModel.__init__ for the final norm layer
    _GemmaModel = modeling_gemma.GemmaModel
    _ORIG_GEMMA_MODEL_INIT = _GemmaModel.__init__

    def _patched_gemma_model_init(self, config):
        _ORIG_GEMMA_MODEL_INIT(self, config)
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        if cond_dim is not None and hasattr(self, "norm"):
            self.norm = _GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    _GemmaModel.__init__ = _patched_gemma_model_init

    logging.getLogger(__name__).info("Patched GemmaRMSNorm with adaRMS (cond) support")

# Inject _gated_residual into modeling_gemma if missing (needed by
# gemma_pytorch.py and acot_vla_pytorch.py which call
# modeling_gemma._gated_residual(...)).
if not hasattr(modeling_gemma, "_gated_residual"):

    def _gated_residual(x, y, gate):
        """Gated residual: x + y * gate if gate is provided, else x + y."""
        if gate is None:
            return x + y
        return x + y * gate

    modeling_gemma._gated_residual = _gated_residual  # noqa: SLF001
# ---------------------------------------------------------------------------


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
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

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
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

    def embed_image(self, image: torch.Tensor):
        result = self.paligemma.model.get_image_features(image)
        # transformers ≥5.x returns BaseModelOutputWithPooling; the projected
        # image features live in .pooler_output (same as the library's own code).
        if hasattr(result, "pooler_output") and result.pooler_output is not None:
            return result.pooler_output
        if isinstance(result, torch.Tensor):
            return result
        if hasattr(result, "last_hidden_state"):
            return result.last_hidden_state
        return result

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

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

        def _match_dtype(embeds: torch.Tensor | None, target_module: nn.Module | None) -> torch.Tensor | None:
            if embeds is None or target_module is None:
                return embeds
            target_dtype = next(target_module.parameters()).dtype
            if embeds.dtype != target_dtype:
                embeds = embeds.to(dtype=target_dtype)
            return embeds

        if inputs_embeds[1] is None:
            prefix_inputs = _match_dtype(inputs_embeds[0], self.paligemma.language_model)
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=prefix_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Suffix-only path: expert processes suffix tokens using VLM's cached KV.
            # We must manually iterate through layers to apply adaRMS conditioning
            # (adarms_cond) and gated residuals at each layer, which HuggingFace's
            # GemmaModel.forward() does NOT support.
            suffix_inputs = _match_dtype(inputs_embeds[1], self.gemma_expert.model)
            expert = self.gemma_expert.model
            expert_adarms = adarms_cond[1] if adarms_cond is not None else None

            hidden_states = suffix_inputs
            position_embeddings = expert.rotary_emb(hidden_states, position_ids=position_ids)

            for layer in expert.layers:
                residual = hidden_states
                hidden_states, gate = layer.input_layernorm(hidden_states, cond=expert_adarms)

                weight_dtype = layer.self_attn.q_proj.weight.dtype
                if hidden_states.dtype != weight_dtype:
                    hidden_states = hidden_states.to(dtype=weight_dtype)

                # Self-attention with cached VLM KV
                hidden_states_attn, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                hidden_states = modeling_gemma._gated_residual(residual, hidden_states_attn, gate)

                # MLP
                residual = hidden_states
                hidden_states, gate = layer.post_attention_layernorm(hidden_states, cond=expert_adarms)
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(dtype=torch.bfloat16)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)

            suffix_output, _ = expert.norm(hidden_states, cond=expert_adarms)
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    weight_dtype = layer.self_attn.q_proj.weight.dtype
                    if hidden_states.dtype != weight_dtype:
                        hidden_states = hidden_states.to(dtype=weight_dtype)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

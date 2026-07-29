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

    Returns (hidden_states, gate) where gate is None when cond is None.
    """
    dtype = x.dtype
    # RMSNorm computation (in float32 for numerical stability)
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + self.eps)

    if cond is None or self.dense is None:
        # Standard RMSNorm path
        out = normed * (1.0 + self.weight.float())
        return out.to(dtype), None

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
    _ORIG_DECODER_FORWARD = _GemmaDecoderLayer.forward

    def _patched_decoder_init(self, config, layer_idx):
        _ORIG_DECODER_INIT(self, config, layer_idx)
        # Re-create layernorms with cond_dim if use_adarms is enabled
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        if cond_dim is not None:
            self.input_layernorm = _GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
            self.post_attention_layernorm = _GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

    def _patched_decoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        adarms_cond=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    _GemmaDecoderLayer.__init__ = _patched_decoder_init
    _GemmaDecoderLayer.forward = _patched_decoder_forward

    # Patch GemmaModel.__init__ for the final norm layer
    _GemmaModel = modeling_gemma.GemmaModel
    _ORIG_GEMMA_MODEL_INIT = _GemmaModel.__init__
    _ORIG_GEMMA_MODEL_FORWARD = _GemmaModel.forward

    def _patched_gemma_model_init(self, config):
        _ORIG_GEMMA_MODEL_INIT(self, config)
        cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        if cond_dim is not None and hasattr(self, "norm"):
            self.norm = _GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def _patched_gemma_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        adarms_cond=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = modeling_gemma.DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = modeling_gemma.create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return modeling_gemma.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    _GemmaModel.__init__ = _patched_gemma_model_init
    _GemmaModel.forward = _patched_gemma_model_forward

    _GemmaForCausalLM = modeling_gemma.GemmaForCausalLM
    _ORIG_GEMMA_CAUSAL_LM_FORWARD = _GemmaForCausalLM.forward

    def _patched_gemma_causal_lm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        adarms_cond=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            adarms_cond=adarms_cond,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return modeling_gemma.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    _GemmaForCausalLM.forward = _patched_gemma_causal_lm_forward

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
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
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
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
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
            if position_ids is None:
                batch_size = inputs_embeds[0].shape[0]
                seq_len = inputs_embeds[0].shape[1]
                position_ids = torch.arange(seq_len, device=inputs_embeds[0].device).unsqueeze(0).expand(batch_size, -1)

            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

            initial_inputs_embeds = inputs_embeds
            position_embeddings = []
            for i, hidden_states in enumerate(initial_inputs_embeds):
                layer = self.paligemma.language_model.layers[0] if i == 0 else self.gemma_expert.model.layers[0]
                hidden_states_for_rope = hidden_states
                if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                    hidden_states_for_rope = hidden_states_for_rope.to(torch.bfloat16)
                cos, sin = models[i].rotary_emb(hidden_states_for_rope, position_ids)
                position_embeddings.append((cos, sin))

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, position_embeddings):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings[i]
                    query_state, key_state = modeling_gemma.apply_rotary_pos_emb(
                        query_state, key_state, cos, sin, unsqueeze_dim=1
                    )

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)
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
                # Reshape using the actual attention projection width from the layer,
                # which mirrors the patched Gemma decoder behavior more closely.
                o_proj = self.paligemma.language_model.layers[layer_idx].self_attn.o_proj
                attn_output_width = getattr(o_proj, "in_features", None)
                if attn_output_width is None:
                    attn_output_width = att_output.shape[-1]
                att_output = att_output.reshape(batch_size, -1, attn_output_width)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    attn_output_slice = att_output[:, start_pos:end_pos]
                    if attn_output_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                        attn_output_slice = attn_output_slice.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(attn_output_slice)

                    # first residual
                    residual = hidden_states
                    out_emb = modeling_gemma._gated_residual(residual, out_emb, gates[i])  # noqa: SLF001
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
                        position_embeddings,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, position_embeddings
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

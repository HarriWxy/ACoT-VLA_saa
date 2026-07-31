"""PyTorch implementation of the physics-conditioned single-step policy."""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger("openpi")


def _is_gemma4_variant(variant: str) -> bool:
    return variant.startswith("gemma4_")


def _has_lora(config) -> bool:
    return "lora" in config.paligemma_variant or "lora" in config.action_expert_variant


class PhysicsEncoderPytorch(nn.Module):
    """Encode continuous physics parameters into conditioning tokens."""

    def __init__(
        self,
        physics_dim: int,
        hidden_dim: int,
        num_tokens: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(physics_dim, hidden_dim)
        self.query_embed = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, physics_params: Tensor) -> Tensor:
        if physics_params.ndim != 2:
            raise ValueError(f"physics_params must have shape [batch, physics_dim], got {physics_params.shape}")

        batch_size = physics_params.shape[0]
        input_dtype = self.input_proj.weight.dtype
        x = F.silu(self.input_proj(physics_params.to(dtype=input_dtype))).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat((x, queries), dim=1)

        for attention, norm in zip(self.attn_layers, self.norms, strict=True):
            normalized = norm(x)
            update, _ = attention(normalized, normalized, normalized, need_weights=False)
            x = x + update

        return self.output_proj(x[:, 1:])


class PhysicsAwareSingleStepPytorch(nn.Module):
    """Single-step action regression with adaRMS conditioning from physics inputs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self._is_gemma4 = _is_gemma4_variant(config.paligemma_variant)

        if self._is_gemma4:
            from openpi.models_pytorch.gemma4_pytorch import Gemma4WithExpertModel  # noqa: PLC0415

            self.paligemma_with_expert = Gemma4WithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True],
                precision=config.dtype,
                gemma4_model_path=getattr(config, "gemma4_model_path", None),
            )
        else:
            self.paligemma_with_expert = PaliGemmaWithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True],
                precision=config.dtype,
            )

        self._is_aligned_expert = (
            self._is_gemma4
            and paligemma_config.width == action_expert_config.width
            and paligemma_config.head_dim == action_expert_config.head_dim
        )

        self.physics_encoder = PhysicsEncoderPytorch(
            physics_dim=config.physics_dim,
            hidden_dim=config.physics_hidden_dim,
            num_tokens=config.num_physics_tokens,
            output_dim=action_expert_config.width,
        )
        self.physics_cond_proj = nn.Sequential(
            nn.Linear(action_expert_config.width, action_expert_config.width),
            nn.SiLU(),
            nn.Linear(action_expert_config.width, action_expert_config.width),
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        if config.use_action_queries:
            self.action_queries = nn.Parameter(torch.randn(config.action_horizon, action_expert_config.width))
            self.action_query_proj = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gradient_checkpointing_enabled = False
        self._lora_injected = False
        if self._is_gemma4 and _has_lora(config):
            self._inject_lora(paligemma_config, action_expert_config)

        torch.set_float32_matmul_precision("high")
        compile_mode = getattr(config, "pytorch_compile_mode", None)
        if compile_mode is not None and os.environ.get("OPENPI_DISABLE_COMPILE", "0") != "1":
            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)

    def _inject_lora(self, vlm_config, expert_config):
        vlm_lora = vlm_config.lora_configs.get("attn") or vlm_config.lora_configs.get("ffn")
        expert_lora = expert_config.lora_configs.get("attn") or expert_config.lora_configs.get("ffn")
        self.paligemma_with_expert.inject_lora(
            vlm_lora_config=vlm_lora if "lora" in self.config.paligemma_variant else None,
            expert_lora_config=expert_lora if "lora" in self.config.action_expert_variant else None,
        )
        self._lora_injected = True

    def _head_modules(self) -> list[nn.Module]:
        modules = [self.physics_encoder, self.physics_cond_proj, self.action_in_proj, self.action_out_proj]
        if hasattr(self, "action_query_proj"):
            modules.append(self.action_query_proj)
        return modules

    def _unfreeze_heads(self) -> int:
        trainable = 0
        for module in self._head_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad=True)
                trainable += parameter.numel()
        if hasattr(self, "action_queries"):
            self.action_queries.requires_grad_(requires_grad=True)
            trainable += self.action_queries.numel()
        return trainable

    def freeze_non_lora_params(self) -> int:
        from openpi.models_pytorch.lora_pytorch import freeze_non_lora_params  # noqa: PLC0415

        freeze_non_lora_params(self.paligemma_with_expert)
        self._unfreeze_heads()
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def freeze_vlm_only(self) -> int:
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad=False)

        if self._is_gemma4:
            expert = self.paligemma_with_expert.gemma4_expert
        else:
            expert = self.paligemma_with_expert.gemma_expert
        for parameter in expert.parameters():
            parameter.requires_grad_(requires_grad=True)
        self._unfreeze_heads()
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def count_trainable_params(self) -> tuple[int, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total

    def _gradient_checkpointing_models(self) -> list[nn.Module]:
        if self._is_gemma4:
            return [
                self.paligemma_with_expert.gemma4_vlm.model,
                self.paligemma_with_expert.gemma4_expert.model,
            ]
        return [
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.paligemma.vision_tower,
            self.paligemma_with_expert.gemma_expert.model,
        ]

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        for model in self._gradient_checkpointing_models():
            if hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        for model in self._gradient_checkpointing_models():
            if hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = False

    def is_gradient_checkpointing_enabled(self) -> bool:
        return self.gradient_checkpointing_enabled

    def _preprocess_observation(self, observation, *, train: bool):
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        physics_params = getattr(processed, "physics_params", None)
        if physics_params is None:
            physics_params = getattr(observation, "physics_params", None)
        return (
            list(processed.images.values()),
            list(processed.image_masks.values()),
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
            processed.state,
            physics_params,
        )

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor | None,
        lang_masks: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = []
        pad_masks = []
        attention_masks = []

        for image, image_mask in zip(images, img_masks, strict=True):
            image_emb = self.paligemma_with_expert.embed_image(image)
            batch_size, num_image_tokens = image_emb.shape[:2]
            embeddings.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(batch_size, num_image_tokens))
            attention_masks.extend([False] * num_image_tokens)

        if lang_tokens is not None:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
            embeddings.append(lang_emb)
            pad_masks.append(lang_masks)
            attention_masks.extend([False] * lang_emb.shape[1])

        if not embeddings:
            raise ValueError("At least one image or language input is required")
        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        batch_size = pad_masks.shape[0]
        attention_masks = torch.tensor(attention_masks, dtype=torch.bool, device=pad_masks.device)
        attention_masks = attention_masks[None].expand(batch_size, -1)
        return embeddings, pad_masks, attention_masks

    def embed_action_suffix(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if hasattr(self, "action_queries"):
            action_tokens = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
            action_tokens = self.action_query_proj(action_tokens)
        else:
            action_tokens = torch.zeros(
                batch_size,
                self.action_horizon,
                self.action_in_proj.out_features,
                device=device,
                dtype=self.action_in_proj.weight.dtype,
            )
        pad_masks = torch.ones(batch_size, self.action_horizon, dtype=torch.bool, device=device)
        attention_masks = torch.zeros_like(pad_masks)
        attention_masks[:, 0] = True
        return action_tokens, pad_masks, attention_masks

    def _encode_physics(self, physics_params: Tensor | None) -> Tensor:
        if physics_params is None:
            raise ValueError("PhysicsAwareSingleStepPytorch requires observation.physics_params")
        physics_tokens = self.physics_encoder(physics_params)
        physics_cond = self.physics_cond_proj(physics_tokens.mean(dim=1))
        return F.silu(physics_cond)

    def _expert_weight_dtype(self) -> torch.dtype:
        if self._is_gemma4:
            layer = self.paligemma_with_expert.gemma4_vlm.model.layers[0]
        else:
            layer = self.paligemma_with_expert.paligemma.language_model.layers[0]
        q_proj = layer.self_attn.q_proj
        if hasattr(q_proj, "base_linear"):
            q_proj = q_proj.base_linear
        return q_proj.weight.dtype

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
        masked_value = -2.3819763e38 if dtype == torch.float32 else torch.finfo(dtype).min
        return torch.where(
            att_2d_masks[:, None],
            torch.zeros((), dtype=dtype, device=att_2d_masks.device),
            torch.full((), masked_value, dtype=dtype, device=att_2d_masks.device),
        )

    def _prepare_attention_masks_for_gemma4(
        self,
        att_2d_masks: Tensor,
        sliding_window: int,
        *,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        full_mask = self._prepare_attention_masks_4d(att_2d_masks, dtype=dtype)
        sequence_length = att_2d_masks.shape[-1]
        if sequence_length <= sliding_window:
            sliding_mask = full_mask
        else:
            query_positions = torch.arange(sequence_length, device=att_2d_masks.device)[:, None]
            key_positions = torch.arange(sequence_length, device=att_2d_masks.device)[None, :]
            window_mask = (query_positions - key_positions) <= sliding_window
            sliding_mask = self._prepare_attention_masks_4d(att_2d_masks & window_mask, dtype=dtype)
        return {"full_attention": full_mask, "sliding_attention": sliding_mask}

    def _prepare_embeddings(self, prefix_embs: Tensor, suffix_embs: Tensor) -> tuple[Tensor, Tensor]:
        dtype = self._expert_weight_dtype()
        return prefix_embs.to(dtype=dtype), suffix_embs.to(dtype=dtype)

    def forward(self, observation, actions: Tensor) -> Tensor:
        images, image_masks, lang_tokens, lang_masks, _, physics_params = self._preprocess_observation(
            observation, train=self.training
        )
        physics_cond = self._encode_physics(physics_params)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_suffix(observation)
        prefix_embs, suffix_embs = self._prepare_embeddings(prefix_embs, suffix_embs)

        pad_masks = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
        att_masks = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        if self._is_gemma4:
            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            attention_mask = self._prepare_attention_masks_for_gemma4(
                att_2d_masks, sliding_window, dtype=prefix_embs.dtype
            )
        else:
            attention_mask = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, physics_cond],
        )
        predicted_actions = self.action_out_proj(suffix_out[:, -self.action_horizon :].to(torch.float32))
        return F.mse_loss(predicted_actions, actions.to(torch.float32), reduction="none").mean(dim=-1)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=1) -> Tensor:
        del device, noise, num_steps
        images, image_masks, lang_tokens, lang_masks, _, physics_params = self._preprocess_observation(
            observation, train=False
        )
        physics_cond = self._encode_physics(physics_params)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )
        prefix_embs, _ = self._prepare_embeddings(prefix_embs, prefix_embs)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        if self._is_gemma4:
            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            prefix_attention_mask = self._prepare_attention_masks_for_gemma4(
                prefix_att_2d_masks, sliding_window, dtype=prefix_embs.dtype
            )
        else:
            prefix_attention_mask = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = None
        if self._is_gemma4 and self._is_aligned_expert:
            self.paligemma_with_expert.gemma4_vlm.model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        elif not self._is_gemma4:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_suffix(observation)
        _, suffix_embs = self._prepare_embeddings(prefix_embs, suffix_embs)
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]

        if self._is_gemma4 and not self._is_aligned_expert:
            full_pad_masks = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
            full_att_masks = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
            full_att_2d_masks = make_att_2d_masks(full_pad_masks, full_att_masks)
            position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            attention_mask = self._prepare_attention_masks_for_gemma4(
                full_att_2d_masks, sliding_window, dtype=prefix_embs.dtype
            )
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, physics_cond],
            )
        else:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat((prefix_pad_2d_masks, suffix_att_2d_masks), dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            if self._is_gemma4:
                sliding_window = self.paligemma_with_expert.gemma4_expert.model.config.sliding_window
                attention_mask = self._prepare_attention_masks_for_gemma4(
                    full_att_2d_masks, sliding_window, dtype=suffix_embs.dtype
                )
                self.paligemma_with_expert.gemma4_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
            else:
                attention_mask = self._prepare_attention_masks_4d(full_att_2d_masks)
                self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, physics_cond],
            )

        suffix_out = outputs_embeds[1][:, -self.action_horizon :].to(torch.float32)
        return self.action_out_proj(suffix_out)


PhysicsAwareSingleStep = PhysicsAwareSingleStepPytorch

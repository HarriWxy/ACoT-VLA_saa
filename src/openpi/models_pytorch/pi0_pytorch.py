import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

# Gemma 4 support
_GEMMA4_VARIANTS = {"gemma4_300m", "gemma4_300m_lora", "gemma4_2b", "gemma4_2b_lora", "gemma4_e2b", "gemma4_e2b_lora"}


def _is_gemma4_variant(variant: str) -> bool:
    return variant in _GEMMA4_VARIANTS


def _has_lora(config) -> bool:
    """Check if any variant in the config uses LoRA."""
    paligemma_lora = "lora" in config.paligemma_variant
    expert_lora = "lora" in config.action_expert_variant
    return paligemma_lora or expert_lora


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self._is_gemma4 = _is_gemma4_variant(config.paligemma_variant)

        if self._is_gemma4:
            from openpi.models_pytorch.gemma4_pytorch import Gemma4WithExpertModel

            self.paligemma_with_expert = Gemma4WithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True] if self.pi05 else [False, False],
                precision=config.dtype,
                gemma4_model_path=getattr(config, "gemma4_model_path", None),
            )
        else:
            self.paligemma_with_expert = PaliGemmaWithExpertModel(
                paligemma_config,
                action_expert_config,
                use_adarms=[False, True] if self.pi05 else [False, False],
                precision=config.dtype,
            )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # 支持 OPENPI_DISABLE_COMPILE=1 环境变量禁用 torch.compile
        # (调试器 / torch.compile 不兼容时使用)
        import os
        _compile_mode = config.pytorch_compile_mode
        if os.environ.get("OPENPI_DISABLE_COMPILE", "0") == "1":
            _compile_mode = None
            logging.info("torch.compile disabled via OPENPI_DISABLE_COMPILE=1")
        if _compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Check if action expert width matches VLM width (enables KV cache reuse)
        self._is_aligned_expert = (
            self._is_gemma4
            and paligemma_config.width == action_expert_config.width
            and paligemma_config.head_dim == action_expert_config.head_dim
        )
        if self._is_aligned_expert:
            logging.info(
                f"Expert aligned with VLM (width={paligemma_config.width}, "
                f"head_dim={paligemma_config.head_dim}) — KV cache reuse enabled"
            )

        # transformers_replace check is only needed for Gemma 2 based models
        # (Gemma 4 is natively supported in transformers 5.x)
        if not self._is_gemma4:
            msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
            try:
                from transformers.models.siglip import check

                if not check.check_whether_transformers_replace_is_installed_correctly():
                    raise ValueError(msg)
            except ImportError:
                raise ValueError(msg) from None

        # --- LoRA injection for Gemma 4 ---
        self._lora_injected = False
        if self._is_gemma4 and _has_lora(config):
            self._inject_lora(paligemma_config, action_expert_config)

    def _inject_lora(self, vlm_config, expert_config):
        """Inject LoRA adapters into Gemma 4 model layers."""
        vlm_lora = vlm_config.lora_configs.get("attn") or vlm_config.lora_configs.get("ffn")
        expert_lora = expert_config.lora_configs.get("attn") or expert_config.lora_configs.get("ffn")

        # Only inject into Gemma4 text models (not vision tower)
        self.paligemma_with_expert.inject_lora(
            vlm_lora_config=vlm_lora if "lora" in self.config.paligemma_variant else None,
            expert_lora_config=expert_lora if "lora" in self.config.action_expert_variant else None,
        )
        self._lora_injected = True
        trainable, total = self.count_trainable_params()
        logging.info(f"LoRA injected: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M trainable ({100 * trainable / total:.2f}%)")

    def freeze_non_lora_params(self):
        """Freeze all parameters except LoRA adapters and action heads.

        Keeps trainable:
        - LoRA A/B parameters (lora_A, lora_B)
        - Action projection layers (action_in_proj, action_out_proj)
        - State/time MLP layers (state_proj, action_time_mlp_*, time_mlp_*)
        """
        from openpi.models_pytorch.lora_pytorch import freeze_non_lora_params as _freeze

        trainable_count = _freeze(self.paligemma_with_expert)

        # Always keep action heads trainable
        trainable_count = self._unfreeze_action_heads()

        logging.info(f"Froze non-LoRA params. Trainable: {trainable_count / 1e6:.2f}M")
        return trainable_count

    def freeze_vlm_only(self):
        """Freeze only the VLM (PaliGemma) and keep action expert fully trainable.

        Unlike freeze_non_lora_params() which freezes everything except LoRA adapters,
        this method freezes the entire VLM (including its LoRA adapters) while keeping
        the action expert's ALL parameters (base weights + LoRA) trainable.

        Use case: fine-tuning only the action expert while reusing a frozen VLM.

        Keeps trainable:
        - Action expert ALL parameters (gemma4_expert / action expert backbone)
        - Action projection layers (action_in_proj, action_out_proj)
        - State/time MLP layers (state_proj, action_time_mlp_*, time_mlp_*)
        """
        trainable_count = 0

        # Step 1: Freeze everything
        for param in self.parameters():
            param.requires_grad_(False)

        # Step 2: Unfreeze action expert (all params)
        if hasattr(self.paligemma_with_expert, "gemma4_expert"):
            for param in self.paligemma_with_expert.gemma4_expert.parameters():
                param.requires_grad_(True)
                trainable_count += param.numel()
        elif hasattr(self.paligemma_with_expert, "expert"):
            for param in self.paligemma_with_expert.expert.parameters():
                param.requires_grad_(True)
                trainable_count += param.numel()

        # Step 3: Unfreeze action heads
        trainable_count = self._unfreeze_action_heads(trainable_count)

        logging.info(f"Froze VLM only. Trainable: {trainable_count / 1e6:.2f}M")
        return trainable_count

    def _unfreeze_action_heads(self, initial_count: int = 0) -> int:
        """Unfreeze action projection and state/time MLP layers.

        Returns:
            Updated trainable parameter count.
        """
        trainable_count = initial_count
        action_head_params = [
            self.action_in_proj.parameters(),
            self.action_out_proj.parameters(),
        ]
        if self.pi05:
            action_head_params.extend([
                self.time_mlp_in.parameters(),
                self.time_mlp_out.parameters(),
            ])
        else:
            action_head_params.extend([
                self.state_proj.parameters(),
                self.action_time_mlp_in.parameters(),
                self.action_time_mlp_out.parameters(),
            ])

        for param_group in action_head_params:
            for p in param_group:
                p.requires_grad_(True)
                trainable_count += p.numel()

        return trainable_count

    def count_trainable_params(self) -> tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def get_lora_state_dict(self) -> dict:
        """Extract only LoRA parameters for saving."""
        return {k: v for k, v in self.state_dict().items() if "lora_A" in k or "lora_B" in k}

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        if self._is_gemma4:
            self.paligemma_with_expert.gemma4_vlm.model.gradient_checkpointing = True
            self.paligemma_with_expert.gemma4_vlm.model.gradient_checkpointing = True
            self.paligemma_with_expert.gemma4_expert.model.gradient_checkpointing = True
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        if self._is_gemma4:
            self.paligemma_with_expert.gemma4_vlm.model.gradient_checkpointing = False
            self.paligemma_with_expert.gemma4_expert.model.gradient_checkpointing = False
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
            self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _prepare_attention_masks_for_gemma4(self, att_2d_masks, sliding_window=512):
        """Prepare both full causal and sliding window 4D masks for Gemma 4."""
        full_mask = self._prepare_attention_masks_4d(att_2d_masks)
        # Sliding window: additionally mask positions outside the window
        seq_len = att_2d_masks.shape[-1]
        position_ids_q = torch.arange(seq_len, device=att_2d_masks.device)[:, None]
        position_ids_k = torch.arange(seq_len, device=att_2d_masks.device)[None, :]
        sliding_mask_2d = (position_ids_q - position_ids_k) <= sliding_window
        combined_2d = att_2d_masks & sliding_mask_2d
        sliding_mask = self._prepare_attention_masks_4d(combined_2d)
        return {"full_attention": full_mask, "sliding_attention": sliding_mask}

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)  # [B, action_horizon, action_dim] -> [B, action_horizon, width] action dim!=32

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if self._is_gemma4:
            first_layer = self.paligemma_with_expert.gemma4_vlm.model.layers[0]
        else:
            first_layer = self.paligemma_with_expert.paligemma.language_model.layers[0]
        # LoRA 注入后 q_proj 是 LoRALinear, 权重在 base_linear.weight
        q_proj = first_layer.self_attn.q_proj
        weight = q_proj.base_linear.weight if hasattr(q_proj, "base_linear") else q_proj.weight
        if weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1 # position_ids shape: [batch_size, seq_len] 

        # Prepare attention masks (Gemma 4 needs dual masks: full + sliding window)
        if self._is_gemma4:
            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            att_2d_masks_4d = self._prepare_attention_masks_for_gemma4(att_2d_masks, sliding_window)
        else:
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        if self._is_gemma4:
            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            prefix_att_2d_masks_4d = self._prepare_attention_masks_for_gemma4(prefix_att_2d_masks, sliding_window)
        else:
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        if self._is_gemma4 and self._is_aligned_expert:
            # Aligned Gemma4: VLM and expert share the same hidden_size/head_dim,
            # so we can reuse VLM's KV cache across denoise steps (same as Gemma2).
            self.paligemma_with_expert.gemma4_vlm.model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        elif self._is_gemma4:
            # Non-aligned Gemma4: VLM and expert have different hidden sizes,
            # can't share KV cache. Store prefix embeddings for joint attention.
            self._cached_prefix_embs = prefix_embs
            self._cached_prefix_pad_masks = prefix_pad_masks
            past_key_values = None
        else:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        if self._is_gemma4 and self._is_aligned_expert:
            # Aligned Gemma4: reuse VLM's cached KV (same pattern as Gemma2)
            first_layer = self.paligemma_with_expert.gemma4_vlm.model.layers[0]
            q_proj = first_layer.self_attn.q_proj
            weight = q_proj.base_linear.weight if hasattr(q_proj, "base_linear") else q_proj.weight
            if weight.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            # Build suffix-only mask: suffix attends to all prefix + causal suffix
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            # Position ids: offset by prefix length
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            full_att_2d_masks_4d = self._prepare_attention_masks_for_gemma4(full_att_2d_masks, sliding_window)
            self.paligemma_with_expert.gemma4_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        elif self._is_gemma4:
            # Non-aligned Gemma4: joint attention with prefix embeddings (no KV cache)
            prefix_embs = self._cached_prefix_embs

            first_layer = self.paligemma_with_expert.gemma4_vlm.model.layers[0]
            q_proj = first_layer.self_attn.q_proj
            weight = q_proj.base_linear.weight if hasattr(q_proj, "base_linear") else q_proj.weight
            if weight.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            # Build full prefix+suffix masks and position ids
            prefix_att_masks = torch.zeros(prefix_len, dtype=torch.bool, device=prefix_pad_masks.device)
            prefix_att_masks = prefix_att_masks.unsqueeze(0).expand(batch_size, -1)
            full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            full_att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            full_att_2d_masks = make_att_2d_masks(full_pad_masks, full_att_masks)
            position_ids = torch.cumsum(full_pad_masks, dim=1) - 1

            sliding_window = self.paligemma_with_expert.gemma4_vlm.model.config.sliding_window
            full_att_2d_masks_4d = self._prepare_attention_masks_for_gemma4(full_att_2d_masks, sliding_window)

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        else:
            # Old architecture: use expert-only forward with VLM's cached KV
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
            self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

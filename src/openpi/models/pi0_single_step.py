"""Single-step action prediction model based on pi0.5 architecture.

This module implements a variant of pi0.5 that replaces iterative flow matching
denoising with direct action prediction. Instead of 10 Euler steps from noise
to actions, it performs a single forward pass to predict clean actions directly.

Architecture differences from pi0.5:
- Uses the same PaliGemma backbone (visual encoder + LLM)
- Same prefix embedding (images + language tokens)
- Replaces diffusion suffix with direct action prediction tokens
- Trained with L2 regression loss on clean actions (no noise/time conditioning)

Inference: 1 forward pass vs 10 in standard pi0.5, ~10x faster.
"""

import logging
from typing import TYPE_CHECKING, override

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models.pi0 import make_attn_mask
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_config import Pi0SingleStepConfig

logger = logging.getLogger("openpi")


class Pi0SingleStep(_model.BaseModel):
    """Single-step action prediction model.

    Uses the same PaliGemma backbone as pi0.5 but predicts actions directly
    without iterative denoising. The model takes observation tokens as prefix
    and action tokens as suffix, then predicts clean actions in one pass.
    """

    def __init__(self, config: "Pi0SingleStepConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Same dual-expert architecture as pi0.5
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,  # Use adaRMSNorm like pi0.5
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Action projection (no time embedding needed for single-step)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Optional: learned action query tokens for better conditioning
        if config.use_action_queries:
            self.action_queries = nnx.Param(
                jax.random.normal(rngs.params(), (config.action_horizon, action_expert_config.width))
            )
            self.action_query_proj = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Encode observation (images + language) as prefix tokens."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        # Embed language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_action_suffix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Create action suffix tokens for direct prediction.

        Unlike pi0.5, this does NOT take noisy actions or timestep as input.
        Instead, it creates learnable action query tokens that attend to the
        observation prefix.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        if hasattr(self, "action_queries"):
            # Use learned action queries (like a decoder)
            batch_size = obs.state.shape[0]
            action_tokens = jnp.broadcast_to(self.action_queries.value, (batch_size, self.action_horizon, -1))
            action_tokens = self.action_query_proj(action_tokens)
        else:
            # Use zero-initialized action tokens (will be predicted directly)
            batch_size = obs.state.shape[0]
            action_tokens = jnp.zeros((batch_size, self.action_horizon, self.action_in_proj.out_features))

        tokens.append(action_tokens)
        input_mask.append(jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_))
        # Autoregressive mask: actions attend to each other
        ar_mask += [True] + [False] * (self.action_horizon - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute L2 regression loss on clean actions.

        Unlike pi0.5's flow matching loss (||v_t - u_t||^2),
        this uses simple regression: ||predicted_actions - actions||^2.
        """
        preprocess_rng, _ = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Encode prefix (images + language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Create action suffix (learnable queries)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_action_suffix(observation)

        # Combine for forward pass
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward through LLM
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None],  # No time conditioning needed
        )

        # Project to action space
        predicted_actions = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # L2 regression loss
        return jnp.mean(jnp.square(predicted_actions - actions), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,  # Always 1 for single-step
        noise: at.Float[at.Array, "b ah ad"] | None = None,  # Ignored, kept for API compatibility
    ) -> _model.Actions:
        """Single-step action prediction.

        Unlike pi0.5's 10-step Euler integration, this performs one forward pass.
        The `noise` parameter is ignored (kept for API compatibility).
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        # Encode prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Fill KV cache with prefix
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Create action suffix
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_action_suffix(observation)

        # Single forward pass with KV cache
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, None],
        )

        # Project to action space
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

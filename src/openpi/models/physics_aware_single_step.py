"""Physics-conditioned single-step action prediction model.

在 Pi0SingleStep 基础上增加 Physics Encoder, 将连续物理参数
 (重力加速度、摩擦系数、质量等) 编码为 token, 通过 adaRMS
 conditioning 通道注入 action expert 的每一层.

架构:
  PaliGemma (Vision + LLM) → prefix KV cache
  Physics Encoder (MLP + CrossAttn) → physics tokens
  Action Expert (adaRMS-conditioned) → 单步动作预测

推理: 1 次 forward pass, ~10x 快于 flow matching.
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
    from openpi.models.pi0_config import PhysicsAwareConfig

logger = logging.getLogger("openpi")


class PhysicsEncoder(nnx.Module):
    """将连续物理参数编码为 token 序列.

    输入: (B, physics_dim) — 如 [g, μ, mass, air_density, terrain_roughness]
    输出: (B, num_tokens, output_dim) — 可注入 action expert 的 token

    内部使用可学习 positional embedding + self-attention 让不同
    physics token 之间交互 (例如重力和摩擦的耦合效应)
    """

    def __init__(
        self,
        physics_dim: int,
        hidden_dim: int,
        num_tokens: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # 输入投影: 标量物理参数 → hidden space
        self.input_proj = nnx.Linear(physics_dim, hidden_dim, rngs=rngs)

        # 可学习的 query tokens (类似 DETR object queries)
        self.query_embed = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, hidden_dim))
        )

        # Self-attention layers
        self.attn_layers = [
            nnx.MultiHeadAttention(
                in_features=hidden_dim,
                num_heads=num_heads,
                rngs=rngs,
                param_dtype=jnp.float32,
            )
            for _ in range(num_layers)
        ]
        self.norms = [nnx.LayerNorm(hidden_dim, rngs=rngs) for _ in range(num_layers)]

        # 输出投影
        self.output_proj = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, physics_params: jax.Array) -> jax.Array:
        """
        Args:
            physics_params: (B, physics_dim) 连续物理参数
        Returns:
            physics_tokens: (B, num_tokens, output_dim)
        """
        batch_size = physics_params.shape[0]

        # 投射到 hidden space
        x = nnx.swish(self.input_proj(physics_params))  # (B, hidden_dim)
        x = x[:, None, :]  # (B, 1, hidden_dim)

        # 拼接可学习 query tokens
        queries = jnp.broadcast_to(
            self.query_embed.value[None, :, :], (batch_size, self.num_tokens, self.hidden_dim)
        )
        x = jnp.concatenate([x, queries], axis=1)  # (B, 1+num_tokens, hidden_dim)

        # Self-attention
        for attn, norm in zip(self.attn_layers, self.norms, strict=True):
            normalized = norm(x)
            x = x + attn(normalized, normalized, normalized, decode=False)

        # 只取 query tokens 的输出 (丢弃第一个 input token)
        x = x[:, 1:, :]  # (B, num_tokens, hidden_dim)

        return self.output_proj(x)  # (B, num_tokens, output_dim)


class PhysicsAwareSingleStep(_model.BaseModel):
    """物理信息引导的单步动作预测模型.

    与 Pi0SingleStep 的区别:
    1. 新增 PhysicsEncoder: 将连续物理参数编码为 token
    2. physics_cond 通过 adaRMS conditioning 通道注入 action expert
    3. compute_loss / sample_actions 接收 physics_params 参数
    """

    def __init__(self, config: "PhysicsAwareConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # === PaliGemma backbone (与 Pi0SingleStep 完全相同) ===
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
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

        # === Physics Encoder (新增) ===
        self.physics_encoder = PhysicsEncoder(
            physics_dim=config.physics_dim,
            hidden_dim=config.physics_hidden_dim,
            num_tokens=config.num_physics_tokens,
            output_dim=action_expert_config.width,
            rngs=rngs,
        )

        # Physics conditioning 聚合投影 (→ adaRMS cond vector)
        self.physics_cond_proj = nnx.Sequential(
            nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs),
            nnx.swish,
            nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs),
        )

        # === Action projection ===
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # === Learned action query tokens ===
        if config.use_action_queries:
            self.action_queries = nnx.Param(
                jax.random.normal(rngs.params(), (config.action_horizon, action_expert_config.width))
            )
            self.action_query_proj = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )

        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """编码视觉 + 语言 prefix (与 Pi0SingleStep 完全相同)."""
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

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
        self, obs: _model.Observation, physics_cond: at.Float[at.Array, "b d"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """创建 action suffix tokens. physics_cond 作为 adaRMS conditioning 返回."""
        input_mask = []
        ar_mask = []
        tokens = []

        if hasattr(self, "action_queries"):
            batch_size = obs.state.shape[0]
            action_tokens = jnp.broadcast_to(
                self.action_queries.value,
                (batch_size, self.action_horizon, self.action_queries.value.shape[-1]),
            )
            action_tokens = self.action_query_proj(action_tokens)
        else:
            batch_size = obs.state.shape[0]
            action_tokens = jnp.zeros(
                (batch_size, self.action_horizon, self.action_in_proj.out_features)
            )

        tokens.append(action_tokens)
        input_mask.append(jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + [False] * (self.action_horizon - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _encode_physics(self, physics_params: jax.Array) -> at.Float[at.Array, "b d"]:
        """物理参数 → adaRMS conditioning vector."""
        physics_tokens = self.physics_encoder(physics_params)  # (B, n_p, d)
        # Mean-pool across tokens → single conditioning vector
        physics_cond = physics_tokens.mean(axis=1)  # (B, d)
        physics_cond = self.physics_cond_proj(physics_cond)
        return nnx.swish(physics_cond)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """单步 L2 回归 loss, physics-conditioned."""
        preprocess_rng, _ = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 1. Physics encoding → conditioning vector
        physics_cond = self._encode_physics(observation.physics_params)

        # 2. Prefix (vision + language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # 3. Action suffix (learnable queries)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_action_suffix(
            observation, physics_cond
        )

        # 4. Build attention mask
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 5. Forward through LLM (physics_cond via adaRMS)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, physics_cond],  # ← 关键: physics conditioning
        )

        # 6. Predict actions & L2 loss
        predicted_actions = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(predicted_actions - actions), axis=-1)

    @override
    def compute_loss_rl(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """单步 L2 回归 loss, physics-conditioned."""
        preprocess_rng, _ = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 1. Physics encoding → conditioning vector
        physics_cond = self._encode_physics(observation.physics_params)

        # 2. Prefix (vision + language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # 3. Action suffix (learnable queries)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_action_suffix(
            observation, physics_cond
        )

        # 4. Build attention mask
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 5. Forward through LLM (physics_cond via adaRMS)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, physics_cond],  # ← 关键: physics conditioning
        )

        # 6. Predict actions & L2 loss
        predicted_actions = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(predicted_actions - actions), axis=-1)


    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """单步推理: 1 次 forward pass."""
        observation = _model.preprocess_observation(None, observation, train=False)

        # 1. Physics conditioning
        physics_cond = self._encode_physics(observation.physics_params)

        # 2. Prefix encoding + KV cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # 3. Action suffix
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_action_suffix(
            observation, physics_cond
        )

        # 4. Single forward pass with KV cache
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_local = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1) - 1
        )

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, physics_cond],
        )

        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

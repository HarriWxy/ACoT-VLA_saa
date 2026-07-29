import inspect
import types

import torch

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.gemma_pytorch import modeling_gemma


class _FakeOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = None


class _FakeLinear:
    def __init__(self):
        self.weight = torch.empty(1, dtype=torch.float32)

    def __call__(self, x):
        return x


class _FakeMLP:
    def __init__(self):
        self.up_proj = types.SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))

    def __call__(self, x):
        return x


class _FakeLayer:
    def __init__(self):
        self.input_calls = []
        self.post_calls = []
        self.self_attn = types.SimpleNamespace(
            q_proj=_FakeLinear(),
            k_proj=_FakeLinear(),
            v_proj=_FakeLinear(),
            o_proj=_FakeLinear(),
            head_dim=1,
            scaling=1.0,
        )
        self.mlp = _FakeMLP()

    def input_layernorm(self, x, cond=None):
        self.input_calls.append(cond)
        return x, None

    def post_attention_layernorm(self, x, cond=None):
        self.post_calls.append(cond)
        return x, None


class _FakeModel:
    def __init__(self, num_layers=1):
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.forward_calls = []
        self.norm_calls = []
        self.rotary_calls = []
        self.gradient_checkpointing = False

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return _FakeOutput(kwargs["inputs_embeds"])

    def norm(self, x, cond=None):
        self.norm_calls.append(cond)
        return x, None

    def rotary_emb(self, hidden_states, position_ids):
        self.rotary_calls.append((hidden_states, position_ids))
        return hidden_states, hidden_states


def _build_model():
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    model.training = False
    model.gradient_checkpointing = False
    model.paligemma = types.SimpleNamespace(
        language_model=_FakeModel(num_layers=1),
        model=types.SimpleNamespace(language_model=types.SimpleNamespace(rotary_emb=lambda x, pos, layer_type=None: (x, x))),
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=1)),
    )
    model.gemma_expert = types.SimpleNamespace(model=_FakeModel(num_layers=1))
    return model


def test_prefix_branch_forwards_attention_mask_position_ids_and_adarms_cond(monkeypatch):
    model = _build_model()
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.apply_rotary_pos_emb",
        lambda q, k, cos, sin, unsqueeze_dim=1: (q, k),
    )
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.eager_attention_forward",
        lambda *args, **kwargs: (torch.zeros(1, 2, 8), None),
    )

    attention_mask = torch.ones(1, 2, dtype=torch.bool)
    position_ids = torch.arange(2).unsqueeze(0)
    adarms_cond = torch.randn(1, 3)

    model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[torch.randn(1, 2, 8), None],
        use_cache=False,
        adarms_cond=[adarms_cond, None],
    )

    forward_kwargs = model.paligemma.language_model.forward_calls[0]
    assert forward_kwargs["attention_mask"] is attention_mask
    assert forward_kwargs["position_ids"] is position_ids
    assert forward_kwargs["adarms_cond"] is adarms_cond


def test_joint_branch_passes_conditioning_to_layernorm_and_final_norm(monkeypatch):
    model = _build_model()
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.apply_rotary_pos_emb",
        lambda q, k, cos, sin, unsqueeze_dim=1: (q, k),
    )
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.eager_attention_forward",
        lambda *args, **kwargs: (torch.zeros(1, 4, 8), None),
    )

    adarms_cond = torch.randn(1, 3)
    model.forward(
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=[torch.randn(1, 2, 8), torch.randn(1, 2, 8)],
        use_cache=False,
        adarms_cond=[adarms_cond, adarms_cond],
    )

    assert model.paligemma.language_model.layers[0].input_calls == [adarms_cond]
    assert model.paligemma.language_model.layers[0].post_calls == [adarms_cond]
    assert model.gemma_expert.model.layers[0].input_calls == [adarms_cond]
    assert model.gemma_expert.model.layers[0].post_calls == [adarms_cond]
    assert model.paligemma.language_model.norm_calls == [adarms_cond]
    assert model.gemma_expert.model.norm_calls == [adarms_cond]


def test_joint_branch_uses_real_hidden_states_for_rotary_embeddings(monkeypatch):
    model = _build_model()
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.apply_rotary_pos_emb",
        lambda q, k, cos, sin, unsqueeze_dim=1: (q, k),
    )
    monkeypatch.setattr(
        "openpi.models_pytorch.gemma_pytorch.modeling_gemma.eager_attention_forward",
        lambda *args, **kwargs: (torch.zeros(1, 4, 8), None),
    )

    hidden_states_a = torch.randn(1, 2, 8)
    hidden_states_b = torch.randn(1, 2, 8)
    model.forward(
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=[hidden_states_a, hidden_states_b],
        use_cache=False,
        adarms_cond=[None, None],
    )

    assert len(model.paligemma.language_model.rotary_calls) == 1
    assert len(model.gemma_expert.model.rotary_calls) == 1
    assert torch.equal(model.paligemma.language_model.rotary_calls[0][0], hidden_states_a)
    assert torch.equal(model.gemma_expert.model.rotary_calls[0][0], hidden_states_b)


def test_gemma_causal_lm_forward_accepts_adarms_cond():
    assert "adarms_cond" in inspect.signature(modeling_gemma.GemmaForCausalLM.forward).parameters

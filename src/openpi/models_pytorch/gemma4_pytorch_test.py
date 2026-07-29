import types

import torch

from openpi.models_pytorch.gemma4_pytorch import Gemma4WithExpertModel


class _FakeOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeOutput(torch.zeros(1, 1, 1))


def test_forward_passes_adarms_cond_to_expert_only_branch():
    model = Gemma4WithExpertModel.__new__(Gemma4WithExpertModel)
    vlm_model = _FakeModel()
    expert_model = _FakeModel()
    model.gemma4_vlm = types.SimpleNamespace(model=vlm_model)
    model.gemma4_expert = types.SimpleNamespace(model=expert_model)

    cond = torch.zeros(1, 2)
    model.forward(
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=[None, torch.zeros(1, 1, 2)],
        use_cache=False,
        adarms_cond=[None, cond],
    )

    assert expert_model.calls[0]["adarms_cond"] is cond

import numpy as np

from openpi.models import model as _model
from openpi.policies import srb_policy


def test_srb_inputs_flatten_and_pad_state():
    transform = srb_policy.SRBInputs(action_dim=32, model_type=_model.ModelType.PI05)

    result = transform(
        {
            "proprio": np.ones(9, dtype=np.float32),
            "state": np.arange(18, dtype=np.float32),
            "image_cam_base": np.zeros((224, 224, 3), dtype=np.uint8),
            "image_cam_wrist": np.ones((224, 224, 3), dtype=np.uint8),
            "prompt": "pick up the sample",
        }
    )

    assert result["state"].shape == (32,)
    assert np.allclose(result["state"][:9], 1.0)
    assert result["state"][9:27].tolist() == list(range(18))
    assert result["state"][27:].tolist() == [0.0] * 5
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_srb_inputs_handles_missing_images():
    transform = srb_policy.SRBInputs(action_dim=32, model_type=_model.ModelType.PI0)

    result = transform({"proprio": np.ones(9, dtype=np.float32), "prompt": "do something"})

    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert not result["image_mask"]["base_0_rgb"]
    assert not result["image_mask"]["left_wrist_0_rgb"]


def test_srb_outputs_slice_actions():
    transform = srb_policy.SRBOutputs(action_dim=7)
    actions = np.random.rand(10, 32).astype(np.float32)

    result = transform({"actions": actions})

    assert result["actions"].shape == (10, 7)
    assert np.allclose(result["actions"], actions[:, :7])
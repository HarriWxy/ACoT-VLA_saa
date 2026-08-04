import numpy as np

from openpi.policies import debug_policy


def test_debug_policy_random_shape_and_range():
    policy = debug_policy.DebugChunkPolicy(
        debug_policy.DebugPolicyConfig(
            action_horizon=4,
            action_dim=7,
            mode="random",
            seed=0,
            action_scale=0.5,
        )
    )

    outputs = policy.infer({"ignored": True})

    assert outputs["actions"].shape == (4, 7)
    assert outputs["actions"].dtype == np.float32
    assert np.all(outputs["actions"] <= 0.5)
    assert np.all(outputs["actions"] >= -0.5)


def test_debug_policy_zero_actions():
    policy = debug_policy.DebugChunkPolicy(
        debug_policy.DebugPolicyConfig(action_horizon=2, action_dim=3, mode="zero")
    )

    outputs = policy.infer({})

    np.testing.assert_array_equal(outputs["actions"], np.zeros((2, 3), dtype=np.float32))
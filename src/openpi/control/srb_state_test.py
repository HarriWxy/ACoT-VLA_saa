import numpy as np

from openpi.control.srb_state import AffineStats
from openpi.control.srb_state import ObservationSchema


def test_observation_schema_flattens_nested_fields_deterministically():
    observation = {
        "state": np.array([1.0, 2.0], dtype=np.float32),
        "proprio_dyn": {
            "joint_vel": np.array([3.0, 4.0], dtype=np.float32),
            "joint_pos": np.array([5.0], dtype=np.float32),
        },
    }
    schema = ObservationSchema.from_observation(observation, ("state", "proprio_dyn"))

    assert schema.paths == ("state", "proprio_dyn.joint_pos", "proprio_dyn.joint_vel")
    np.testing.assert_allclose(schema.pack(observation), [1.0, 2.0, 5.0, 3.0, 4.0])


def test_affine_stats_preserve_physics_distinctions():
    values = np.array([[1.6, 0.4], [3.7, 0.6], [9.8, 0.5]], dtype=np.float32)
    stats = AffineStats.from_array(values)
    normalized = stats.normalize(values)

    np.testing.assert_allclose(stats.denormalize(normalized), values, atol=1e-6)
    assert not np.allclose(normalized[0], normalized[1])

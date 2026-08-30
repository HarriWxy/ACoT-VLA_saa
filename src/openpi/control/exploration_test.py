import numpy as np

from openpi.control.exploration import SmoothExplorationConfig
from openpi.control.exploration import SmoothRandomExplorer


def test_smooth_explorer_respects_bounds_and_rate_limits():
    low = np.array([-1.0, -0.5], dtype=np.float32)
    high = np.array([1.0, 0.5], dtype=np.float32)
    explorer = SmoothRandomExplorer(
        low,
        high,
        SmoothExplorationConfig(
            num_sinusoids=2,
            action_rate_fraction=0.1,
            dt=0.04,
        ),
        seed=3,
    )
    sequence = explorer.sample_sequence(64)

    assert sequence.shape == (64, 2)
    assert np.all(sequence >= low)
    assert np.all(sequence <= high)
    assert np.all(np.abs(np.diff(sequence, axis=0)) <= np.array([0.1, 0.05]) + 1e-6)

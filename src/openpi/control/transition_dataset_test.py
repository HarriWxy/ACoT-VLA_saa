import numpy as np

from openpi.control.transition_dataset import TransitionDataset


def test_transition_dataset_round_trip_and_episode_split(tmp_path):
    dataset = TransitionDataset(
        state=np.arange(24, dtype=np.float32).reshape(6, 4),
        action=np.zeros((6, 2), dtype=np.float32),
        next_state=np.ones((6, 4), dtype=np.float32),
        command=np.zeros((6, 1), dtype=np.float32),
        physics=np.tile(np.array([[1.6, 0.5]], dtype=np.float32), (6, 1)),
        reward=np.arange(6, dtype=np.float32),
        terminated=np.array([False, False, True, False, False, True]),
        truncated=np.zeros(6, dtype=bool),
        episode_id=np.array([0, 0, 0, 1, 1, 1]),
        metadata={"name": "test"},
    )
    path = tmp_path / "transitions.npz"
    dataset.save(path)
    restored = TransitionDataset.load(path)

    np.testing.assert_allclose(restored.state, dataset.state)
    np.testing.assert_array_equal(restored.episode_id, dataset.episode_id)
    assert restored.metadata == {"name": "test"}
    train, validation = restored.split_by_episode(validation_fraction=0.5, seed=0)
    assert set(restored.episode_id[train]).isdisjoint(set(restored.episode_id[validation]))

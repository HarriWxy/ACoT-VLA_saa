import jax.numpy as jnp
import numpy as np

from RLtune.episode_dataset import EpisodeAwareDataset
from RLtune.grpo_algo import compute_advantage_weights
from RLtune.grpo_algo import compute_advantages
from RLtune.train_offline_jax import get_resume_start_epoch
from RLtune.train_offline_jax import prepare_advantages


def test_prepare_advantages_repeats_each_episode_for_its_sampled_frames() -> None:
    advantages = jnp.array([0.2, -0.5, 1.0], dtype=jnp.float32)

    prepared = prepare_advantages(advantages, frame_counts=[2, 2, 1])

    assert prepared.shape == (5,)
    assert prepared[0] == 0.2
    assert prepared[1] == 0.2
    assert prepared[2] == -0.5
    assert prepared[3] == -0.5
    assert prepared[4] == 1.0


def test_rloo_advantages_are_written_to_the_original_batch() -> None:
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    prompt_indices = np.array([0, 0], dtype=np.int16)

    advantages = compute_advantages(rewards, prompt_indices, estimator="rloo", n_samples=2)

    np.testing.assert_allclose(advantages, np.array([-1.0, 1.0], dtype=np.float32))


def test_grpo_episode_groups_do_not_mix_tasks() -> None:
    class _HfDataset:
        def __init__(self):
            self.column_names = ("episode_index", "reward", "task_index")
            self.columns = {
                "episode_index": [0, 0, 1, 1],
                "reward": [0.0, 1.0, 0.0, 1.0],
                "task_index": [0, 0, 1, 1],
            }

        def __getitem__(self, index):
            if isinstance(index, str):
                return self.columns[index]
            return {key: values[index] for key, values in self.columns.items()}

    raw_dataset = type(
        "RawDataset",
        (),
        {"hf_dataset": _HfDataset(), "meta": type("Meta", (), {"tasks": ["task-0", "task-1"]})()},
    )()
    transformed_dataset = type(
        "TransformedDataset",
        (),
        {"_dataset": raw_dataset, "__len__": lambda self: 4},
    )()
    dataset = EpisodeAwareDataset(transformed_dataset)

    groups = dataset.sample_episodes_for_grpo(
        n_groups=8,
        n_samples_per_group=2,
        rng=np.random.default_rng(0),
    )

    for group in groups:
        assert len({dataset.get_episode_prompt(episode_index) for episode_index in group}) == 1


def test_advantage_weights_are_positive_normalized_and_ordered() -> None:
    weights = compute_advantage_weights(jnp.array([-100.0, 0.0, 100.0]), max_weight=10.0)

    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights > 0)
    np.testing.assert_allclose(float(jnp.mean(weights)), 1.0)
    assert weights[0] < weights[1] < weights[2]


def test_resume_starts_after_the_latest_completed_checkpoint_epoch() -> None:
    class _CheckpointManager:
        def all_steps(self):
            return (5, 10, 20)

    assert get_resume_start_epoch(_CheckpointManager(), resuming=False) == 0
    assert get_resume_start_epoch(_CheckpointManager(), resuming=True) == 20

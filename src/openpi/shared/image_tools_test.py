import pathlib
import sys

import jax
import jax.numpy as jnp
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from openpi.models import model as _model
from openpi.shared import image_tools
from RLtune import episode_dataset


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)


def test_preprocess_observation_accepts_channels_first_torch_images():
    obs_dict = {
        "image": {
            "base_0_rgb": torch.zeros((2, 3, 32, 32), dtype=torch.uint8),
        },
        "image_mask": {"base_0_rgb": torch.ones((2,), dtype=torch.bool)},
        "state": torch.zeros((2, 4), dtype=torch.float32),
    }

    observation = _model.Observation.from_dict(obs_dict)
    assert observation.images["base_0_rgb"].shape == (2, 32, 32, 3)

    processed = _model.preprocess_observation(
        jax.random.key(0),
        observation,
        train=False,
        image_keys=("base_0_rgb",),
        image_resolution=(224, 224),
    )
    assert processed.images["base_0_rgb"].shape == (2, 224, 224, 3)


def test_episode_dataset_collates_channels_first_images_as_channels_last():
    class _FakeDataset:
        def __init__(self):
            self._dataset = type("Raw", (), {"hf_dataset": {"episode_index": [0, 0, 1, 1], "reward": [0.1, 0.2, 0.3, 0.4]}})()

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            image = torch.zeros((3, 8, 8), dtype=torch.uint8)
            return {
                "image": {"base_0_rgb": image},
                "image_mask": {"base_0_rgb": torch.ones(1, dtype=torch.bool)},
                "state": torch.zeros((4,), dtype=torch.float32),
                "actions": torch.zeros((16, 19), dtype=torch.float32),
            }

    dataset = episode_dataset.EpisodeAwareDataset(_FakeDataset())
    observation, actions, rewards = dataset.sample_batch_from_episodes([0, 1], frames_per_episode=2)

    assert observation.images["base_0_rgb"].shape == (4, 8, 8, 3)
    assert actions.shape[0] == 4
    assert rewards.shape == (4,)

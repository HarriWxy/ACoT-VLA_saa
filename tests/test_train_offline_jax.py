import jax.numpy as jnp

from RLtune.train_offline_jax import prepare_advantages


def test_prepare_advantages_repeats_to_batch_size() -> None:
    advantages = jnp.array([0.2, -0.5, 1.0], dtype=jnp.float32)

    prepared = prepare_advantages(advantages, batch_size=5)

    assert prepared.shape == (5,)
    assert prepared[0] == 0.2
    assert prepared[1] == -0.5
    assert prepared[2] == 1.0
    assert prepared[3] == 0.2
    assert prepared[4] == -0.5

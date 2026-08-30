import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.srb_world_model import SRBWorldModel
from openpi.models.srb_world_model import SRBWorldModelConfig
from openpi.models.srb_world_model import stack_ensemble


def test_srb_world_model_and_ensemble_shapes():
    config = SRBWorldModelConfig(
        state_dim=4,
        action_dim=2,
        command_dim=3,
        physics_dim=5,
        hidden_dim=16,
        num_layers=2,
    )
    model = SRBWorldModel(config)
    state = jnp.zeros((8, config.state_dim))
    action = jnp.zeros((8, config.action_dim))
    command = jnp.zeros((8, config.command_dim))
    physics = jnp.zeros((8, config.physics_dim))
    keys = jax.random.split(jax.random.key(0), 3)
    parameters = stack_ensemble([model.init(key, state, action, command, physics)["params"] for key in keys])

    delta, reward, fall = model.apply(
        {"params": jax.tree.map(lambda value: value[0], parameters)},
        state,
        action,
        command,
        physics,
    )
    assert delta.shape == (8, 4)
    assert reward.shape == (8, 1)
    assert fall.shape == (8, 1)
    assert np.all(np.isfinite(np.asarray(delta)))

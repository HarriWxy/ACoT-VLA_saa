import jax
import jax.numpy as jnp
import numpy as np

from openpi.control.cem_planner import CEMConfig
from openpi.control.cem_planner import CEMPlanner
from openpi.control.cem_planner import colored_noise
from openpi.control.srb_state import AffineStats
from openpi.models.srb_world_model import SRBWorldModel
from openpi.models.srb_world_model import SRBWorldModelConfig
from openpi.models.srb_world_model import stack_ensemble


def _make_planner() -> CEMPlanner:
    model_config = SRBWorldModelConfig(
        state_dim=3,
        action_dim=2,
        command_dim=1,
        physics_dim=2,
        hidden_dim=16,
        num_layers=1,
    )
    model = SRBWorldModel(model_config)
    sample = (
        jnp.zeros((1, 3)),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 2)),
    )
    keys = jax.random.split(jax.random.key(1), 3)
    parameters = stack_ensemble([model.init(key, *sample)["params"] for key in keys])
    stats = {
        "state": AffineStats(np.zeros(3), np.ones(3)),
        "action": AffineStats(np.zeros(2), np.ones(2)),
        "command": AffineStats(np.zeros(1), np.ones(1)),
        "physics": AffineStats(np.zeros(2), np.ones(2)),
    }
    return CEMPlanner(
        model=model,
        parameters=parameters,
        action_low=np.array([-1.0, -0.5], dtype=np.float32),
        action_high=np.array([1.0, 0.5], dtype=np.float32),
        stats=stats,
        config=CEMConfig(
            horizon=4,
            num_samples=16,
            num_iterations=2,
            elite_size=4,
            seed=2,
        ),
    )


def test_colored_noise_has_expected_shape():
    noise = colored_noise(jax.random.key(0), (7, 5, 3), rho=0.8)
    assert noise.shape == (7, 5, 3)
    assert np.all(np.isfinite(np.asarray(noise)))


def test_cem_plan_is_bounded_and_has_receding_horizon_shape():
    planner = _make_planner()
    result = planner.plan(
        state=np.zeros(3, dtype=np.float32),
        command=np.zeros(1, dtype=np.float32),
        physics=np.zeros(2, dtype=np.float32),
        previous_action=np.zeros(2, dtype=np.float32),
    )

    assert result.actions.shape == (4, 2)
    assert np.all(result.actions >= np.array([-1.0, -0.5]) - 1e-6)
    assert np.all(result.actions <= np.array([1.0, 0.5]) + 1e-6)

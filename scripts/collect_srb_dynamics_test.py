from types import SimpleNamespace

from scripts.collect_srb_dynamics import Args
from scripts.collect_srb_dynamics import _apply_physics_overrides


def test_apply_physics_overrides_updates_native_and_vla_config_fields():
    actuator = SimpleNamespace(effort_limit_sim=10.0, effort_limit=None)
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics_material=SimpleNamespace(static_friction=1.0, dynamic_friction=1.0)),
        agent_rate=0.04,
        action_delay_steps=0,
        _robot=SimpleNamespace(asset_cfg=SimpleNamespace(actuators={"legs": actuator}), payload=None),
    )
    args = Args(
        friction=0.7,
        payload_mass_scale=1.2,
        motor_strength_scale=0.5,
        action_delay=0.08,
    )

    application = _apply_physics_overrides(env_cfg, gravity=3.72, args=args)

    assert env_cfg.sim.physics_material.static_friction == 0.7
    assert env_cfg.sim.physics_material.dynamic_friction == 0.7
    assert env_cfg.action_delay_steps == 2
    assert actuator.effort_limit_sim == 5.0
    assert env_cfg.vla_physics["gravity"] == 3.72
    assert env_cfg.vla_physics["payload_mass_scale"] == 1.2
    assert application["applied"] == {
        "gravity": True,
        "friction": True,
        "action_delay": True,
        "payload_mass_scale": False,
        "motor_strength_scale": True,
    }


def test_apply_physics_overrides_marks_non_native_delay_for_the_vla_task():
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics_material=SimpleNamespace(static_friction=1.0, dynamic_friction=1.0)),
        agent_rate=0.04,
        _robot=SimpleNamespace(asset_cfg=SimpleNamespace(actuators={}), payload=None),
    )

    application = _apply_physics_overrides(env_cfg, gravity=1.62, args=Args(action_delay=0.08))

    assert not hasattr(env_cfg, "action_delay_steps")
    assert env_cfg.vla_physics["action_delay_steps"] == 2
    assert not application["applied"]["action_delay"]

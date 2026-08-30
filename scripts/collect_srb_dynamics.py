"""Collect transition data for the low-dimensional SRB world model.

Unlike the VLA collector, this script records a complete transition after the
environment step and stores the physics/domain vector explicitly.  SRB is
imported only after AppLauncher starts, so the schema and explorer remain
usable in CPU-only unit tests.

Example (run in the SRB/Isaac Sim Python environment)::

    python scripts/collect_srb_dynamics.py \
        --domains moon,mars,earth \
        --episodes-per-domain 100 \
        --state-keys state,proprio,proprio_dyn,state_dyn \
        --output data/srb_dynamics/transitions.npz
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from openpi.control.exploration import SmoothExplorationConfig
from openpi.control.exploration import SmoothRandomExplorer
from openpi.control.srb_state import ObservationSchema
from openpi.control.srb_state import parse_paths
from openpi.control.transition_dataset import TransitionDataset

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    env_id: str = "srb/locomotion_velocity_tracking"
    domains: str = "moon,mars,earth"
    output: Path = Path("data/srb_dynamics/transitions.npz")

    episodes_per_domain: int = 100
    max_steps: int = 250
    action_repeat: int = 5
    sim_dt: float = 1.0 / 125.0

    # These are explicit paths into the SRB observation dictionary.  Include
    # proprio_dyn/state_dyn when the task exposes them to make the model closer
    # to Markov; the default keeps compatibility with the existing collector.
    state_keys: str = "state,proprio"
    command_keys: str = "command"

    friction: float = 0.5
    payload_mass_scale: float = 1.0
    motor_strength_scale: float = 1.0
    # Seconds. Converted to the SRB action_delay_steps setting before env creation.
    action_delay: float = 0.0

    seed: int = 0
    device: str = "cuda:0"
    headless: bool = True
    enable_cameras: bool = False


_DOMAIN_GRAVITY = {
    "asteroid": 0.14219,
    "earth": 9.80665,
    "mars": 3.72076,
    "moon": 1.62496,
    "orbit": 0.0,
}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def _to_scalar(value: Any) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _to_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _info_reports_fall(info: Any) -> bool:
    """Read common fall flags without depending on one SRB wrapper version."""

    if not isinstance(info, dict):
        return False
    for key in ("fall", "fell", "is_fallen", "fallen", "failure"):
        if key in info:
            return _to_bool(info[key])
    return False


def _get_action_space(env: Any) -> Any:
    if hasattr(env.unwrapped, "single_action_space"):
        return env.unwrapped.single_action_space
    return env.action_space


def _set_domain(env_cfg: Any, domain_name: str) -> tuple[Any, float]:
    domain_name = domain_name.strip().lower()
    if domain_name not in _DOMAIN_GRAVITY:
        raise ValueError(f"Unknown domain '{domain_name}'. Choose from {sorted(_DOMAIN_GRAVITY)}.")

    # SRB's Domain enum is the source of truth for gravity and terrain setup.
    # Keep this import here so the module can be imported without Isaac Sim.
    from srb.core.domain import Domain  # noqa: PLC0415

    domain = Domain.from_str(domain_name)
    if domain is None:
        raise ValueError(f"SRB does not define domain '{domain_name}'.")
    if not hasattr(env_cfg, "domain") or not hasattr(env_cfg, "sim"):
        raise AttributeError("The selected SRB task config has no domain/simulation fields.")

    # Domain-dependent assets and gravity are configured in __post_init__.  A
    # simple assignment after load_cfg_from_registry leaves sim.gravity at the
    # original default domain, so rebuild the config with the selected domain.
    env_cfg = dataclasses.replace(env_cfg, domain=domain, gravity=domain)
    return env_cfg, float(domain.gravity_magnitude)


def _scale_numeric(value: Any, scale: float) -> Any:
    if isinstance(value, dict):
        return {key: _scale_numeric(item, scale) for key, item in value.items()}
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value) * scale
    return value


def _apply_motor_strength_scale(env_cfg: Any, scale: float) -> bool:
    """Scale explicit actuator effort limits when the selected robot exposes them."""

    robot = getattr(env_cfg, "_robot", None)
    asset_cfg = getattr(robot, "asset_cfg", None)
    actuators = getattr(asset_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return False

    applied = False
    for actuator in actuators.values():
        for field in ("effort_limit_sim", "effort_limit"):
            value = getattr(actuator, field, None)
            if value is None:
                continue
            setattr(actuator, field, _scale_numeric(value, scale))
            applied = True
    return applied


def _apply_payload_mass_scale(env_cfg: Any, scale: float) -> bool:
    """Scale a configured payload mass when the task exposes one pre-spawn."""

    robot = getattr(env_cfg, "_robot", None)
    payload = getattr(robot, "payload", None)
    asset_cfg = getattr(payload, "asset_cfg", None)
    spawn = getattr(asset_cfg, "spawn", None)
    mass_props = getattr(spawn, "mass_props", None)
    mass = getattr(mass_props, "mass", None)
    if mass is None:
        return False
    mass_props.mass = _scale_numeric(mass, scale)
    return True


def _apply_physics_overrides(env_cfg: Any, gravity: float, args: Args) -> dict[str, Any]:
    """Apply SRB-native fields and expose custom values for the VLA task config.

    The supplied SRB fork natively supports the material and action-delay fields.
    ``vla_physics`` is intentionally attached before ``gymnasium.make``. A
    custom VLA task can consume it while creating its runtime scene, including
    payload mass or an actuator model it owns.
    """

    if not hasattr(env_cfg.sim, "physics_material"):
        raise AttributeError("The selected SRB task config has no sim.physics_material field.")
    material = env_cfg.sim.physics_material
    material.static_friction = args.friction
    material.dynamic_friction = args.friction

    agent_dt = float(getattr(env_cfg, "agent_rate", args.sim_dt))
    if agent_dt <= 0:
        raise ValueError(f"Environment agent_rate must be positive, got {agent_dt}.")
    action_delay_steps = round(args.action_delay / agent_dt)
    action_delay_applied = hasattr(env_cfg, "action_delay_steps")
    if action_delay_applied:
        env_cfg.action_delay_steps = action_delay_steps

    payload_applied = _apply_payload_mass_scale(env_cfg, args.payload_mass_scale)
    motor_applied = _apply_motor_strength_scale(env_cfg, args.motor_strength_scale)
    vla_physics = {
        "gravity": gravity,
        "friction": args.friction,
        "payload_mass_scale": args.payload_mass_scale,
        "motor_strength_scale": args.motor_strength_scale,
        "action_delay_s": args.action_delay,
        "action_delay_steps": action_delay_steps,
    }
    # This is the extension point for the user's vla task. Plain attributes are
    # also provided for configs that prefer direct field access.
    env_cfg.vla_physics = vla_physics
    env_cfg.payload_mass_scale = args.payload_mass_scale
    env_cfg.motor_strength_scale = args.motor_strength_scale
    env_cfg.action_delay_s = args.action_delay
    return {
        "vla_physics_field": "vla_physics",
        "applied": {
            "gravity": True,
            "friction": True,
            "action_delay": action_delay_applied,
            "payload_mass_scale": payload_applied,
            "motor_strength_scale": motor_applied,
        },
    }


def _make_physics(gravity: float, args: Args) -> np.ndarray:
    return np.asarray(
        [
            gravity,
            args.friction,
            args.payload_mass_scale,
            args.motor_strength_scale,
            args.action_delay,
        ],
        dtype=np.float32,
    )


def main(args: Args) -> None:
    if args.episodes_per_domain <= 0 or args.max_steps <= 0 or args.action_repeat <= 0:
        raise ValueError("episodes_per_domain, max_steps and action_repeat must be positive.")
    if args.sim_dt <= 0:
        raise ValueError("sim_dt must be positive.")
    if args.friction < 0 or args.payload_mass_scale <= 0 or args.motor_strength_scale <= 0:
        raise ValueError("friction must be non-negative; mass and motor scales must be positive.")
    if args.action_delay < 0:
        raise ValueError("action_delay must be non-negative.")

    # SRB imports must happen after Isaac Sim's AppLauncher is initialized.
    from srb.core.app import AppLauncher  # noqa: PLC0415
    from srb.utils.cache import update_offline_srb_cache  # noqa: PLC0415
    from srb.utils.cfg import load_cfg_from_registry  # noqa: PLC0415
    from srb.utils.path import SRB_APPS_DIR  # noqa: PLC0415

    enable_cameras = args.enable_cameras or args.env_id.endswith("_visual")
    experience = SRB_APPS_DIR.joinpath(
        f"srb.{'headless.' if args.headless else ''}{'rendering.' if enable_cameras else ''}kit"
    )
    launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=enable_cameras,
        experience=experience.as_posix(),
    )

    try:
        import gymnasium  # noqa: PLC0415
        from srb import tasks as _  # noqa: PLC0415

        update_offline_srb_cache()
        state_roots = parse_paths(args.state_keys)
        command_roots = parse_paths(args.command_keys)
        domains = tuple(domain.strip().lower() for domain in args.domains.split(",") if domain.strip())
        if not domains:
            raise ValueError("At least one domain is required.")

        records: list[dict[str, Any]] = []
        state_schema: ObservationSchema | None = None
        command_schema: ObservationSchema | None = None
        dataset_action_low: np.ndarray | None = None
        dataset_action_high: np.ndarray | None = None
        physics_application: dict[str, Any] | None = None
        episode_index = 0

        for domain_index, domain_name in enumerate(domains):
            env_cfg = load_cfg_from_registry(args.env_id, "task_cfg")
            env_cfg.seed = args.seed + domain_index
            env_cfg.num_envs = 1
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = args.device
            env_cfg, gravity = _set_domain(env_cfg, domain_name)
            application = _apply_physics_overrides(env_cfg, gravity, args)
            if physics_application is None:
                physics_application = application
            elif application["applied"] != physics_application["applied"]:
                raise ValueError("Physics application capabilities changed between domains.")
            physics = _make_physics(gravity, args)

            env = gymnasium.make(args.env_id, cfg=env_cfg)
            try:
                action_space = _get_action_space(env)
                low = _to_numpy(action_space.low).reshape(-1).astype(np.float32)
                high = _to_numpy(action_space.high).reshape(-1).astype(np.float32)
                if dataset_action_low is None:
                    dataset_action_low = low.copy()
                    dataset_action_high = high.copy()
                elif not np.allclose(low, dataset_action_low) or not np.allclose(high, dataset_action_high):
                    raise ValueError("Action bounds changed between domains; use separate datasets.")
                explorer = SmoothRandomExplorer(
                    low=low,
                    high=high,
                    config=SmoothExplorationConfig(dt=args.action_repeat * args.sim_dt),
                    seed=args.seed + domain_index,
                )

                for local_episode in range(args.episodes_per_domain):
                    seed = args.seed + 1000 * domain_index + local_episode
                    obs, _ = env.reset(seed=seed)
                    if state_schema is None:
                        state_schema = ObservationSchema.from_observation(obs, state_roots)
                        command_schema = ObservationSchema.from_observation(obs, command_roots)
                        logger.info(
                            "State schema: %d dims (%s); command schema: %d dims (%s)",
                            state_schema.dim,
                            state_schema.paths,
                            command_schema.dim,
                            command_schema.paths,
                        )
                    elif command_schema is None:
                        raise AssertionError("command schema must be initialized with state schema.")

                    explorer.reset()
                    for block_index in range(args.max_steps):
                        state = state_schema.pack(obs)
                        command = command_schema.pack(obs)
                        action = explorer.sample(block_index * args.action_repeat * args.sim_dt)

                        reward_sum = 0.0
                        terminated = False
                        truncated = False
                        fell = False
                        next_obs = obs
                        for _ in range(args.action_repeat):
                            next_obs, reward, terminated_value, truncated_value, info = env.step(action[None, ...])
                            reward_sum += _to_scalar(reward)
                            fell = _info_reports_fall(info)
                            terminated = _to_bool(terminated_value) or fell
                            truncated = _to_bool(truncated_value)
                            if terminated or truncated:
                                break

                        if block_index + 1 == args.max_steps and not terminated and not truncated:
                            # The collector's own horizon is a truncation boundary too.
                            truncated = True

                        records.append(
                            {
                                "state": state,
                                "action": action,
                                "next_state": state_schema.pack(next_obs),
                                "command": command,
                                "physics": physics.copy(),
                                "reward": reward_sum,
                                "terminated": terminated,
                                "truncated": truncated,
                                "fall": fell,
                                "episode_id": episode_index,
                            }
                        )
                        obs = next_obs
                        if terminated or truncated:
                            break
                    logger.info(
                        "domain=%s episode=%d/%d collected (%d transitions total)",
                        domain_name,
                        local_episode + 1,
                        args.episodes_per_domain,
                        len(records),
                    )
                    episode_index += 1
            finally:
                env.close()

        if (
            state_schema is None
            or command_schema is None
            or dataset_action_low is None
            or dataset_action_high is None
            or physics_application is None
        ):
            raise RuntimeError("No transitions were collected.")
        metadata = {
            "state_schema": state_schema.to_dict(),
            "command_schema": command_schema.to_dict(),
            "action_low": dataset_action_low.tolist(),
            "action_high": dataset_action_high.tolist(),
            "domains": list(domains),
            "action_repeat": args.action_repeat,
            "sim_dt": args.sim_dt,
            "decision_hz": 1.0 / (args.action_repeat * args.sim_dt),
            "physics_order": [
                "gravity",
                "friction",
                "payload_mass_scale",
                "motor_strength_scale",
                "action_delay",
            ],
            "physics_application": physics_application,
        }
        dataset = TransitionDataset.from_records(records, metadata=metadata)
        dataset.save(args.output)
        logger.info(
            "Saved %d transitions from %d episodes to %s (state=%d, action=%d, command=%d, physics=%d)",
            dataset.size,
            dataset.episode_count,
            args.output,
            dataset.state_dim,
            dataset.action_dim,
            dataset.command_dim,
            dataset.physics_dim,
        )
    finally:
        launcher.app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

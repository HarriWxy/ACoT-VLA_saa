"""Run the learned SRB world model with receding-horizon CEM-MPC.

The controller consumes state, velocity command and explicit physics values;
it does not use the VLA checkpoint or imitate logged actions.  Run the
collection and training scripts first in the SRB/Isaac Sim environment.

Example::

    python examples/srb/run_cem_mpc.py \
        --checkpoint checkpoints/srb_world_model \
        --domain mars \
        --env-id srb/locomotion_velocity_tracking
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

# Isaac Sim owns most of the GPU memory in this process. Prevent JAX from
# reserving the whole device before the simulator is initialized.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import tyro

from openpi.control.cem_planner import CEMConfig
from openpi.control.cem_planner import CEMPlanner
from openpi.control.srb_state import ObservationSchema
from openpi.control.world_model_io import load_world_model

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    checkpoint: Path = Path("checkpoints/srb_world_model")
    env_id: str = "srb/locomotion_velocity_tracking"
    domain: str = "moon"
    max_steps: int = 250
    action_repeat: int | None = None

    friction: float = 0.5
    payload_mass_scale: float = 1.0
    motor_strength_scale: float = 1.0
    action_delay: float = 0.0

    num_samples: int = 2048
    num_iterations: int = 5
    elite_size: int = 128
    uncertainty_cost: float = 1.0
    seed: int = 0
    device: str = "cuda:0"
    headless: bool = False
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


def _physics(domain: str, args: Args) -> np.ndarray:
    domain = domain.lower()
    if domain not in _DOMAIN_GRAVITY:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {sorted(_DOMAIN_GRAVITY)}.")
    return np.asarray(
        [
            _DOMAIN_GRAVITY[domain],
            args.friction,
            args.payload_mass_scale,
            args.motor_strength_scale,
            args.action_delay,
        ],
        dtype=np.float32,
    )


def main(args: Args) -> None:
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    bundle = load_world_model(args.checkpoint)
    metadata = bundle.metadata
    state_metadata = metadata.get("state_schema")
    command_metadata = metadata.get("command_schema")
    if state_metadata is None or command_metadata is None:
        raise ValueError("Checkpoint metadata has no state_schema/command_schema.")
    state_schema = ObservationSchema.from_dict(state_metadata)
    command_schema = ObservationSchema.from_dict(command_metadata)

    action_low = np.asarray(metadata.get("action_low"), dtype=np.float32)
    action_high = np.asarray(metadata.get("action_high"), dtype=np.float32)
    if action_low.ndim != 1 or action_high.shape != action_low.shape:
        raise ValueError("Checkpoint metadata must contain one-dimensional action_low/action_high.")
    action_repeat = args.action_repeat or int(metadata.get("action_repeat", 1))
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive.")

    planner = CEMPlanner(
        model=bundle.model,
        parameters=bundle.parameters,
        action_low=action_low,
        action_high=action_high,
        stats=bundle.stats,
        config=CEMConfig(
            num_samples=args.num_samples,
            num_iterations=args.num_iterations,
            elite_size=args.elite_size,
            uncertainty_cost=args.uncertainty_cost,
            seed=args.seed,
        ),
    )
    physics = _physics(args.domain, args)
    if args.domain.lower() not in _DOMAIN_GRAVITY:
        raise ValueError(f"Unknown SRB domain '{args.domain}'.")

    # Isaac Sim/SRB imports must happen after AppLauncher is constructed.
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
        from srb.core.domain import Domain  # noqa: PLC0415

        update_offline_srb_cache()
        domain = Domain.from_str(args.domain)
        if domain is None:
            raise ValueError(f"Unknown SRB domain '{args.domain}'.")
        env_cfg = load_cfg_from_registry(args.env_id, "task_cfg")
        env_cfg.seed = args.seed
        env_cfg.num_envs = 1
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args.device
        env_cfg.domain = domain
        env = gymnasium.make(args.env_id, cfg=env_cfg)
        try:
            obs, _ = env.reset(seed=args.seed)
            if state_schema.dim != state_schema.pack(obs).size:
                raise ValueError("Runtime observation does not match checkpoint state schema.")
            if command_schema.dim != command_schema.pack(obs).size:
                raise ValueError("Runtime observation does not match checkpoint command schema.")

            action_space = env.unwrapped.single_action_space
            previous_action = (
                ((_to_numpy(action_space.low) + _to_numpy(action_space.high)) / 2.0).reshape(-1).astype(np.float32)
            )
            plan: np.ndarray | None = None
            total_reward = 0.0

            for step in range(args.max_steps):
                state = state_schema.pack(obs)
                command = command_schema.pack(obs)
                result = planner.plan(
                    state=state,
                    command=command,
                    physics=physics,
                    previous_action=previous_action,
                    warm_start=plan,
                )
                plan = result.actions
                action = result.first_action

                step_reward = 0.0
                terminated = False
                truncated = False
                for _ in range(action_repeat):
                    obs, reward, terminated_value, truncated_value, _info = env.step(action[None, ...])
                    step_reward += _to_scalar(reward)
                    terminated = _to_bool(terminated_value)
                    truncated = _to_bool(truncated_value)
                    if terminated or truncated:
                        break
                total_reward += step_reward
                previous_action = action
                if step % 10 == 0:
                    logger.info(
                        "step=%d reward=%.4f predicted=%.4f uncertainty=%.4f",
                        step,
                        step_reward,
                        float(np.max(result.predicted_scores)),
                        float(np.min(result.predicted_uncertainty)),
                    )
                if terminated or truncated:
                    logger.info("Episode ended at decision step %d", step)
                    break

            logger.info("Total reward: %.4f", total_reward)
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

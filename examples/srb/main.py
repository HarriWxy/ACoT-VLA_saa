import collections
import dataclasses
import logging

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    env_id: str = "srb/sample_collection_visual"
    prompt: str = "collect the sample"

    host: str = "0.0.0.0"
    port: int = 8000

    seed: int = 0
    device: str = "cuda:0"
    headless: bool = False
    enable_cameras: bool = True

    max_steps: int = 250
    replan_steps: int = 4
    log_interval: int = 10


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _to_bool(value) -> bool:
    return bool(np.asarray(_to_numpy(value)).reshape(-1)[0])


def _prepare_request(obs: dict, prompt: str) -> dict:
    request = {key: _to_numpy(value) for key, value in obs.items()}
    request["prompt"] = prompt
    return request


def main(args: Args) -> None:
    from srb.core.app import AppLauncher
    from srb.utils.cache import update_offline_srb_cache
    from srb.utils.cfg import load_cfg_from_registry
    from srb.utils.path import SRB_APPS_DIR

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
        import gymnasium
        from srb import tasks as _  # noqa: F401

        update_offline_srb_cache()

        env_cfg = load_cfg_from_registry(args.env_id, "task_cfg")
        env_cfg.seed = args.seed
        env_cfg.num_envs = 1
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args.device

        env = gymnasium.make(args.env_id, cfg=env_cfg)
        try:
            client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            logging.info("Server metadata: %s", client.get_server_metadata())

            obs, _ = env.reset()
            action_plan: collections.deque[np.ndarray] = collections.deque()

            for step in range(args.max_steps):
                if not action_plan:
                    result = client.infer(_prepare_request(obs, args.prompt))
                    action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if action_chunk.ndim != 2:
                        raise ValueError(f"Expected action chunk with 2 dims, got {action_chunk.shape}")
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"Need at least {args.replan_steps} planned actions, got {len(action_chunk)}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                expected_action_dim = int(env.unwrapped.single_action_space.shape[0])
                if action.shape[-1] != expected_action_dim:
                    raise ValueError(
                        f"Policy produced action dim {action.shape[-1]}, but SRB env expects {expected_action_dim}. "
                        "Update SRBDataConfig.action_dim to match the environment action space."
                    )
                low = np.asarray(env.unwrapped.single_action_space.low, dtype=np.float32)
                high = np.asarray(env.unwrapped.single_action_space.high, dtype=np.float32)
                action = np.clip(action, low, high)

                obs, reward, terminated, truncated, _ = env.step(action[None, ...])

                if step % max(1, args.log_interval) == 0:
                    logging.info(
                        "step=%d reward=%.4f terminated=%s truncated=%s",
                        step,
                        float(np.asarray(_to_numpy(reward)).reshape(-1)[0]),
                        _to_bool(terminated),
                        _to_bool(truncated),
                    )

                if _to_bool(terminated) or _to_bool(truncated):
                    logging.info("Episode finished at step %d", step)
                    break
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
import collections
import dataclasses
import logging
from pathlib import Path

import h5py
import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

@dataclasses.dataclass
class Args:
    env_id: str = "srb/sample_collection_visual"
    prompt: str = "collect the sample"

    host: str = "0.0.0.0"
    port: int = 8000

    output_dir: Path = Path("srb_data")
    num_episodes: int = 10
    max_steps: int = 250
    replan_steps: int = 4

    seed: int = 0
    device: str = "cuda:0"
    headless: bool = False
    enable_cameras: bool = True

def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value

def _prepare_request(obs: dict, prompt: str) -> dict:
    request = {key: _to_numpy(value) for key, value in obs.items()}
    request["prompt"] = prompt
    return request

def save_episode(episode_id: int, output_dir: Path, data: dict):
    """将单次 episode 的数据保存为 HDF5 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"episode_{episode_id:04d}.hdf5"
    
    with h5py.File(file_path, "w") as f:
        # 保存动作
        f.create_dataset("action", data=data["actions"])
        
        # 保存观测数据
        obs_group = f.create_group("observations")
        for key, value in data["observations"].items():
            obs_group.create_dataset(key, data=value)
            
    logging.info(f"Successfully saved episode {episode_id} to {file_path}")

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
            logging.info("Connected to expert server. Metadata: %s", client.get_server_metadata())

            for ep_id in range(args.num_episodes):
                logging.info(f"Collecting episode {ep_id + 1}/{args.num_episodes}...")
                obs, _ = env.reset()
                
                # 用于存储当前 episode 的所有数据
                episode_data = {
                    "observations": collections.defaultdict(list),
                    "actions": [],
                }
                
                action_plan: collections.deque[np.ndarray] = collections.deque()

                for step in range(args.max_steps):
                    # 1. 记录当前观测 (Observation)
                    current_obs_numpy = {key: _to_numpy(value) for key, value in obs.items()}
                    for key, value in current_obs_numpy.items():
                        episode_data["observations"][key].append(value)

                    # 2. 获取动作 (Action)
                    if not action_plan:
                        result = client.infer(_prepare_request(obs, args.prompt))
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2:
                            raise ValueError(f"Expected action chunk with 2 dims, got {action_chunk.shape}")
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    
                    # 动作裁剪以符合环境限制
                    low = np.asarray(env.unwrapped.single_action_space.low, dtype=np.float32)
                    high = np.asarray(env.unwrapped.single_action_space.high, dtype=np.float32)
                    action = np.clip(action, low, high)

                    # 记录动作
                    episode_data["actions"].append(action)

                    # 3. 执行动作并获取下一帧观测
                    obs, reward, terminated, truncated, _ = env.step(action[None, ...])

                    if terminated or truncated:
                        logging.info(f"Episode {ep_id} finished at step {step}")
                        break
                
                # 将列表转换为 numpy 数组以便保存
                final_obs = {k: np.array(v) for k, v in episode_data["observations"].items()}
                final_actions = np.array(episode_data["actions"])
                
                save_episode(ep_id, args.output_dir, {
                    "observations": final_obs,
                    "actions": final_actions
                })

        finally:
            env.close()
    finally:
        launcher.app.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

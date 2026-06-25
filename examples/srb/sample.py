import collections
import dataclasses
import logging
import shutil
from pathlib import Path

import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass
class Args:
    env_id: str = "srb/sample_collection_visual"
    prompt: str = "collect the sample"

    host: str = "0.0.0.0"
    port: int = 8000

    repo_id: str = "srb_dataset"
    fps: int = 10
    robot_type: str = "srb"

    num_episodes: int = 10
    max_steps: int = 250
    replan_steps: int = 4

    seed: int = 0
    device: str = "cuda:0"
    headless: bool = False
    enable_cameras: bool = True

    # 图像存储模式: "video" (编码为 mp4) 或 "image" (保存为 png)
    image_mode: str = "video"
    image_writer_processes: int = 10
    image_writer_threads: int = 5

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


def _detect_obs_features(obs: dict) -> dict:
    """根据首次观测自动检测特征定义, 用于创建 LeRobotDataset。

    SRB 环境返回的 obs key 到 LeRobot feature 的映射:
      - proprio / state / ... → observation.state  (拼接为一维向量)
      - image_base           → observation.images.image_base
      - image_wrist          → observation.images.image_wrist
    """
    state_keys = []
    image_keys = []
    for key in obs:
        val = _to_numpy(obs[key])
        if val.ndim >= 2 and val.shape[-1] == 3:
            image_keys.append(key)
        elif val.ndim == 1 or (val.ndim == 2 and val.shape[0] == 1):
            state_keys.append(key)

    features: dict = {}

    # 拼接所有低维状态到 observation.state
    if state_keys:
        state_dim = sum(_to_numpy(obs[k]).reshape(-1).shape[0] for k in state_keys)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
        }

    # 每个图像 key 对应一个 observation.images.<key> feature
    for key in image_keys:
        img = _to_numpy(obs[key])
        h, w = img.shape[:2]
        features[f"observation.images.{key}"] = {
            "dtype": "image",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    # 动作维度在创建时尚未知, 需要从第一次推理结果中获取
    return features, state_keys, image_keys


def create_lerobot_dataset(
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict,
    image_mode: str = "video",
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集。若已存在则先删除。"""
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        logging.info("Removed existing dataset at %s", dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=(image_mode == "video"),
        tolerance_s=0.0001,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

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

            # ── 第 1 步: 用首次观测检测特征结构 ──────────────────────
            obs, _ = env.reset()
            features, state_keys, image_keys = _detect_obs_features(obs)
            action_space = env.unwrapped.single_action_space
            action_dim = int(action_space.shape[0])

            # 将 action feature 加入 features
            features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
            }

            logging.info("State keys: %s, Image keys: %s, Action dim: %d", state_keys, image_keys, action_dim)
            logging.info("LeRobot features: %s", list(features.keys()))

            # ── 第 2 步: 创建 LeRobot 数据集 ─────────────────────────
            dataset = create_lerobot_dataset(
                repo_id=args.repo_id,
                fps=args.fps,
                robot_type=args.robot_type,
                features=features,
                image_mode=args.image_mode,
                image_writer_processes=args.image_writer_processes,
                image_writer_threads=args.image_writer_threads,
            )

            # ── 第 3 步: 逐 episode 采集数据 ─────────────────────────
            for ep_id in range(args.num_episodes):
                logging.info("Collecting episode %d / %d ...", ep_id + 1, args.num_episodes)
                obs, _ = env.reset()

                action_plan: collections.deque[np.ndarray] = collections.deque()

                for step in range(args.max_steps):
                    # 1) 构建 LeRobot frame
                    frame: dict = {}

                    # 拼接低维状态
                    state_parts = [_to_numpy(obs[k]).reshape(-1) for k in state_keys if k in obs]
                    if state_parts:
                        frame["observation.state"] = np.concatenate(state_parts).astype(np.float32)

                    # 图像
                    for key in image_keys:
                        if key in obs:
                            img = _to_numpy(obs[key])
                            # 确保 uint8 格式
                            if np.issubdtype(img.dtype, np.floating):
                                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                            frame[f"observation.images.{key}"] = img

                    # 2) 推理获取动作
                    if not action_plan:
                        result = client.infer(_prepare_request(obs, args.prompt))
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2:
                            raise ValueError(f"Expected action chunk with 2 dims, got {action_chunk.shape}")
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                    low = np.asarray(action_space.low, dtype=np.float32)
                    high = np.asarray(action_space.high, dtype=np.float32)
                    action = np.clip(action, low, high)

                    frame["action"] = action

                    # 3) 写入 dataset
                    dataset.add_frame(frame)

                    # 4) 执行动作
                    obs, reward, terminated, truncated, _ = env.step(action[None, ...])

                    if terminated or truncated:
                        logging.info("Episode %d finished at step %d", ep_id, step)
                        break

                # 保存当前 episode, task 使用 prompt 文本
                dataset.save_episode(task=args.prompt)
                logging.info(
                    "Episode %d saved. Total episodes in dataset: %d", ep_id, dataset.num_episodes
                )

            logging.info(
                "Data collection complete. Dataset saved at %s",
                LEROBOT_HOME / args.repo_id,
            )

        finally:
            env.close()
    finally:
        launcher.app.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

import asyncio
import dataclasses
import http
import logging
import time
import traceback
import os

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _to_numpy(value) -> np.ndarray:
    """Convert various types to numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Parse image to uint8 format (H, W, C)."""
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def _detect_obs_features(obs: dict) -> tuple[dict, list[str], list[str]]:
    """根据首次观测自动检测特征定义, 用于创建 LeRobotDataset。

    SRB 环境返回的 obs key 到 LeRobot feature 的映射:
      - proprio / state / ... → observation.state  (拼接为一维向量)
      - image_base           → observation.images.image_base
      - image_wrist          → observation.images.image_wrist
    """
    state_keys = ("state","proprio",)
    image_keys = []
    for key in obs:
        if key == "prompt":
            continue
        val = _to_numpy(obs[key])
        # 图像判定: (H, W, 3) 且 H >= 32 (排除小向量误判)
        if val.ndim == 3 and val.shape[-1] in (3, 4) and val.shape[0] >= 32:
            image_keys.append(key)
        # elif val.ndim == 1 or (val.ndim == 2 and val.shape[0] == 1):
        #     state_keys.append(key)

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
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }

    return features, state_keys, image_keys


@dataclasses.dataclass
class ServerConfig:
    """WebsocketPolicyServer 的配置参数。"""
    policy: _base_policy.BasePolicy
    host: str = "0.0.0.0"
    port: int | None = None
    metadata: dict | None = None
    
    # 数据集配置 
    repo_id: str = "srb_tracking" # srb_dataset
    fps: int = 10
    robot_type: str = "srb"
    
    # 数据收集配置
    enable_data_collection: bool = True
    image_mode: str = "video"  # "video" or "image"
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    
    # 动作配置
    action_dim: int = 32
    
    # ── 探索噪声配置 ──────────────────────────────────────────
    # 噪声模式:
    #   "none"   — 不加噪声
    #   "output" — 仅对最终输出动作加后处理噪声 (最简单)
    #   "initial"— 替换流匹配去噪的初始噪声 (影响整个生成轨迹, 推荐)
    #   "both"   — 同时使用 initial + output (最强探索)
    exploration_mode: str = "none"
    # 高斯噪声标准差 (用于 output 模式)
    exploration_noise_std: float = 0.05
    # Ornstein-Uhlenbeck 参数 (用于时间相关噪声, 比独立高斯更适合机器人探索)
    #   dx = theta * (mu - x) * dt + sigma * dW
    ou_theta: float = 0.15   # 均值回归速度 (越大越快回到 0)
    ou_sigma: float = 0.3    # 噪声强度
    # 初始噪声缩放因子 (用于 initial 模式, >1 增大探索范围)
    initial_noise_scale: float = 1.0
    # num_steps: 流匹配去噪步数 (传给 sample_actions)
    num_steps: int | None = None  # None 表示使用模型默认值


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    支持从客户端接收观测数据并收集到 LeRobotDataset 中用于后续 finetune。
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        config: ServerConfig | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._config = config or ServerConfig(
            policy=policy,
            host=host,
            port=port,
            metadata=metadata,
        )
        
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        
        # 数据收集状态
        self._dataset: LeRobotDataset | None = None
        self._state_keys: list[str] = []
        self._image_keys: list[str] = []
        self._features: dict = {}
        self._initialized: bool = False
        
        # 探索噪声状态 (Ornstein-Uhlenbeck 过程)
        self._ou_state: np.ndarray | None = None  # 当前 OU 状态

    def _init_dataset_from_obs(self, obs: dict) -> None:
        """根据首次接收到的观测数据动态初始化数据集特征。

        如果数据集已存在（支持多连接复用），则直接复用；否则根据检测到的特征创建新数据集。
        """
        if self._initialized:
            return
        
        # 检测观测特征
        self._features, self._state_keys, self._image_keys = _detect_obs_features(obs)
        
        # 添加动作特征
        self._features["action"] = {
            "dtype": "float32",
            "shape": (self._config.action_dim,),
            "names": ["actions"],
        }
        
        # 添加奖励特征 (用于强化学习微调)
        self._features["reward"] = {
            "dtype": "float32",
            "shape": (1,1),
            "names": ["reward"],
        }
        
        logger.info("Detected state keys: %s", self._state_keys)
        logger.info("Detected image keys: %s", self._image_keys)
        logger.info("LeRobot features: %s", list(self._features.keys()))
        
        dataset_path = "./dataset/" + self._config.repo_id
        
        # 检查数据集是否完整且可用
        meta_path = os.path.join(dataset_path, "meta", "info.json")
        meta_dir = os.path.join(dataset_path, "meta")
        data_dir = os.path.join(dataset_path, "data")
        
        # 检查数据集是否完整（需要info.json、episodes.jsonl、tasks.jsonl和parquet数据文件）
        has_info = os.path.exists(meta_path)
        has_episodes = os.path.exists(os.path.join(meta_dir, "episodes.jsonl"))
        has_tasks = os.path.exists(os.path.join(meta_dir, "tasks.jsonl"))
        has_data = False
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.parquet'):
                        has_data = True
                        break
                if has_data:
                    break
        
        is_complete = has_info and has_episodes and has_tasks and has_data
        
        # 如果数据集不存在或不完整，则创建新数据集
        if not os.path.exists(dataset_path) or not is_complete:
            # 如果目录存在但不完整，需要删除并重新创建
            if os.path.exists(dataset_path) and not is_complete:
                import shutil
                logger.warning("Incomplete dataset found at %s, removing and recreating...", dataset_path)
                shutil.rmtree(dataset_path)
            
            self._dataset = LeRobotDataset.create(
                root=dataset_path,
                repo_id=self._config.repo_id,
                fps=self._config.fps,
                robot_type=self._config.robot_type,
                features=self._features,
                # use_videos=(self._config.image_mode == "video"),
                tolerance_s=0.01,
                image_writer_processes=self._config.image_writer_processes,
                image_writer_threads=self._config.image_writer_threads,
            )
            logger.info("Created new LeRobotDataset with dynamic features at %s", dataset_path)
        else:
            # 使用本地数据集，跳过HuggingFace检查
            import json
            
            # 设置环境变量以强制离线模式
            original_env = os.environ.copy()
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            try:
                # 直接从本地meta.json读取数据集信息，避免网络请求
                meta_path = os.path.join(dataset_path, "meta", "info.json")
                with open(meta_path, 'r') as f:
                    meta_info = json.load(f)
                
                # 创建LeRobotDataset实例，确保完全离线
                self._dataset = LeRobotDataset(
                    repo_id=self._config.repo_id,
                    root=dataset_path,
                    download_videos=False,
                )
                
                # LeRobotDataset() 构造函数不初始化 episode_buffer（只有 .create() 会）
                # 追加新数据前必须手动创建
                self._dataset.episode_buffer = self._dataset.create_episode_buffer()
                
                # 验证本地数据集
                logger.info("Loaded local LeRobotDataset at %s with %d episodes", 
                           dataset_path, self._dataset.num_episodes)
                
                # 检查本地数据集的特征是否与预期一致
                if self._features:
                    local_features = set(self._dataset.meta.features.keys())
                    expected_features = set(self._features.keys())
                    missing_features = expected_features - local_features
                    if missing_features:
                        logger.warning("Local dataset missing features: %s", missing_features)
                
                logger.info("Reusing existing LeRobotDataset at %s", dataset_path)
                
            finally:
                # 恢复原始环境变量
                os.environ.clear()
                os.environ.update(original_env)
        
        self._initialized = True

    def _collect_frame(self, obs: dict, action: np.ndarray, task: str = "") -> None:
        """收集一帧数据到数据集中。"""
        if not self._config.enable_data_collection or self._dataset is None:
            return
        
        frame: dict = {}
        
        # 拼接低维状态
        if self._state_keys:
            state_parts = [_to_numpy(obs[k]).reshape(-1) for k in self._state_keys if k in obs]
            if state_parts:
                frame["observation.state"] = np.concatenate(state_parts).astype(np.float32)
        
        # 处理图像
        for key in self._image_keys:
            if key in obs:
                img = _parse_image(obs[key])
                frame[f"observation.images.{key}"] = img
        
        # 添加动作
        frame["action"] = action.astype(np.float32)
        frame["task"] = task
        # reward 存为 (1,1) ndarray, 匹配 feature shape, 兼容 validate_frame 和 Array2D
        reward = obs.get("reward", np.zeros((1,1), dtype=np.float32)).astype(np.float32)
        if isinstance(reward, np.ndarray) and reward.ndim <= 2:
            reward = reward.reshape(1, 1)
        else:
            reward = np.array([[float(reward)]], dtype=np.float32)
        frame["reward"] = reward
        # frame["reward"] = obs.get("reward", np.zeros(1, dtype=np.float32)).astype(np.float32) # reward  
        
        # 写入数据集
        self._dataset.add_frame(frame)

    def save_episode(self,) -> None:
        """保存当前 episode。"""
        if not self._config.enable_data_collection or self._dataset is None:
            return
        
        self._dataset.save_episode()
        logger.info("Episode saved. Total episodes: %d", self._dataset.num_episodes)

    def _reset_exploration_state(self) -> None:
        """重置探索噪声状态 (episode 开始时调用)。"""
        self._ou_state = None
        logger.debug("Exploration state reset")

    def _generate_initial_noise(self, action_horizon: int) -> np.ndarray:
        """生成用于替换流匹配初始噪声的探索噪声。

        返回 shape: (action_horizon, action_dim) 的噪声数组。
        通过缩放标准正态分布来控制探索范围。
        """
        scale = self._config.initial_noise_scale
        noise = np.random.randn(action_horizon, self._config.action_dim).astype(np.float32) * scale
        return noise

    def _apply_output_noise(self, action: np.ndarray) -> np.ndarray:
        """对输出动作应用 Ornstein-Uhlenbeck 时间相关噪声。

        OU 过程: dx = theta * (mu - x) * dt + sigma * dW
        - theta: 均值回归速度
        - sigma: 噪声强度
        - mu: 均值 (这里为 0)

        比独立高斯噪声更适合机器人探索, 因为噪声在时间上具有连续性。
        """
        cfg = self._config
        
        if cfg.exploration_noise_std <= 0:
            return action
        
        dt = 1.0  # 离散时间步
        
        # 初始化 OU 状态
        if self._ou_state is None or self._ou_state.shape != action.shape:
            self._ou_state = np.zeros_like(action)
        
        # OU 过程更新: x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma * sqrt(dt) * dW
        dW = np.random.randn(*action.shape).astype(np.float32)
        self._ou_state = (
            self._ou_state
            + cfg.ou_theta * (0.0 - self._ou_state) * dt
            + cfg.ou_sigma * np.sqrt(dt) * dW
        )
        
        # 缩放 OU 输出到目标标准差范围
        noise = self._ou_state * cfg.exploration_noise_std
        
        return action + noise

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        episode_step = 0
        task = ""
        
        # 探索模式配置
        mode = self._config.exploration_mode
        use_initial_noise = mode in ("initial", "both")
        use_output_noise = mode in ("output", "both")
        if mode not in ("none", "output", "initial", "both"):
            logger.warning("Unknown exploration_mode: %s, falling back to 'none'", mode)
            use_initial_noise = False
            use_output_noise = False
        
        if mode != "none":
            logger.info(
                "Exploration enabled: mode=%s, noise_std=%.4f, ou_theta=%.3f, ou_sigma=%.3f, initial_scale=%.3f",
                mode, self._config.exploration_noise_std,
                self._config.ou_theta, self._config.ou_sigma,
                self._config.initial_noise_scale,
            )
        
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                done = obs["done"] if "done" in obs else False
                task = obs["prompt"] if "prompt" in obs else ""
                # 支持两种消息格式：
                # 1. 简单格式: 只包含观测数据
                # 2. 完整格式: 包含 obs, task, done 等字段
                # 如果 episode 结束，保存当前数据
                if done:
                    self.save_episode()
                    episode_step = 0
                    self._reset_exploration_state()
                    logger.info("Episode completed and saved")
                    await websocket.send(packer.pack({}))
                    continue

                
                # 如果是首次接收到观测数据且启用了数据收集，动态初始化数据集
                if self._config.enable_data_collection and not self._initialized:
                    self._init_dataset_from_obs(obs)
                
                # ── 层次 1: 生成初始探索噪声 (替换流匹配初始噪声) ──
                initial_noise = None
                if use_initial_noise:
                    initial_noise = self._generate_initial_noise(
                        action_horizon=self._config.action_dim  # 会被 infer 内部截断
                    )
                    # 包装为与 Policy.infer 期望格式兼容的 noise 参数
                    # Policy.infer 期望 noise shape: (action_horizon, action_dim) 或 (1, action_horizon, action_dim)
                
                infer_time = time.monotonic()
                # 传递初始噪声和 num_steps 到推理
                infer_kwargs = {}
                if initial_noise is not None:
                    infer_kwargs["noise"] = initial_noise
                if self._config.num_steps is not None:
                    infer_kwargs["num_steps"] = self._config.num_steps
                result = self._policy.infer(obs, **infer_kwargs)
                infer_time = time.monotonic() - infer_time
                
                # 提取动作
                action = result.get("actions", np.zeros(self._config.action_dim))
                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 2:
                    action = action[0]  # 取第一个动作

                # ── 层次 2: 输出后处理噪声 (Ornstein-Uhlenbeck) ──
                if use_output_noise:
                    action = self._apply_output_noise(action)
                
                # 收集数据
                self._collect_frame(obs, action, task)
                
                # 添加服务器时间信息
                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
                episode_step += 1
                
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                # 保存未完成的 episode
                if episode_step > 0:
                    self.save_episode()  # 要先关闭仿真再退出这边
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

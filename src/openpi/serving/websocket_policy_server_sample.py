import asyncio
from collections.abc import Mapping
import dataclasses
import http
import logging
import os
import time
import traceback
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


_TRANSPORT_KEYS = frozenset(
    {
        "done",
        "terminated",
        "truncated",
        "reward",
        "executed_action",
        "task",
    }
)


@dataclasses.dataclass(frozen=True)
class _Request:
    """One request plus feedback for the action returned by the prior request."""

    observation: dict[str, Any]
    task: str
    reward: Any | None
    done: bool
    executed_action: Any | None


@dataclasses.dataclass(frozen=True)
class _PendingFrame:
    """A policy action waiting for the next observation/feedback message."""

    observation: dict[str, Any]
    action: np.ndarray
    task: str


def _to_numpy(value: Any) -> np.ndarray:
    """Convert various types to numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _to_bool(value: Any) -> bool:
    array = _to_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one boolean value, got shape {array.shape}.")
    return bool(array[0])


def _to_task(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_reward(value: Any | None) -> np.ndarray:
    """Return the scalar transition reward in the LeRobot feature shape."""

    if value is None:
        return np.zeros((1, 1), dtype=np.float32)
    reward = _to_numpy(value).astype(np.float32).reshape(-1)
    if reward.size != 1:
        raise ValueError(f"Expected a scalar reward, got shape {reward.shape}.")
    return reward.reshape(1, 1)


def _normalize_executed_action(value: Any, expected_dim: int) -> np.ndarray:
    action = _to_numpy(value).astype(np.float32)
    if action.ndim > 1:
        if action.shape[0] != 1:
            raise ValueError(
                "executed_action must contain exactly one action; send one feedback message per recorded frame."
            )
        action = action[0]
    action = action.reshape(-1)
    if action.size != expected_dim:
        raise ValueError(f"executed_action has {action.size} values; expected {expected_dim}.")
    return action


def _parse_request(message: Mapping[str, Any]) -> _Request:
    """Normalize the legacy flat and the explicit ``{obs, ...}`` protocols.

    The current request is the post-step observation for the prior returned action.
    ``reward`` and ``executed_action`` therefore describe that prior action, rather
    than the action generated for this request.
    """

    envelope = dict(message)
    wrapped_obs = envelope.get("obs")
    observation = dict(wrapped_obs) if isinstance(wrapped_obs, Mapping) else dict(envelope)

    def value_for(key: str) -> Any | None:
        if key in envelope:
            return envelope[key]
        return observation.get(key)

    task_value = value_for("task")
    if task_value is None:
        task_value = value_for("prompt")
    task = _to_task(task_value) if task_value is not None else ""
    if task and "prompt" not in observation:
        observation["prompt"] = task

    done = any(
        _to_bool(value)
        for value in (value_for("done"), value_for("terminated"), value_for("truncated"))
        if value is not None
    )
    reward = value_for("reward")
    executed_action = value_for("executed_action")
    for key in _TRANSPORT_KEYS:
        observation.pop(key, None)
    return _Request(
        observation=observation,
        task=task,
        reward=reward,
        done=done,
        executed_action=executed_action,
    )


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
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an image with shape (H, W, 3), got {image.shape}.")
    return image.astype(np.uint8)


def _looks_like_image(value: Any) -> bool:
    image = _to_numpy(value)
    if image.ndim != 3:
        return False
    return (image.shape[-1] in (1, 3, 4) and image.shape[0] >= 32 and image.shape[1] >= 32) or (
        image.shape[0] in (1, 3, 4) and image.shape[1] >= 32 and image.shape[2] >= 32
    )


def _detect_obs_features(
    obs: Mapping[str, Any],
    *,
    configured_state_keys: tuple[str, ...] | None,
    configured_image_keys: tuple[str, ...] | None,
) -> tuple[dict, list[str], list[str]]:
    """根据首次观测自动检测特征定义, 用于创建 LeRobotDataset。

    SRB 环境返回的 obs key 到 LeRobot feature 的映射:
      - proprio / state / ... → observation.state  (拼接为一维向量)
      - image_base           → observation.images.image_base
      - image_wrist          → observation.images.image_wrist
    """
    if configured_image_keys is None:
        image_keys = [key for key in sorted(obs) if _looks_like_image(obs[key])]
    else:
        missing_image_keys = [key for key in configured_image_keys if key not in obs]
        if missing_image_keys:
            raise KeyError(f"Configured image keys are missing: {missing_image_keys}.")
        image_keys = list(configured_image_keys)

    if configured_state_keys is None:
        state_keys = []
        for key in sorted(obs):
            if key == "prompt" or key in image_keys:
                continue
            value = _to_numpy(obs[key])
            if np.issubdtype(value.dtype, np.number) and value.ndim <= 2:
                state_keys.append(key)
    else:
        missing_state_keys = [key for key in configured_state_keys if key not in obs]
        if missing_state_keys:
            raise KeyError(f"Configured state keys are missing: {missing_state_keys}.")
        state_keys = list(configured_state_keys)

    if not state_keys:
        raise ValueError("No numeric low-dimensional state fields were detected.")

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
        img = _parse_image(obs[key])
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
    repo_id: str = "srb_tracking"  # srb_dataset
    fps: int = 10
    robot_type: str = "srb"

    # 数据收集配置
    enable_data_collection: bool = True
    # ``None`` 自动收集全部数值低维字段；显式指定可固定 VLA state 的布局。
    state_keys: tuple[str, ...] | None = ("state", "proprio")
    # ``None`` 自动发现 HWC/CHW 图像字段。
    image_keys: tuple[str, ...] | None = None
    image_mode: str = "video"  # "video" or "image"
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    # 非完整数据集默认不删除，避免服务重启时静默丢失已有采集结果。
    recreate_incomplete_dataset: bool = False

    # 动作配置
    # 策略对外返回的实际动作维度。None 时由首次推理结果自动确定。
    action_dim: int | None = None

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
    ou_theta: float = 0.15  # 均值回归速度 (越大越快回到 0)
    ou_sigma: float = 0.3  # 噪声强度
    # 初始噪声缩放因子 (用于 initial 模式, >1 增大探索范围)
    initial_noise_scale: float = 1.0
    # num_steps: 流匹配去噪步数 (传给 sample_actions)
    num_steps: int | None = None  # None 表示使用模型默认值
    # initial/both 模式必须提供模型内部的 action chunk 形状；它和对外动作维度可不同。
    model_action_horizon: int | None = None
    model_action_dim: int | None = None


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
        self._action_dim: int | None = None
        # LeRobot uses one mutable episode buffer, so collection cannot safely
        # interleave frames from concurrent websocket clients.
        self._collection_lock = asyncio.Lock()

        # 探索噪声状态 (Ornstein-Uhlenbeck 过程)
        self._ou_state: np.ndarray | None = None  # 当前 OU 状态

    def _init_dataset_from_obs(self, obs: Mapping[str, Any], action_dim: int) -> None:
        """根据首次接收到的观测数据动态初始化数据集特征。

        如果数据集已存在（支持多连接复用），则直接复用；否则根据检测到的特征创建新数据集。
        """
        if self._initialized:
            return

        # 检测观测特征
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if self._config.action_dim is not None and self._config.action_dim != action_dim:
            raise ValueError(
                f"Policy returned action dim {action_dim}, but ServerConfig.action_dim is {self._config.action_dim}."
            )

        self._features, self._state_keys, self._image_keys = _detect_obs_features(
            obs,
            configured_state_keys=self._config.state_keys,
            configured_image_keys=self._config.image_keys,
        )
        self._action_dim = action_dim

        # 添加动作特征
        self._features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        }

        # 添加奖励特征 (用于强化学习微调)
        self._features["reward"] = {
            "dtype": "float32",
            "shape": (1, 1),
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
                    if f.endswith(".parquet"):
                        has_data = True
                        break
                if has_data:
                    break

        is_complete = has_info and has_episodes and has_tasks and has_data

        # 如果数据集不存在或不完整，则创建新数据集
        if not os.path.exists(dataset_path) or not is_complete:
            # 目录可能包含一轮中断采集；除非显式允许，不能静默删除。
            if os.path.exists(dataset_path) and not is_complete:
                if not self._config.recreate_incomplete_dataset:
                    raise RuntimeError(
                        f"Incomplete dataset found at {dataset_path}. Set "
                        "recreate_incomplete_dataset=True only if deleting it is intended."
                    )
                import shutil

                logger.warning("Incomplete dataset found at %s, removing and recreating...", dataset_path)
                shutil.rmtree(dataset_path)

            self._dataset = LeRobotDataset.create(
                root=dataset_path,
                repo_id=self._config.repo_id,
                fps=self._config.fps,
                robot_type=self._config.robot_type,
                features=self._features,
                use_videos=(self._config.image_mode == "video"),
                tolerance_s=0.01,
                image_writer_processes=self._config.image_writer_processes,
                image_writer_threads=self._config.image_writer_threads,
            )
            logger.info("Created new LeRobotDataset with dynamic features at %s", dataset_path)
        else:
            # 使用本地数据集，跳过HuggingFace检查
            # 设置环境变量以强制离线模式
            original_env = os.environ.copy()
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

            try:
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
                logger.info(
                    "Loaded local LeRobotDataset at %s with %d episodes", dataset_path, self._dataset.num_episodes
                )

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

    def _collect_frame(
        self,
        obs: Mapping[str, Any],
        action: np.ndarray,
        reward: Any | None,
    ) -> None:
        """Commit a completed observation/action/reward frame to LeRobot."""

        if not self._config.enable_data_collection or self._dataset is None:
            return
        if self._action_dim is None:
            raise RuntimeError("Dataset action dimension was not initialized.")

        missing_state_keys = [key for key in self._state_keys if key not in obs]
        if missing_state_keys:
            raise KeyError(f"Observation is missing state keys required by the dataset: {missing_state_keys}.")
        missing_image_keys = [key for key in self._image_keys if key not in obs]
        if missing_image_keys:
            raise KeyError(f"Observation is missing image keys required by the dataset: {missing_image_keys}.")

        frame: dict[str, Any] = {}
        state = np.concatenate([_to_numpy(obs[key]).reshape(-1) for key in self._state_keys]).astype(np.float32)
        expected_state_dim = int(self._features["observation.state"]["shape"][0])
        if state.size != expected_state_dim:
            raise ValueError(f"State has {state.size} values; expected {expected_state_dim}.")
        frame["observation.state"] = state

        for key in self._image_keys:
            image = _parse_image(obs[key])
            expected_shape = tuple(self._features[f"observation.images.{key}"]["shape"])
            if image.shape != expected_shape:
                raise ValueError(f"Image '{key}' has shape {image.shape}; expected {expected_shape}.")
            frame[f"observation.images.{key}"] = image

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self._action_dim:
            raise ValueError(f"Action has {action.size} values; expected {self._action_dim}.")
        frame["action"] = action
        frame["reward"] = _normalize_reward(reward)
        self._dataset.add_frame(frame)

    def _commit_pending_frame(self, pending: _PendingFrame, request: _Request) -> None:
        """Commit the previous action using feedback carried by this request."""

        action = pending.action
        if request.executed_action is not None:
            action = _normalize_executed_action(request.executed_action, pending.action.size)
        self._collect_frame(pending.observation, action, request.reward)

    def save_episode(self, task: str = "") -> None:
        """保存当前 episode。"""
        if not self._config.enable_data_collection or self._dataset is None:
            return

        if task:
            self._dataset.save_episode(task=task)
        else:
            self._dataset.save_episode()
        logger.info("Episode saved. Total episodes: %d", self._dataset.num_episodes)

    def _reset_exploration_state(self) -> None:
        """重置探索噪声状态 (episode 开始时调用)。"""
        self._ou_state = None
        logger.debug("Exploration state reset")

    def _generate_initial_noise(self) -> np.ndarray:
        """生成用于替换流匹配初始噪声的探索噪声。

        返回 shape: (model_action_horizon, model_action_dim) 的噪声数组。
        通过缩放标准正态分布来控制探索范围。
        """
        horizon = self._config.model_action_horizon
        action_dim = self._config.model_action_dim
        if horizon is None or action_dim is None:
            raise ValueError("initial/both exploration requires model_action_horizon and model_action_dim.")
        scale = self._config.initial_noise_scale
        noise = np.random.randn(horizon, action_dim).astype(np.float32) * scale
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
        self._ou_state = self._ou_state + cfg.ou_theta * (0.0 - self._ou_state) * dt + cfg.ou_sigma * np.sqrt(dt) * dW

        # 缩放 OU 输出到目标标准差范围
        noise = self._ou_state * cfg.exploration_noise_std

        return action + noise

    def _prepare_action_result(
        self,
        result: Mapping[str, Any],
        *,
        apply_output_noise: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Return the outgoing result and its first, actually returned action."""

        if "actions" not in result:
            raise KeyError("Policy result is missing the required 'actions' field.")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            if actions.size == 0:
                raise ValueError("Policy returned an empty action.")
            first_action = actions.reshape(-1).copy()
        elif actions.ndim == 2:
            if actions.shape[0] == 0 or actions.shape[1] == 0:
                raise ValueError(f"Policy returned an empty action chunk with shape {actions.shape}.")
            first_action = actions[0].reshape(-1).copy()
        else:
            raise ValueError(f"Policy actions must be rank 1 or 2, got shape {actions.shape}.")

        outgoing = dict(result)
        if apply_output_noise:
            first_action = self._apply_output_noise(first_action)
            returned_actions = actions.copy()
            if returned_actions.ndim == 1:
                returned_actions = first_action
            else:
                returned_actions[0] = first_action
            outgoing["actions"] = returned_actions
        return outgoing, first_action

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
        """Serve one client, serializing collection around LeRobot's episode buffer."""

        if self._config.enable_data_collection:
            async with self._collection_lock:
                await self._serve_connection(websocket)
        else:
            await self._serve_connection(websocket)

    async def _serve_connection(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        committed_steps = 0
        episode_task = ""
        pending: _PendingFrame | None = None
        self._reset_exploration_state()

        mode = self._config.exploration_mode
        use_initial_noise = mode in ("initial", "both")
        use_output_noise = mode in ("output", "both")
        if mode not in ("none", "output", "initial", "both"):
            logger.warning("Unknown exploration_mode: %s, falling back to 'none'", mode)
            use_initial_noise = False
            use_output_noise = False
        elif mode != "none":
            logger.info(
                "Exploration enabled: mode=%s, noise_std=%.4f, ou_theta=%.3f, ou_sigma=%.3f, initial_scale=%.3f",
                mode,
                self._config.exploration_noise_std,
                self._config.ou_theta,
                self._config.ou_sigma,
                self._config.initial_noise_scale,
            )

        while True:
            try:
                start_time = time.monotonic()
                message = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(message, Mapping):
                    raise TypeError(f"Expected a mapping request, got {type(message).__name__}.")
                request = _parse_request(message)

                # The new observation/reward closes the action returned by the
                # previous inference. This avoids pairing reward_t-1 with action_t.
                if pending is not None:
                    self._commit_pending_frame(pending, request)
                    committed_steps += 1
                    episode_task = episode_task or pending.task
                    pending = None

                if request.done:
                    if committed_steps > 0:
                        self.save_episode(episode_task)
                    else:
                        logger.warning("Received done without a completed data frame; no episode was saved.")
                    committed_steps = 0
                    episode_task = ""
                    self._reset_exploration_state()
                    logger.info("Episode completed and saved")
                    await websocket.send(packer.pack({}))
                    continue

                initial_noise = self._generate_initial_noise() if use_initial_noise else None
                infer_kwargs: dict[str, Any] = {}
                if initial_noise is not None:
                    infer_kwargs["noise"] = initial_noise
                if self._config.num_steps is not None:
                    infer_kwargs["num_steps"] = self._config.num_steps

                infer_time = time.monotonic()
                result = self._policy.infer(request.observation, **infer_kwargs)
                infer_time = time.monotonic() - infer_time
                if not isinstance(result, Mapping):
                    raise TypeError(f"Policy result must be a mapping, got {type(result).__name__}.")
                outgoing, action = self._prepare_action_result(result, apply_output_noise=use_output_noise)

                if self._config.enable_data_collection:
                    if not self._initialized:
                        self._init_dataset_from_obs(request.observation, action.size)
                    elif self._action_dim != action.size:
                        raise ValueError(
                            "Policy action dimension changed from "
                            f"{self._action_dim} to {action.size} within one dataset."
                        )
                    pending = _PendingFrame(
                        observation=request.observation,
                        action=action,
                        task=request.task,
                    )
                    episode_task = episode_task or request.task

                outgoing["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    outgoing["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(outgoing))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                if pending is not None:
                    logger.warning(
                        "Dropping one unconfirmed frame on disconnect; send a final done "
                        "message with feedback to keep it."
                    )
                if committed_steps > 0:
                    self.save_episode(episode_task)
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

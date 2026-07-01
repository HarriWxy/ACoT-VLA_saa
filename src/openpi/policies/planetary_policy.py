"""行星环境的输入/输出 transforms。

将行星机器人仿真环境的原始观测转换为模型期望的格式，
包括图像、关节状态、以及物理参数 (重力、摩擦等)。
"""

import dataclasses
from collections.abc import Sequence

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """将图像统一为 (H, W, 3) uint8 格式。"""
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = image.transpose(1, 2, 0)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


# 行星物理参数的默认值 (地球基准)
PLANETARY_DEFAULTS = {
    "gravity": 9.81,
    "friction_coeff": 0.5,
    "robot_mass": 5.0,
    "air_density": 1.225,
    "terrain_roughness": 0.3,
}


@dataclasses.dataclass(frozen=True)
class PlanetaryInputs(transforms.DataTransformFn):
    """将行星环境观测转换为模型输入格式。

    输入数据格式 (来自仿真环境):
    {
        "image_base": (H, W, 3) uint8,
        "image_wrist": (H, W, 3) uint8,  (optional)
        "state": (action_dim,) float32,    关节状态
        "prompt": str,                     任务描述
        "gravity": float,                  重力加速度 m/s²
        "friction_coeff": float,           摩擦系数
        "robot_mass": float,               机器人质量 kg
        "air_density": float,              空气密度 kg/m³ (optional)
        "terrain_roughness": float,        地形粗糙度 (optional)
    }
    """

    action_dim: int
    physics_keys: Sequence[str] = (
        "gravity", "friction_coeff", "robot_mass", "air_density", "terrain_roughness"
    )
    image_keys: Sequence[str] = ("image_base", "image_wrist")

    def __call__(self, data: dict) -> dict:
        # 解析图像
        base_image = self._resolve_image(data, self.image_keys[0])
        wrist_image = (
            self._resolve_image(data, self.image_keys[1])
            if len(self.image_keys) > 1
            else None
        )
        if base_image is None:
            base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        if wrist_image is None:
            wrist_image = np.zeros_like(base_image)

        # 构造模型期望的图像格式
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (
            np.bool_(base_image.any()),
            np.bool_(wrist_image.any()),
            np.False_,
        )

        # 构造 state
        state = np.asarray(data.get("state", np.zeros(self.action_dim)), dtype=np.float32)
        if state.ndim > 1:
            state = state.reshape(-1)

        # 构造 physics_params
        physics_values = []
        for key in self.physics_keys:
            val = data.get(key, PLANETARY_DEFAULTS.get(key, 0.0))
            physics_values.append(float(val))
        physics_params = np.array(physics_values, dtype=np.float32)

        inputs = {
            "state": state,
            "image": dict(zip(names, images)),
            "image_mask": dict(zip(names, image_masks)),
            "physics_params": physics_params,
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

    def _resolve_image(self, data: dict, key: str) -> np.ndarray | None:
        if key in data:
            return _parse_image(data[key])
        if "image" in data and isinstance(data["image"], dict) and key in data["image"]:
            return _parse_image(data["image"][key])
        return None


@dataclasses.dataclass(frozen=True)
class PlanetaryOutputs(transforms.DataTransformFn):
    """将模型输出转换为行星环境期望的动作格式。"""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]  # 去掉 batch 维度
        return {"actions": actions[:, : self.action_dim]}
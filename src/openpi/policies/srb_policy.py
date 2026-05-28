import dataclasses
from collections.abc import Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_srb_example() -> dict:
    """Creates a random input example for the SRB policy."""
    return {
        "proprio": np.random.rand(9).astype(np.float32),
        "state": np.random.rand(18).astype(np.float32),
        "image_cam_base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_cam_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "complete the task",
    }


def _flatten_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1).astype(np.float32)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3:
        raise ValueError(f"Expected image to have 3 dims, got shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image.astype(np.uint8)


def _build_state(data: dict, observation_keys: Sequence[str]) -> np.ndarray:
    state_parts = [_flatten_vector(data[key]) for key in observation_keys if key in data]
    if not state_parts:
        raise ValueError(f"Expected at least one observation key in {observation_keys}, got {tuple(data)}")
    return np.concatenate(state_parts, axis=-1)


def _resolve_image(data: dict, image_key: str) -> np.ndarray | None:
    if image_key in data:
        return _parse_image(data[image_key])
    if "image" in data and isinstance(data["image"], dict) and image_key in data["image"]:
        return _parse_image(data["image"][image_key])
    return None


@dataclasses.dataclass(frozen=True)
class SRBInputs(transforms.DataTransformFn):
    """Converts SRB observations into the model input format.

    Expected SRB inputs are flat dictionaries produced by SRB envs, e.g. keys such as
    `state`, `proprio`, `state_dyn`, `proprio_dyn`, `image_cam_base`, and `image_cam_wrist`.
    The transform also accepts an `image` dictionary for offline datasets if users choose to
    store the raw camera frames under nested keys.
    """

    action_dim: int
    model_type: _model.ModelType
    observation_keys: Sequence[str] = ("proprio", "state")
    image_keys: Sequence[str] = ("image_cam_base", "image_cam_wrist")
    default_image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION
    strict_state_dim: bool = False

    def __call__(self, data: dict) -> dict:
        state = _build_state(data, self.observation_keys)
        if state.shape[-1] > self.action_dim:
            if self.strict_state_dim:
                raise ValueError(
                    f"SRB state dim {state.shape[-1]} exceeds model action dim {self.action_dim}. "
                    "Either reduce observation_keys or disable strict_state_dim."
                )
            state = state[: self.action_dim]
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _resolve_image(data, self.image_keys[0])
        wrist_image = _resolve_image(data, self.image_keys[1]) if len(self.image_keys) > 1 else None
        if base_image is None:
            height, width = self.default_image_resolution
            base_image = np.zeros((height, width, 3), dtype=np.uint8)
        if wrist_image is None:
            wrist_image = np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (
                    np.bool_(base_image.any()),
                    np.bool_(wrist_image.any()),
                    np.False_,
                )
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.shape[-1] > self.action_dim:
                actions = actions[..., : self.action_dim]
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SRBOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
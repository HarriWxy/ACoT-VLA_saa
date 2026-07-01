"""Gfootball multi-agent VLA policy transforms.

Converts gfootball raw observations into the model input format and
converts model outputs back to discrete football actions.

Observation space (per agent):
  - rendered frame (72x96x4 or rendered) → image input
  - simple115_v2 vector (115-dim) → state input
  - task prompt (e.g. "left team player 3: attack")

Action space: Discrete(19) football actions
  0=idle, 1-8=move, 9=long_pass, 10=high_pass, 11=short_pass,
  12=shot, 13=sprint, 14=release_move, 15=release_sprint,
  16=slide, 17=dribble, 18=release_dribble
"""

import dataclasses
from collections.abc import Sequence
from typing import Any

import einops
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms
from openpi.models import model as _model
from openpi.models import acot_vla as _acot_vla
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at


# Football action names for prompt generation
FOOTBALL_ACTIONS = [
    "idle", "left", "top_left", "top", "top_right",
    "right", "bottom_right", "bottom", "bottom_left",
    "long_pass", "high_pass", "short_pass", "shot",
    "sprint", "release_move", "release_sprint",
    "slide", "dribble", "release_dribble",
]

# Player role names (gfootball role indices)
PLAYER_ROLES = {
    0: "goalkeeper",
    1: "center_back",
    2: "left_back",
    3: "right_back",
    4: "defensive_midfielder",
    5: "central_midfielder",
    6: "left_midfielder",
    7: "right_midfielder",
    8: "attacking_midfielder",
    9: "center_forward",
    10: "striker",
}


def make_gfootball_example(n_agents: int = 4, image_shape: tuple[int, int] = (72, 96)) -> dict:
    """Creates a random input example for the gfootball policy."""
    h, w = image_shape
    return {
        "image": np.random.randint(256, size=(h, w, 3), dtype=np.uint8),
        "state": np.random.rand(115).astype(np.float32),
        "actions": np.random.randint(0, 19, size=(1,)).astype(np.int32),
        "prompt": "left team player 0: control the ball and attack",
        "agent_id": 0,
        "team_id": 0,
        "player_role": "goalkeeper",
    }


def _parse_image(image: np.ndarray, target_shape: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Parse and normalize image from various formats."""
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


def _flatten_vector(value: np.ndarray) -> np.ndarray:
    """Flatten nested array to 1D float32."""
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1).astype(np.float32)


def _build_simple115_obs(obs: dict) -> np.ndarray:
    """Build simple115_v2 observation vector from raw gfootball observation.

    This produces a 115-dim vector:
      - left_team positions: 11*2 = 22
      - left_team directions: 11*2 = 22
      - right_team positions: 11*2 = 22
      - right_team directions: 11*2 = 22
      - ball position: 3
      - ball direction: 3
      - active player one-hot: 11
      - game_mode one-hot: 7
      - score: 2
      - steps_left (normalized): 1
    Total: 115
    """
    parts = []

    # Left team positions and directions (pad/truncate to 11)
    left_team = np.array(obs.get("left_team", np.zeros((11, 2))))
    left_dir = np.array(obs.get("left_team_direction", np.zeros((11, 2))))
    if len(left_team) < 11:
        left_team = np.pad(left_team, ((0, 11 - len(left_team)), (0, 0)))
        left_dir = np.pad(left_dir, ((0, 11 - len(left_dir)), (0, 0)))
    parts.append(left_team[:11].flatten())
    parts.append(left_dir[:11].flatten())

    # Right team positions and directions
    right_team = np.array(obs.get("right_team", np.zeros((11, 2))))
    right_dir = np.array(obs.get("right_team_direction", np.zeros((11, 2))))
    if len(right_team) < 11:
        right_team = np.pad(right_team, ((0, 11 - len(right_team)), (0, 0)))
        right_dir = np.pad(right_dir, ((0, 11 - len(right_dir)), (0, 0)))
    parts.append(right_team[:11].flatten())
    parts.append(right_dir[:11].flatten())

    # Ball position and direction
    ball = np.array(obs.get("ball", [0, 0, 0]))[:3]
    ball_dir = np.array(obs.get("ball_direction", [0, 0, 0]))[:3]
    parts.append(ball)
    parts.append(ball_dir)

    # Active player one-hot (11 dims)
    active = obs.get("active", 0)
    active_onehot = np.zeros(11, dtype=np.float32)
    if 0 <= active < 11:
        active_onehot[active] = 1.0
    parts.append(active_onehot)

    # Game mode one-hot (7 dims: Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty)
    game_mode = obs.get("game_mode", 0)
    mode_onehot = np.zeros(7, dtype=np.float32)
    if 0 <= game_mode < 7:
        mode_onehot[game_mode] = 1.0
    parts.append(mode_onehot)

    # Score
    score = np.array(obs.get("score", [0, 0]))[:2].astype(np.float32)
    parts.append(score)

    # Steps left (normalized by max 3000)
    steps_left = np.array([obs.get("steps_left", 3000) / 3000.0], dtype=np.float32)
    parts.append(steps_left)

    return np.concatenate(parts, axis=-1).astype(np.float32)


def get_player_role_name(obs: dict, player_index: int) -> str:
    """Get the role name for a player from observation."""
    roles = obs.get("left_team_roles", []) if player_index < 11 else obs.get("right_team_roles", [])
    local_idx = player_index if player_index < 11 else player_index - 11
    if local_idx < len(roles):
        return PLAYER_ROLES.get(int(roles[local_idx]), "player")
    return "player"


def build_agent_prompt(
    player_index: int,
    obs: dict,
    task_instruction: str | None = None,
) -> str:
    """Build a natural language prompt for a specific agent.

    Args:
        player_index: Global player index (0-10 left, 11-21 right)
        obs: Raw gfootball observation dict
        task_instruction: Optional task override

    Returns:
        Prompt string describing the agent's role and objective
    """
    team = "left" if player_index < 11 else "right"
    local_idx = player_index if player_index < 11 else player_index - 11
    role = get_player_role_name(obs, player_index)

    if task_instruction:
        return f"{team} team {role} (player {local_idx}): {task_instruction}"

    # Default task based on role
    if role == "goalkeeper":
        return f"{team} team goalkeeper: defend the goal and clear the ball"
    elif role in ("center_back", "left_back", "right_back"):
        return f"{team} team {role}: defend and build up play from the back"
    elif role in ("defensive_midfielder", "central_midfielder"):
        return f"{team} team {role}: control the midfield and distribute the ball"
    elif role in ("left_midfielder", "right_midfielder"):
        return f"{team} team {role}: provide width and cross the ball"
    elif role in ("attacking_midfielder",):
        return f"{team} team {role}: create chances and assist forwards"
    elif role in ("center_forward", "striker"):
        return f"{team} team {role}: score goals and lead the attack"
    else:
        return f"{team} team player {local_idx}: play football and help your team score"


@dataclasses.dataclass(frozen=True)
class GfootballInputs(transforms.DataTransformFn):
    """Converts gfootball observations into the model input format.

    Expected inputs from the dataset or inference environment:
      - "image": rendered game frame (H, W, 3) or (H, W, 4)
      - "state": simple115_v2 vector (115-dim) or raw obs dict
      - "actions": discrete action index (int) or one-hot (19-dim)
      - "prompt": task description string
      - "agent_id": player index (optional)
      - "team_id": 0=left, 1=right (optional)
    """

    action_dim: int = 19
    model_type: _model.ModelType = _model.ModelType.PI0_FAST
    use_rendered_image: bool = True
    default_image_resolution: tuple[int, int] = (224, 224)

    def __call__(self, data: dict) -> dict:
        # Parse image
        image = None
        if "image" in data:
            image = _parse_image(data["image"], self.default_image_resolution)
        elif "frame" in data:
            image = _parse_image(data["frame"], self.default_image_resolution)
        elif "pixels" in data:
            image = _parse_image(data["pixels"], self.default_image_resolution)

        if image is None:
            h, w = self.default_image_resolution
            image = np.zeros((h, w, 3), dtype=np.uint8)

        # Parse state
        state = None
        if "state" in data:
            state_raw = data["state"]
            if isinstance(state_raw, dict):
                # Raw observation dict → build simple115 vector
                state = _build_simple115_obs(state_raw)
            else:
                state = _flatten_vector(state_raw)
        elif "obs" in data and isinstance(data["obs"], dict):
            state = _build_simple115_obs(data["obs"])
        elif "simple115" in data:
            state = _flatten_vector(data["simple115"])

        if state is None:
            state = np.zeros(115, dtype=np.float32)

        # Pad or truncate state to 115
        if state.shape[-1] > 115:
            state = state[:115]
        elif state.shape[-1] < 115:
            state = np.pad(state, (0, 115 - state.shape[-1]))

        # Parse actions
        actions = None
        if "actions" in data:
            act = np.asarray(data["actions"])
            if act.ndim == 0:
                # Scalar discrete action → one-hot
                act_idx = int(act)
                actions = np.zeros((1, self.action_dim), dtype=np.float32)
                if 0 <= act_idx < self.action_dim:
                    actions[0, act_idx] = 1.0
            elif act.ndim == 1:
                if act.shape[0] == self.action_dim:
                    # Already one-hot
                    actions = act.reshape(1, -1).astype(np.float32)
                elif act.shape[0] == 1:
                    # Single action index
                    act_idx = int(act[0])
                    actions = np.zeros((1, self.action_dim), dtype=np.float32)
                    if 0 <= act_idx < self.action_dim:
                        actions[0, act_idx] = 1.0
                else:
                    # Action sequence of indices
                    T = act.shape[0]
                    actions = np.zeros((T, self.action_dim), dtype=np.float32)
                    for t in range(T):
                        idx = int(act[t])
                        if 0 <= idx < self.action_dim:
                            actions[t, idx] = 1.0
            elif act.ndim == 2:
                actions = act.astype(np.float32)

        # Build prompt
        prompt = data.get("prompt", "")
        if not prompt:
            agent_id = data.get("agent_id", 0)
            obs_for_prompt = data.get("obs", data.get("state", {}))
            if isinstance(obs_for_prompt, dict):
                prompt = build_agent_prompt(agent_id, obs_for_prompt)
            else:
                team = "left" if agent_id < 11 else "right"
                local_idx = agent_id if agent_id < 11 else agent_id - 11
                prompt = f"{team} team player {local_idx}: play football"

        # Build model inputs
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (image, np.zeros_like(image), np.zeros_like(image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (image, np.zeros_like(image), np.zeros_like(image))
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.ACOT_VLA_PI0 | _model.ModelType.ACOT_VLA_PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (image, np.zeros_like(image), np.zeros_like(image))
                image_masks = (np.True_, np.False_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if actions is not None:
            inputs["actions"] = actions

        if prompt:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class GfootballOutputs(transforms.DataTransformFn):
    """Converts model outputs back to gfootball discrete actions.

    The model predicts action distributions. We convert them to discrete action indices.
    """

    action_dim: int = 19
    action_horizon: int = 1  # Football is single-step action

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 3:
            # (batch, horizon, action_dim) → take first timestep
            actions = actions[:, 0, :]
        if actions.ndim == 2:
            # (horizon, action_dim) → take first timestep or argmax
            if actions.shape[0] > 1:
                actions = actions[0:1, :]
            # Convert from continuous to discrete via argmax
            discrete_actions = np.argmax(actions, axis=-1)
            return {"actions": discrete_actions}
        elif actions.ndim == 1:
            # Already discrete indices
            return {"actions": actions}
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class GfootballMultiAgentInputs(transforms.DataTransformFn):
    """Multi-agent wrapper that processes observations for all agents on one team.

    This is used during training to create batched training samples from
    a single team's worth of observations.
    """

    action_dim: int = 19
    model_type: _model.ModelType = _model.ModelType.PI0_FAST
    n_agents: int = 4  # Number of agents per team (4 for 5v5, 11 for 11v11)
    team_id: int = 0  # 0=left, 1=right

    def __call__(self, data: dict) -> dict:
        """Process multi-agent data.

        Expected input format:
          - "observations": list of per-agent observations (length n_agents)
          - "actions": list of per-agent actions (length n_agents)
          - "task_instruction": optional task description
        """
        observations = data.get("observations", [])
        actions_list = data.get("actions", [])
        task = data.get("task_instruction", "")

        if not observations:
            raise ValueError("No observations provided for multi-agent processing")

        # Process each agent's data
        all_inputs = []
        single_transform = GfootballInputs(
            action_dim=self.action_dim,
            model_type=self.model_type,
        )

        for i, obs in enumerate(observations):
            agent_data = {
                "state": obs,
                "agent_id": self.team_id * 11 + i,
                "team_id": self.team_id,
                "prompt": build_agent_prompt(self.team_id * 11 + i, obs, task),
            }
            # Extract image if available
            if "frame" in obs:
                agent_data["image"] = obs["frame"]
            elif "pixels" in obs:
                agent_data["image"] = obs["pixels"]

            if i < len(actions_list):
                agent_data["actions"] = actions_list[i]

            processed = single_transform(agent_data)
            all_inputs.append(processed)

        # Stack into batch dimension
        batched = {
            "state": np.stack([x["state"] for x in all_inputs]),
            "image": {
                name: np.stack([x["image"][name] for x in all_inputs])
                for name in all_inputs[0]["image"]
            },
            "image_mask": {
                name: np.stack([x["image_mask"][name] for x in all_inputs])
                for name in all_inputs[0]["image_mask"]
            },
            "prompt": [x.get("prompt", "") for x in all_inputs],
        }

        if "actions" in all_inputs[0]:
            batched["actions"] = np.stack([x["actions"] for x in all_inputs])

        return batched


@dataclasses.dataclass(frozen=True)
class ACOTGfootballConfig(_acot_vla.ACOTConfig):
    """ACoT-VLA config specialized for gfootball multi-agent scenarios.

    Extends the base ACOTConfig with gfootball-specific defaults:
      - action_dim=19 (discrete football actions)
      - action_horizon=1 (single-step decisions)
      - coarse_action_horizon=5 (short planning horizon for football)
    """

    action_dim: int = 19
    action_horizon: int = 1
    coarse_action_horizon: int = 5
    pi05: bool = True

    def __post_init__(self):
        # Call parent post_init to set max_token_len and discrete_state_input
        super().__post_init__()

    @override
    def get_freeze_filter(self, **kwargs) -> nnx.filterlib.Filter:
        """Get freeze filter with gfootball-specific defaults."""
        return super().get_freeze_filter(**kwargs)


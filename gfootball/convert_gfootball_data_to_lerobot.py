"""Convert gfootball episode data to LeRobot dataset format for VLA training.

This script collects episodes from the gfootball environment using a specified
policy (random, MAPPO, SAC, or custom) and converts them into the LeRobot
dataset format compatible with the ACoT-VLA training pipeline.

Usage:
    python convert_gfootball_data_to_lerobot.py \
        --output_dir ./dataset/gfootball_lerobot \
        --n_episodes 1000 \
        --scenario football_5v5_malib \
        --policy random \
        --team_id 0

    # With existing MAPPO policy:
    python convert_gfootball_data_to_lerobot.py \
        --output_dir ./dataset/gfootball_lerobot \
        --n_episodes 500 \
        --scenario football_5v5_malib \
        --policy mappo \
        --policy_path /path/to/mappo/checkpoint \
        --team_id 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add gfootball project to path
GFOOTBALL_DIR = "/home/omnisky/Desktop/algos/gfootball"
sys.path.insert(0, GFOOTBALL_DIR)

import gfootball.env as football_env
from env.chooseenv import make


# Football action names
FOOTBALL_ACTIONS = [
    "idle", "left", "top_left", "top", "top_right",
    "right", "bottom_right", "bottom", "bottom_left",
    "long_pass", "high_pass", "short_pass", "shot",
    "sprint", "release_move", "release_sprint",
    "slide", "dribble", "release_dribble",
]

# Player roles
PLAYER_ROLES = {
    0: "goalkeeper", 1: "center_back", 2: "left_back", 3: "right_back",
    4: "defensive_midfielder", 5: "central_midfielder",
    6: "left_midfielder", 7: "right_midfielder",
    8: "attacking_midfielder", 9: "center_forward", 10: "striker",
}


def build_simple115_obs(obs: dict) -> np.ndarray:
    """Build simple115_v2 observation vector from raw gfootball observation."""
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

    # Active player one-hot
    active = obs.get("active", 0)
    active_onehot = np.zeros(11, dtype=np.float32)
    if 0 <= active < 11:
        active_onehot[active] = 1.0
    parts.append(active_onehot)

    # Game mode one-hot
    game_mode = obs.get("game_mode", 0)
    mode_onehot = np.zeros(7, dtype=np.float32)
    if 0 <= game_mode < 7:
        mode_onehot[game_mode] = 1.0
    parts.append(mode_onehot)

    # Score
    score = np.array(obs.get("score", [0, 0]))[:2].astype(np.float32)
    parts.append(score)

    # Steps left (normalized)
    steps_left = np.array([obs.get("steps_left", 3000) / 3000.0], dtype=np.float32)
    parts.append(steps_left)

    return np.concatenate(parts, axis=-1).astype(np.float32)


def get_player_role(obs: dict, player_index: int) -> str:
    """Get the role name for a player."""
    if player_index < 11:
        roles = obs.get("left_team_roles", [])
        local_idx = player_index
    else:
        roles = obs.get("right_team_roles", [])
        local_idx = player_index - 11

    if local_idx < len(roles):
        return PLAYER_ROLES.get(int(roles[local_idx]), "player")
    return "player"


def build_prompt(player_index: int, obs: dict, task_instruction: str = "") -> str:
    """Build natural language prompt for an agent."""
    team = "left" if player_index < 11 else "right"
    local_idx = player_index if player_index < 11 else player_index - 11
    role = get_player_role(obs, player_index)

    if task_instruction:
        return f"{team} team {role} (player {local_idx}): {task_instruction}"

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
        return f"{team} team player {local_idx}: play football"


def render_frame(env_core, obs: dict) -> np.ndarray:
    """Render a game frame from the environment.

    Uses the gfootball SMM (Spatial Map Maker) representation which produces
    a 72x96x4 image. We convert it to RGB.
    """
    try:
        # Try to get rendered pixels from the environment
        frame = env_core.render(mode="rgb_array")
        if frame is not None:
            return frame
    except Exception:
        pass

    # Fallback: create a simple representation from observation
    # Use ball and player positions to create a minimal spatial representation
    h, w = 72, 96
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw field (green background)
    frame[:, :, 1] = 80  # Dark green

    # Draw ball position
    ball = obs.get("ball", [0, 0, 0])
    bx = int((ball[0] + 1) / 2 * (w - 1))
    by = int((ball[1] + 0.42) / 0.84 * (h - 1))
    bx = np.clip(bx, 0, w - 1)
    by = np.clip(by, 0, h - 1)
    frame[max(0, by-2):by+3, max(0, bx-2):bx+3] = [255, 255, 255]  # White ball

    # Draw left team players (blue)
    left_team = obs.get("left_team", [])
    for pos in left_team:
        px = int((pos[0] + 1) / 2 * (w - 1))
        py = int((pos[1] + 0.42) / 0.84 * (h - 1))
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        frame[max(0, py-2):py+3, max(0, px-2):px+3] = [0, 0, 255]

    # Draw right team players (red)
    right_team = obs.get("right_team", [])
    for pos in right_team:
        px = int((pos[0] + 1) / 2 * (w - 1))
        py = int((pos[1] + 0.42) / 0.84 * (h - 1))
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        frame[max(0, py-2):py+3, max(0, px-2):px+3] = [255, 0, 0]

    return frame


class RandomPolicy:
    """Random action policy for data collection."""

    def __init__(self, n_actions: int = 19):
        self.n_actions = n_actions

    def act(self, obs: dict) -> int:
        return np.random.randint(0, self.n_actions)


class HeuristicPolicy:
    """Simple heuristic policy based on ball position and player role."""

    def __init__(self):
        pass

    def act(self, obs: dict) -> int:
        ball = obs.get("ball", [0, 0, 0])
        ball_owned_team = obs.get("ball_owned_team", -1)
        active = obs.get("active", 0)
        roles = obs.get("left_team_roles", [])

        # Get current player role
        role = roles[active] if active < len(roles) else -1

        # Goalkeeper
        if role == 0:
            if ball_owned_team == 0:
                return 11  # short_pass
            return 9  # long_pass (clear)

        # Defenders
        if role in (1, 2, 3):
            if ball_owned_team == 0:
                if ball[0] < -0.3:
                    return 9  # long_pass forward
                return 11  # short_pass
            # Move towards ball
            return self._move_towards_ball(obs)

        # Midfielders
        if role in (4, 5, 6, 7, 8):
            if ball_owned_team == 0:
                if ball[0] > 0.3:
                    return 12  # shot
                return 11  # short_pass
            return self._move_towards_ball(obs)

        # Forwards
        if role in (9, 10):
            if ball_owned_team == 0:
                if ball[0] > 0.5:
                    return 12  # shot
                return 11  # short_pass
            return self._move_towards_ball(obs)

        # Default: move towards ball
        return self._move_towards_ball(obs)

    def _move_towards_ball(self, obs: dict) -> int:
        """Move towards the ball."""
        ball = obs.get("ball", [0, 0, 0])
        active = obs.get("active", 0)
        left_team = obs.get("left_team", [])

        if active < len(left_team):
            player_pos = left_team[active]
            dx = ball[0] - player_pos[0]
            dy = ball[1] - player_pos[1]

            # Map direction to movement action
            angle = np.arctan2(dy, dx)
            if angle < -np.pi * 7/8 or angle >= np.pi * 7/8:
                return 1  # left
            elif angle < -np.pi * 5/8:
                return 2  # top_left
            elif angle < -np.pi * 3/8:
                return 3  # top
            elif angle < -np.pi * 1/8:
                return 4  # top_right
            elif angle < np.pi * 1/8:
                return 5  # right
            elif angle < np.pi * 3/8:
                return 6  # bottom_right
            elif angle < np.pi * 5/8:
                return 7  # bottom
            else:
                return 8  # bottom_left

        return 0  # idle


def collect_episodes(
    scenario: str = "football_5v5_malib",
    n_episodes: int = 100,
    max_steps: int = 3000,
    policy_type: str = "random",
    policy_path: str | None = None,
    team_id: int = 0,
    render: bool = True,
) -> list[dict]:
    """Collect episodes from the gfootball environment.

    Args:
        scenario: Environment scenario name
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        policy_type: Policy type ("random", "heuristic", "mappo", "sac")
        policy_path: Path to policy checkpoint (for mappo/sac)
        team_id: Team to collect data for (0=left, 1=right)
        render: Whether to render frames

    Returns:
        List of episode dictionaries
    """
    # Create environment
    game = make(scenario)
    n_agents = game.agent_nums[team_id]

    # Create policy
    if policy_type == "random":
        policy = RandomPolicy()
    elif policy_type == "heuristic":
        policy = HeuristicPolicy()
    elif policy_type == "mappo":
        # Load MAPPO policy
        if policy_path is None:
            policy_path = os.path.join(GFOOTBALL_DIR, "agents/football_5v5_mappo")
        sys.path.insert(0, policy_path)
        from submission import my_controller as mappo_controller
        policy = mappo_controller
    else:
        policy = RandomPolicy()

    episodes = []

    for ep_idx in tqdm(range(n_episodes), desc="Collecting episodes"):
        obs_list = game.reset()
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "frames": [],
        }

        for step in range(max_steps):
            # Get team observations
            if team_id == 0:
                team_obs = obs_list[:n_agents]
            else:
                team_obs = obs_list[n_agents:]

            # Render frame if needed
            if render:
                frame = render_frame(game.env_core, team_obs[0]["obs"])
                episode_data["frames"].append(frame)

            # Collect actions for each agent
            team_actions = []
            for agent_idx in range(n_agents):
                obs = team_obs[agent_idx]
                raw_obs = obs.get("obs", obs)

                if policy_type in ("random", "heuristic"):
                    action = policy.act(raw_obs)
                elif policy_type == "mappo":
                    # MAPPO expects specific input format
                    action = policy(obs, None)  # Simplified call
                    if isinstance(action, (list, np.ndarray)):
                        action = int(action[0]) if len(action) > 0 else 0
                else:
                    action = np.random.randint(0, 19)

                team_actions.append(action)

                # Store per-agent data
                episode_data["observations"].append({
                    "obs": raw_obs,
                    "state": build_simple115_obs(raw_obs),
                    "agent_id": team_id * 11 + agent_idx,
                    "player_role": get_player_role(raw_obs, team_id * 11 + agent_idx),
                    "prompt": build_prompt(team_id * 11 + agent_idx, raw_obs),
                })
                episode_data["actions"].append(team_actions[-1])

            # Step environment
            # Build joint action for all players
            joint_action = []
            for i in range(game.n_player):
                one_hot = [0] * 19
                if team_id == 0 and i < n_agents:
                    one_hot[team_actions[i]] = 1
                elif team_id == 1 and i >= game.agent_nums[0] and i < game.agent_nums[0] + n_agents:
                    one_hot[team_actions[i - game.agent_nums[0]]] = 1
                else:
                    # Other team: random or heuristic
                    one_hot[np.random.randint(0, 19)] = 1
                joint_action.append([one_hot])

            obs_list, reward, done, info_before, info_after = game.step(joint_action)

            # Store reward
            if team_id == 0:
                episode_data["rewards"].append(reward[0])
            else:
                episode_data["rewards"].append(reward[game.agent_nums[0]])

            if done:
                break

        episodes.append(episode_data)

    return episodes


def save_lerobot_dataset(episodes: list[dict], output_dir: str):
    """Save collected episodes in LeRobot-compatible format.

    Creates:
      - meta/info.json: dataset metadata
      - meta/episodes.jsonl: episode metadata
      - meta/episodes_stats.jsonl: per-episode statistics
      - meta/tasks.jsonl: task descriptions
      - data/chunk-000/episode_XXXXX.hdf5: episode data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    (output_dir / "meta").mkdir(exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Collect all actions and states for normalization stats
    all_states = []
    all_actions = []
    for ep in episodes:
        for obs_data in ep["observations"]:
            all_states.append(obs_data["state"])
        all_actions.extend(ep["actions"])

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)

    # Compute normalization stats
    state_mean = all_states.mean(axis=0).tolist()
    state_std = all_states.std(axis=0).tolist()
    action_mean = float(all_actions.mean())
    action_std = float(all_actions.std())

    # Write info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "gfootball",
        "total_episodes": len(episodes),
        "total_frames": sum(len(ep["actions"]) for ep in episodes),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": len(episodes),
        "fps": 10,
        "splits": {
            "train": f"0:{len(episodes)}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [115],
                "names": None,
            },
            "observation.images.image_base": {
                "dtype": "video",
                "shape": [72, 96, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "int64",
                "shape": [1],
                "names": FOOTBALL_ACTIONS,
            },
            "prompt": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
        },
    }

    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Write episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx, ep in enumerate(episodes):
            ep_meta = {
                "episode_index": ep_idx,
                "tasks": ["play football"],
                "length": len(ep["actions"]),
            }
            f.write(json.dumps(ep_meta) + "\n")

    # Write tasks.jsonl
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "play football"}) + "\n")

    # Write episodes_stats.jsonl
    with open(output_dir / "meta" / "episodes_stats.jsonl", "w") as f:
        for ep_idx, ep in enumerate(episodes):
            ep_states = np.array([o["state"] for o in ep["observations"]])
            ep_actions = np.array(ep["actions"])
            stats = {
                "episode_index": ep_idx,
                "stats": {
                    "observation.state": {
                        "mean": ep_states.mean(axis=0).tolist(),
                        "std": ep_states.std(axis=0).tolist(),
                        "min": ep_states.min(axis=0).tolist(),
                        "max": ep_states.max(axis=0).tolist(),
                    },
                    "action": {
                        "mean": [float(ep_actions.mean())],
                        "std": [float(ep_actions.std())],
                        "min": [float(ep_actions.min())],
                        "max": [float(ep_actions.max())],
                    },
                },
            }
            f.write(json.dumps(stats) + "\n")

    # Write norm_stats.json (for compatibility with ACoT-VLA pipeline)
    norm_stats = {
        "observation.state": {
            "mean": state_mean,
            "std": state_std,
        },
        "action": {
            "mean": [action_mean],
            "std": [action_std],
        },
    }
    assets_dir = output_dir.parent.parent / "assets" / "gfootball_train" / "gfootball_dataset"
    assets_dir.mkdir(parents=True, exist_ok=True)
    with open(assets_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Save episodes as HDF5
    try:
        import h5py
        for ep_idx, ep in enumerate(tqdm(episodes, desc="Saving episodes")):
            filepath = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.hdf5"
            with h5py.File(filepath, "w") as hf:
                # States
                states = np.array([o["state"] for o in ep["observations"]])
                hf.create_dataset("observation.state", data=states, compression="gzip")

                # Actions
                actions = np.array(ep["actions"])
                hf.create_dataset("action", data=actions, compression="gzip")

                # Frames (if available)
                if ep["frames"]:
                    frames = np.array(ep["frames"])
                    hf.create_dataset("observation.images.image_base", data=frames, compression="gzip")

                # Prompts
                prompts = [o["prompt"] for o in ep["observations"]]
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset("prompt", data=prompts, dtype=dt)

                # Rewards
                rewards = np.array(ep["rewards"])
                hf.create_dataset("reward", data=rewards, compression="gzip")

                # Metadata
                hf.attrs["episode_index"] = ep_idx
                hf.attrs["length"] = len(ep["actions"])

    except ImportError:
        print("Warning: h5py not installed. Saving as numpy arrays instead.")
        for ep_idx, ep in enumerate(tqdm(episodes, desc="Saving episodes")):
            filepath = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.npz"
            np.savez_compressed(
                filepath,
                observation_state=np.array([o["state"] for o in ep["observations"]]),
                action=np.array(ep["actions"]),
                reward=np.array(ep["rewards"]),
            )

    print(f"Saved {len(episodes)} episodes to {output_dir}")
    print(f"  Total frames: {sum(len(ep['actions']) for ep in episodes)}")
    print(f"  Norm stats saved to {assets_dir / 'norm_stats.json'}")


def main():
    parser = argparse.ArgumentParser(description="Convert gfootball data to LeRobot format")
    parser.add_argument("--output_dir", type=str, default="./dataset/gfootball_lerobot",
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode")
    parser.add_argument("--scenario", type=str, default="football_5v5_malib",
                        choices=["football_5v5_malib", "football_11_vs_11_stochastic"],
                        help="Environment scenario")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "heuristic", "mappo"],
                        help="Policy type for data collection")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Path to policy checkpoint")
    parser.add_argument("--team_id", type=int, default=0,
                        help="Team to collect data for (0=left, 1=right)")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable frame rendering")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Collecting {args.n_episodes} episodes from {args.scenario}...")
    print(f"  Policy: {args.policy}")
    print(f"  Team: {'left' if args.team_id == 0 else 'right'}")

    episodes = collect_episodes(
        scenario=args.scenario,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        policy_type=args.policy,
        policy_path=args.policy_path,
        team_id=args.team_id,
        render=not args.no_render,
    )

    save_lerobot_dataset(episodes, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

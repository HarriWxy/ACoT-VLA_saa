"""Multi-agent VLA inference for gfootball.

This script loads a trained VLA model and runs multi-agent inference
in the gfootball environment. Each agent independently queries the
VLA model with its observation and receives a discrete action.

Usage:
    # Run with default checkpoint:
    python serve_gfootball.py --config pi0_fast_gfootball_5v5 \
        --checkpoint_dir ./checkpoints/pi0_fast_gfootball_5v5/exp_name/30000

    # Run with rendering:
    python serve_gfootball.py --config pi0_fast_gfootball_5v5 \
        --checkpoint_dir ./checkpoints/pi0_fast_gfootball_5v5/exp_name/30000 \
        --render

    # Run against heuristic baseline:
    python serve_gfootball.py --config pi0_fast_gfootball_5v5 \
        --checkpoint_dir ./checkpoints/pi0_fast_gfootball_5v5/exp_name/30000 \
        --opponent_policy heuristic

    # Run as WebSocket server for remote inference:
    python serve_gfootball.py --config pi0_fast_gfootball_5v5 \
        --checkpoint_dir ./checkpoints/pi0_fast_gfootball_5v5/exp_name/30000 \
        --serve --port 8000
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add gfootball project to path
GFOOTBALL_DIR = "/home/omnisky/Desktop/algos/gfootball"
sys.path.insert(0, GFOOTBALL_DIR)

import gfootball.env as football_env
from env.chooseenv import make

# Add ACoT-VLA project to path
ACOT_VLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACOT_VLA_DIR)

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import gfootball_policy
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultiAgentVLAController:
    """Multi-agent controller using VLA model for gfootball.

    Each agent on the controlled team independently queries the VLA model
    with its observation and receives a discrete action.
    """

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        n_agents: int = 4,
        team_id: int = 0,
        default_prompt: str | None = None,
    ):
        """Initialize the multi-agent VLA controller.

        Args:
            config_name: Training config name (e.g., "pi0_fast_gfootball_5v5")
            checkpoint_dir: Path to trained checkpoint
            n_agents: Number of agents on the controlled team
            team_id: 0=left team, 1=right team
            default_prompt: Default prompt for agents
        """
        self.n_agents = n_agents
        self.team_id = team_id
        self.default_prompt = default_prompt

        # Load config and create policy
        logger.info(f"Loading config: {config_name}")
        config = _config.get_config(config_name)

        logger.info(f"Loading checkpoint from: {checkpoint_dir}")
        self.policy = _policy_config.create_trained_policy(
            config, checkpoint_dir, default_prompt=default_prompt
        )

        logger.info(f"VLA controller initialized for {n_agents} agents (team {team_id})")

    def get_actions(
        self,
        observations: list[dict],
        task_instruction: str = "",
    ) -> list[int]:
        """Get actions for all agents on the controlled team.

        Args:
            observations: List of per-agent raw observations
            task_instruction: Optional task description

        Returns:
            List of discrete action indices (0-18) for each agent
        """
        actions = []

        for agent_idx in range(self.n_agents):
            obs = observations[agent_idx]
            raw_obs = obs.get("obs", obs)

            # Build agent data for VLA model
            agent_id = self.team_id * 11 + agent_idx
            prompt = gfootball_policy.build_agent_prompt(
                agent_id, raw_obs, task_instruction
            )

            # Build simple115 state vector
            state = gfootball_policy._build_simple115_obs(raw_obs)

            # Get image (use rendered frame if available, otherwise build from obs)
            image = self._get_image(raw_obs)

            # Prepare model input
            data = {
                "state": state,
                "image": image,
                "prompt": prompt,
                "agent_id": agent_id,
                "team_id": self.team_id,
            }

            # Query VLA model
            result = self.policy.infer(data)
            action = result["actions"]

            # Convert to discrete index
            if isinstance(action, np.ndarray):
                if action.ndim > 1:
                    action = action[0]
                if action.ndim == 1 and action.shape[0] > 1:
                    # One-hot or continuous → argmax
                    action = int(np.argmax(action))
                else:
                    action = int(action)
            elif isinstance(action, (list, tuple)):
                action = int(action[0])

            actions.append(action)

        return actions

    def _get_image(self, obs: dict) -> np.ndarray:
        """Get or render image from observation."""
        # Try to get rendered pixels
        if "frame" in obs:
            return gfootball_policy._parse_image(obs["frame"])
        if "pixels" in obs:
            return gfootball_policy._parse_image(obs["pixels"])

        # Build minimal spatial representation from observation
        h, w = 72, 96
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Field background (green)
        frame[:, :, 1] = 80

        # Ball (white)
        ball = obs.get("ball", [0, 0, 0])
        bx = int((ball[0] + 1) / 2 * (w - 1))
        by = int((ball[1] + 0.42) / 0.84 * (h - 1))
        bx, by = np.clip(bx, 0, w-1), np.clip(by, 0, h-1)
        frame[max(0, by-2):by+3, max(0, bx-2):bx+3] = [255, 255, 255]

        # Left team (blue)
        for pos in obs.get("left_team", []):
            px = int((pos[0] + 1) / 2 * (w - 1))
            py = int((pos[1] + 0.42) / 0.84 * (h - 1))
            px, py = np.clip(px, 0, w-1), np.clip(py, 0, h-1)
            frame[max(0, py-2):py+3, max(0, px-2):px+3] = [0, 0, 255]

        # Right team (red)
        for pos in obs.get("right_team", []):
            px = int((pos[0] + 1) / 2 * (w - 1))
            py = int((pos[1] + 0.42) / 0.84 * (h - 1))
            px, py = np.clip(px, 0, w-1), np.clip(py, 0, h-1)
            frame[max(0, py-2):py+3, max(0, px-2):px+3] = [255, 0, 0]

        return frame


class HeuristicPolicy:
    """Simple heuristic policy for the opponent team."""

    def __init__(self, n_agents: int = 4):
        self.n_agents = n_agents

    def get_actions(self, observations: list[dict]) -> list[int]:
        actions = []
        for obs in observations:
            raw_obs = obs.get("obs", obs)
            actions.append(self._act(raw_obs))
        return actions

    def _act(self, obs: dict) -> int:
        ball = obs.get("ball", [0, 0, 0])
        ball_owned_team = obs.get("ball_owned_team", -1)
        active = obs.get("active", 0)
        roles = obs.get("right_team_roles", obs.get("left_team_roles", []))
        role = roles[active] if active < len(roles) else -1

        if role == 0:  # GK
            return 9 if ball_owned_team != 0 else 11
        if role in (1, 2, 3):  # DEF
            return 11 if ball_owned_team == 0 else self._move_to_ball(obs)
        if role in (4, 5, 6, 7, 8):  # MID
            if ball_owned_team == 0:
                return 12 if ball[0] > 0.3 else 11
            return self._move_to_ball(obs)
        if role in (9, 10):  # FWD
            if ball_owned_team == 0:
                return 12 if ball[0] > 0.5 else 11
            return self._move_to_ball(obs)
        return self._move_to_ball(obs)

    def _move_to_ball(self, obs: dict) -> int:
        ball = obs.get("ball", [0, 0, 0])
        active = obs.get("active", 0)
        team = obs.get("right_team", obs.get("left_team", []))
        if active < len(team):
            dx = ball[0] - team[active][0]
            dy = ball[1] - team[active][1]
            angle = np.arctan2(dy, dx)
            if angle < -np.pi * 7/8 or angle >= np.pi * 7/8:
                return 1
            elif angle < -np.pi * 5/8:
                return 2
            elif angle < -np.pi * 3/8:
                return 3
            elif angle < -np.pi * 1/8:
                return 4
            elif angle < np.pi * 1/8:
                return 5
            elif angle < np.pi * 3/8:
                return 6
            elif angle < np.pi * 5/8:
                return 7
            else:
                return 8
        return 0


class RandomPolicy:
    """Random action policy."""

    def __init__(self, n_agents: int = 4):
        self.n_agents = n_agents

    def get_actions(self, observations: list[dict]) -> list[int]:
        return [np.random.randint(0, 19) for _ in range(self.n_agents)]


def run_match(
    vla_controller: MultiAgentVLAController,
    opponent_policy: Any,
    scenario: str = "football_5v5_malib",
    n_episodes: int = 10,
    max_steps: int = 3000,
    render: bool = False,
    vla_team: int = 0,
) -> dict:
    """Run a match between VLA team and opponent team.

    Args:
        vla_controller: VLA controller for the controlled team
        opponent_policy: Policy for the opponent team
        scenario: Environment scenario
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render
        vla_team: Which team the VLA controls (0=left, 1=right)

    Returns:
        Dictionary with match statistics
    """
    game = make(scenario)
    n_agents_per_team = game.agent_nums[0]

    stats = {
        "vla_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "vla_goals": 0,
        "opponent_goals": 0,
        "episode_rewards": [],
        "episode_lengths": [],
    }

    for ep_idx in range(n_episodes):
        obs_list = game.reset()
        ep_reward_vla = 0.0
        ep_reward_opp = 0.0

        for step in range(max_steps):
            # Split observations by team
            left_obs = obs_list[:n_agents_per_team]
            right_obs = obs_list[n_agents_per_team:]

            # Get VLA team actions
            if vla_team == 0:
                vla_actions = vla_controller.get_actions(left_obs)
            else:
                vla_actions = vla_controller.get_actions(right_obs)

            # Get opponent team actions
            if vla_team == 0:
                opp_actions = opponent_policy.get_actions(right_obs)
            else:
                opp_actions = opponent_policy.get_actions(left_obs)

            # Build joint action
            joint_action = []
            for i in range(game.n_player):
                one_hot = [0] * 19
                if i < n_agents_per_team:
                    if vla_team == 0:
                        one_hot[vla_actions[i]] = 1
                    else:
                        one_hot[opp_actions[i]] = 1
                else:
                    agent_idx = i - n_agents_per_team
                    if vla_team == 1:
                        one_hot[vla_actions[agent_idx]] = 1
                    else:
                        one_hot[opp_actions[agent_idx]] = 1
                joint_action.append([one_hot])

            # Step environment
            obs_list, reward, done, info_before, info_after = game.step(joint_action)

            # Track rewards
            ep_reward_vla += reward[vla_team * n_agents_per_team]
            ep_reward_opp += reward[(1 - vla_team) * n_agents_per_team]

            if done:
                break

        # Record episode stats
        stats["episode_rewards"].append(ep_reward_vla)
        stats["episode_lengths"].append(step + 1)

        # Determine winner
        if ep_reward_vla > ep_reward_opp:
            stats["vla_wins"] += 1
        elif ep_reward_vla < ep_reward_opp:
            stats["opponent_wins"] += 1
        else:
            stats["draws"] += 1

        stats["vla_goals"] += max(0, int(ep_reward_vla))
        stats["opponent_goals"] += max(0, int(ep_reward_opp))

        logger.info(
            f"Episode {ep_idx + 1}/{n_episodes}: "
            f"VLA reward={ep_reward_vla:.1f}, Opp reward={ep_reward_opp:.1f}, "
            f"Steps={step + 1}"
        )

    # Summary
    logger.info("=" * 60)
    logger.info(f"Match Results ({n_episodes} episodes):")
    logger.info(f"  VLA Wins: {stats['vla_wins']}")
    logger.info(f"  Opponent Wins: {stats['opponent_wins']}")
    logger.info(f"  Draws: {stats['draws']}")
    logger.info(f"  VLA Goals: {stats['vla_goals']}")
    logger.info(f"  Opponent Goals: {stats['opponent_goals']}")
    logger.info(f"  Win Rate: {stats['vla_wins']/n_episodes*100:.1f}%")
    logger.info(f"  Avg Reward: {np.mean(stats['episode_rewards']):.2f}")
    logger.info(f"  Avg Episode Length: {np.mean(stats['episode_lengths']):.1f}")
    logger.info("=" * 60)

    return stats


def serve_websocket(
    vla_controller: MultiAgentVLAController,
    port: int = 8000,
):
    """Start a WebSocket server for remote inference.

    This allows the gfootball environment to connect remotely
    and query the VLA model for actions.
    """
    try:
        import websockets
        import asyncio
    except ImportError:
        logger.error("websockets package required for server mode. Install with: pip install websockets")
        return

    async def handle_client(websocket, path):
        logger.info(f"Client connected from {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)

                if data.get("type") == "get_actions":
                    observations = data["observations"]
                    task = data.get("task_instruction", "")
                    actions = vla_controller.get_actions(observations, task)
                    response = {"type": "actions", "actions": actions}
                    await websocket.send(json.dumps(response))

                elif data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    async def main():
        server = await websockets.serve(handle_client, "0.0.0.0", port)
        logger.info(f"WebSocket server started on ws://0.0.0.0:{port}")
        await server.wait_closed()

    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VLA inference for gfootball")
    parser.add_argument("--config", type=str, default="pi0_fast_gfootball_5v5",
                        help="Training config name")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--scenario", type=str, default="football_5v5_malib",
                        choices=["football_5v5_malib", "football_11_vs_11_stochastic"],
                        help="Environment scenario")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode")
    parser.add_argument("--vla_team", type=int, default=0,
                        help="Team controlled by VLA (0=left, 1=right)")
    parser.add_argument("--opponent_policy", type=str, default="heuristic",
                        choices=["random", "heuristic"],
                        help="Opponent policy type")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--serve", action="store_true",
                        help="Run as WebSocket server")
    parser.add_argument("--port", type=int, default=8000,
                        help="WebSocket server port")
    parser.add_argument("--default_prompt", type=str, default=None,
                        help="Default prompt for all agents")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    # Determine number of agents
    if "5v5" in args.scenario:
        n_agents = 4
    else:
        n_agents = 11

    # Initialize VLA controller
    vla_controller = MultiAgentVLAController(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        n_agents=n_agents,
        team_id=args.vla_team,
        default_prompt=args.default_prompt,
    )

    if args.serve:
        # Run as WebSocket server
        serve_websocket(vla_controller, args.port)
    else:
        # Run match
        if args.opponent_policy == "random":
            opponent = RandomPolicy(n_agents)
        else:
            opponent = HeuristicPolicy(n_agents)

        stats = run_match(
            vla_controller=vla_controller,
            opponent_policy=opponent,
            scenario=args.scenario,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            render=args.render,
            vla_team=args.vla_team,
        )

        # Save results
        if args.save_results:
            with open(args.save_results, "w") as f:
                # Convert numpy values for JSON serialization
                serializable_stats = {
                    k: v if not isinstance(v, list) else
                    [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                    for k, v in stats.items()
                }
                json.dump(serializable_stats, f, indent=2)
            logger.info(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()

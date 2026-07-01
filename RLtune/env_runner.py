"""Environment rollout runner for RL training.

Collects trajectories by running the VLA policy in SRB environments.
Uses the existing WebSocket client-server architecture for policy inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .grpo_algo import TrajectoryData
from .rl_config import EnvConfig
from .rl_config import GRPOConfig

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor/array to numpy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_request(obs: dict, prompt: str) -> dict:
    """Prepare observation dict for policy server."""
    request = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            request[key] = {k: _to_numpy(v) for k, v in value.items()}
        else:
            request[key] = _to_numpy(value)
    request["prompt"] = prompt
    return request


class EnvRunner:
    """Runs episodes in SRB environments and collects trajectories.

    This class manages environment creation, stepping, and trajectory collection.
    It communicates with the VLA policy via the WebSocket client.

    Usage:
        runner = EnvRunner(config, env_config)
        trajectories = runner.collect_rollouts(policy_host, policy_port, n_episodes=8)
    """

    def __init__(self, config: GRPOConfig, env_config: EnvConfig | None = None):
        self.config = config
        self.env_config = env_config or EnvConfig(
            env_id=config.env_id,
            prompt=config.default_prompt,
            max_steps=config.max_episode_steps,
            replan_steps=config.replan_steps,
        )

    def collect_rollouts(
        self,
        policy_host: str = "0.0.0.0",
        policy_port: int = 8899,
        n_episodes: int | None = None,
        prompt: str | None = None,
        task_variations: list[str] | None = None,
    ) -> list[TrajectoryData]:
        """Collect rollout trajectories using the served policy.

        Args:
            policy_host: Host address of the policy server.
            policy_port: Port of the policy server.
            n_episodes: Number of episodes to collect (default: config.n_samples).
            prompt: Task prompt override.
            task_variations: Optional list of task prompts for diverse sampling.

        Returns:
            List of TrajectoryData objects.
        """
        n_episodes = n_episodes or self.config.n_samples
        prompt = prompt or self.env_config.prompt

        trajectories = []
        for ep_idx in range(n_episodes):
            ep_prompt = prompt
            if task_variations and len(task_variations) > 0:
                ep_prompt = task_variations[ep_idx % len(task_variations)]

            traj = self._run_single_episode(
                policy_host=policy_host,
                policy_port=policy_port,
                prompt=ep_prompt,
                episode_idx=ep_idx,
                seed=self.env_config.seed + ep_idx,
            )
            trajectories.append(traj)

        return trajectories

    def _run_single_episode(
        self,
        policy_host: str,
        policy_port: int,
        prompt: str,
        episode_idx: int,
        seed: int = 0,
    ) -> TrajectoryData:
        """Run a single episode and collect the trajectory.

        This method creates an SRB environment, runs the policy, and returns
        the trajectory data.
        """
        from openpi_client import websocket_client_policy as _websocket_client

        try:
            # Create environment
            env, _task_description = self._create_env(seed)

            # Connect to policy server
            client = _websocket_client.WebsocketClientPolicy(policy_host, policy_port)

            # Reset environment
            obs, info = env.reset(seed=seed)
            if isinstance(obs, tuple):
                obs, info = obs

            total_reward = 0.0
            success = False
            step_count = 0
            all_obs = []
            all_actions = []

            for step in range(self.env_config.max_steps):
                # Prepare and send observation to policy
                request = _prepare_request(obs, prompt)
                result = client.infer(request)

                # Extract actions
                actions = result.get("actions", result.get("action"))
                if actions is None:
                    logger.warning(f"Episode {episode_idx}: No actions returned by policy at step {step}")
                    break

                actions = _to_numpy(actions)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)

                # Execute actions in environment
                # Use replan_steps: execute first N actions, then replan
                n_exec = min(self.env_config.replan_steps, actions.shape[0])
                for a_idx in range(n_exec):
                    action = actions[a_idx]
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result

                    all_obs.append(obs)
                    all_actions.append(action)
                    total_reward += reward
                    step_count += 1

                    if done:
                        break

                if done:
                    # Check success from info
                    success = bool(info.get("success", info.get("is_success", reward > 0)))
                    break

            # If episode ended without done, check final info
            if not done and step_count >= self.env_config.max_steps:
                success = bool(info.get("success", info.get("is_success", False)))

            # Close environment
            env.close()

            return TrajectoryData(
                observation=all_obs,
                actions=np.array(all_actions) if all_actions else np.zeros((1, self.config.action_dim)),
                reward=1.0 if success else 0.0,
                success=success,
                episode_length=step_count,
                prompt_index=hash(prompt) % (2**31),
            )

        except Exception as e:
            logger.error(f"Episode {episode_idx} failed: {e}")
            return TrajectoryData(
                observation=[],
                actions=np.zeros((1, self.config.action_dim)),
                reward=0.0,
                success=False,
                episode_length=0,
                prompt_index=hash(prompt) % (2**31),
            )

    def _create_env(self, seed: int = 0):
        """Create and return an SRB environment.

        Returns:
            env: Gymnasium environment instance.
            task_description: Natural language task description.
        """
        try:
            from srb.core.app import AppLauncher
            from srb.utils.cache import update_offline_srb_cache
            from srb.utils.cfg import load_cfg_from_registry
            from srb.utils.path import SRB_APPS_DIR

            enable_cameras = self.env_config.enable_cameras or self.env_config.env_id.endswith("_visual")
            experience = SRB_APPS_DIR.joinpath(
                f"srb.{'headless.' if self.env_config.headless else ''}{'rendering.' if enable_cameras else ''}kit"
            )

            launcher = AppLauncher(
                headless=self.env_config.headless,
                enable_cameras=enable_cameras,
                experience=experience.as_posix(),
            )

            import gymnasium
            from srb import tasks as _  # noqa: F401

            update_offline_srb_cache()

            env_cfg = load_cfg_from_registry(self.env_config.env_id, "task_cfg")
            env_cfg.seed = seed
            env_cfg.num_envs = 1
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = self.env_config.device

            env = gymnasium.make(self.env_config.env_id, cfg=env_cfg)
            task_description = self.env_config.prompt

            return env, task_description

        except ImportError as e:
            raise ImportError(
                "SRB environment not found. Install with: pip install srb\n"
                f"Original error: {e}"
            )


class ParallelEnvRunner:
    """Parallel environment runner using multiprocessing.

    Collects rollouts from multiple environment instances simultaneously.
    Each worker runs its own environment and communicates results back.
    """

    def __init__(self, config: GRPOConfig, num_workers: int | None = None):
        self.config = config
        self.num_workers = num_workers or config.num_workers

    def collect_rollouts(
        self,
        policy_host: str = "0.0.0.0",
        policy_port: int = 8899,
        n_episodes: int | None = None,
        prompt: str | None = None,
    ) -> list[TrajectoryData]:
        """Collect rollouts in parallel across multiple workers.

        Note: This assumes the policy server can handle concurrent connections.
        For single-GPU setups, sequential collection (EnvRunner) is recommended.
        """
        n_episodes = n_episodes or self.config.n_samples
        episodes_per_worker = n_episodes // self.num_workers
        remainder = n_episodes % self.num_workers

        # Distribute episodes across workers
        worker_episodes = [episodes_per_worker] * self.num_workers
        for i in range(remainder):
            worker_episodes[i] += 1

        # For now, fall back to sequential execution
        # (Parallel env creation with SRB is complex due to GPU context)
        runner = EnvRunner(self.config)
        return runner.collect_rollouts(
            policy_host=policy_host,
            policy_port=policy_port,
            n_episodes=n_episodes,
            prompt=prompt,
        )


def compute_rollout_metrics(trajectories: list[TrajectoryData]) -> dict[str, float]:
    """Compute metrics from a batch of trajectories.

    Args:
        trajectories: List of collected trajectories.

    Returns:
        Dictionary of metrics.
    """
    if not trajectories:
        return {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "mean_episode_length": 0.0,
            "num_episodes": 0.0,
        }

    rewards = [t.reward for t in trajectories]
    successes = [t.success for t in trajectories]
    lengths = [t.episode_length for t in trajectories]

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "max_episode_length": float(np.max(lengths)) if lengths else 0.0,
        "min_episode_length": float(np.min(lengths)) if lengths else 0.0,
        "num_episodes": float(len(trajectories)),
    }

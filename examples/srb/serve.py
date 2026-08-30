import dataclasses
import logging
import os
from typing import Literal

from openpi_client import base_policy as _base_policy
import tyro

from openpi.policies import debug_policy as _debug_policy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SAMPLE = True  # True=使用 websocket_policy_server_sample, False=使用 websocket_policy_server

if SAMPLE:
    from openpi.serving import websocket_policy_server_sample as wps_sample
else:
    from openpi.serving import websocket_policy_server as wps_sample


@dataclasses.dataclass
class Args:
    policy_mode: Literal["model", "random", "zero"] = "random"
    config_name: str = "physics_aware_srb_train_tracking"
    checkpoint_dir: str = "checkpoints/physics_aware_srb_train_tracking/srb_physics_aware/rl_grpo_offline_jax/80"
    action_horizon: int = 8
    action_dim: int = 19
    random_seed: int = 0
    random_action_scale: float = 0.25
    host: str = "0.0.0.0"  # ""127.168.1.116
    port: int = 8899
    default_prompt: str | None = None

    # ── 探索噪声参数 ──
    # 噪声模式: none / output / initial / both
    exploration_mode: str = "none"
    exploration_noise_std: float = 0.05  # 输出噪声标准差
    ou_theta: float = 0.15  # OU 均值回归速度
    ou_sigma: float = 0.3  # OU 噪声强度
    initial_noise_scale: float = 1.0  # 初始噪声缩放因子
    num_steps: int | None = None  # 流匹配去噪步数, None=模型默认


def _create_policy(args: Args) -> _base_policy.BasePolicy:
    if args.policy_mode in ("random", "zero"):
        return _debug_policy.DebugChunkPolicy(
            _debug_policy.DebugPolicyConfig(
                action_horizon=args.action_horizon,
                action_dim=args.action_dim,
                mode=args.policy_mode,
                seed=args.random_seed,
                action_scale=args.random_action_scale,
            )
        )

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)
    return _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )


def main(args: Args) -> None:
    policy = _create_policy(args)

    if SAMPLE:
        config = wps_sample.ServerConfig(
            policy=policy,
            host=args.host,
            port=args.port,
            action_dim=args.action_dim,
            exploration_mode=args.exploration_mode,
            exploration_noise_std=args.exploration_noise_std,
            ou_theta=args.ou_theta,
            ou_sigma=args.ou_sigma,
            initial_noise_scale=args.initial_noise_scale,
            num_steps=args.num_steps,
            model_action_horizon=args.action_horizon,
            model_action_dim=args.action_dim,
        )

        server = wps_sample.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
            config=config,
        )
    else:
        server = wps_sample.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

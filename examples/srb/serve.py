import dataclasses
import logging
from typing import Literal

import tyro

from openpi_client import base_policy as _base_policy
from openpi.policies import debug_policy as _debug_policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    policy_mode: Literal["model", "random", "zero"] = "model"
    config_name: str = "pi05_srb"
    checkpoint_dir: str = ".cache/openpi/openpi-assets/checkpoints/pi05_base"
    action_horizon: int = 16
    action_dim: int = 8
    random_seed: int = 0
    random_action_scale: float = 0.25
    host: str = "0.0.0.0" # ""127.168.1.116
    port: int = 8899
    default_prompt: str | None = None


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

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
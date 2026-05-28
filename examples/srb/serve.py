import dataclasses
import logging

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_srb"
    checkpoint_dir: str = ".cache/openpi/openpi-assets/checkpoints/pi05_base"
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str | None = None


def main(args: Args) -> None:
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

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
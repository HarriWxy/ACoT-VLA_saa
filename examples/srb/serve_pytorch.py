import dataclasses
import logging
import pathlib
from typing import Literal

import tyro

from openpi_client import base_policy as _base_policy
from openpi.policies import debug_policy as _debug_policy
# from openpi.serving import websocket_policy_server

SAMPLE = False  # True=使用 websocket_policy_server_sample, False=使用 websocket_policy_server

if SAMPLE:
    from openpi.serving import websocket_policy_server_sample as wps_sample
else:
    from openpi.serving import websocket_policy_server as wps_sample

import os
os.environ["OPENPI_DISABLE_COMPILE"] = "1"  # 禁用 torch.compile, 避免与调试器冲突

@dataclasses.dataclass
class Args:
    policy_mode: Literal["model", "random", "zero"] = "model"
    # PyTorch 模型专用配置
    config_name: str = "srb_train_gemma4"
    checkpoint_dir: str = "checkpoints/gemma4"
    # PyTorch 推理设备: "cpu", "cuda", "cuda:0", "cuda:1" 等
    pytorch_device: str | None = None  # None = 自动选择 (有 GPU 用 cuda, 否则 cpu)
    action_horizon: int = 16
    action_dim: int = 37
    random_seed: int = 0
    random_action_scale: float = 0.25
    host: str = "0.0.0.0" # ""127.168.1.116
    port: int = 8899
    default_prompt: str | None = None
    
    # ── 探索噪声参数 ──
    # 噪声模式: none / output / initial / both
    exploration_mode: str = "none"
    exploration_noise_std: float = 0.05   # 输出噪声标准差
    ou_theta: float = 0.15                # OU 均值回归速度
    ou_sigma: float = 0.3                 # OU 噪声强度
    initial_noise_scale: float = 1.0      # 初始噪声缩放因子
    num_steps: int | None = None          # 流匹配去噪步数, None=模型默认

    # ── ActionProjectionHead 配置 ──
    # 当 checkpoint 训练时 action_dim 与实际需要的 action_dim 不同时使用
    # 例如: checkpoint 训练时 action_dim=32, 但需要输出 37 维动作
    use_action_proj_head: bool = True     # 是否启用 ActionProjectionHead
    action_proj_head_path: str | None = None  # projection head 权重路径, None=不加载
    model_action_dim: int = 32             # 模型内部 action_dim (checkpoint 训练时的维度)


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
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)

    # 检查 checkpoint 中是否有 model.safetensors (PyTorch 格式)
    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    weight_path = checkpoint_path / "model.safetensors"
    if weight_path.exists():
        logging.info(f"检测到 PyTorch checkpoint: {weight_path}")
    else:
        logging.warning(f"未在 {checkpoint_path} 找到 model.safetensors, 将尝试 JAX 加载")

    # ── 加载 norm_stats ──
    # PyTorch checkpoint 目录下通常没有 assets/ 子目录,
    # 需要从 assets_base_dir/<config_name>/<repo_id>/ 手动加载.
    norm_stats = None
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is not None:
        # 优先从 checkpoint 目录加载 (如果存在)
        ckpt_assets_dir = checkpoint_path / "assets"
        if (ckpt_assets_dir / data_config.asset_id / "norm_stats.json").exists():
            norm_stats = _normalize.load(ckpt_assets_dir / data_config.asset_id)
            logging.info(f"从 checkpoint 目录加载 norm_stats: {ckpt_assets_dir / data_config.asset_id}")
        else:
            # 回退到项目 assets 目录
            project_assets_dir = train_config.assets_dirs
            try:
                norm_stats = _normalize.load(project_assets_dir / data_config.asset_id)
                logging.info(f"从项目 assets 目录加载 norm_stats: {project_assets_dir / data_config.asset_id}")
            except FileNotFoundError:
                logging.warning(
                    f"未找到 norm_stats (checked: {ckpt_assets_dir}, {project_assets_dir}). "
                    "推理时将跳过归一化, 结果可能不正确!"
                )

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        norm_stats=norm_stats,
    )

    # ── ActionProjectionHead: 处理 action_dim 不匹配 ──
    if args.use_action_proj_head and args.model_action_dim != args.action_dim:
        import torch
        from openpi.models_pytorch.action_proj_head import ActionProjectionHead

        logging.info(
            f"启用 ActionProjectionHead: model_action_dim={args.model_action_dim} → action_dim={args.action_dim}"
        )
        proj_head = ActionProjectionHead(
            actual_action_dim=args.action_dim,
            model_action_dim=args.model_action_dim,
        )

        # 加载 projection head 权重 (如果指定)
        if args.action_proj_head_path:
            proj_head_path = pathlib.Path(args.action_proj_head_path)
            if proj_head_path.exists():
                proj_head.load_state_dict(torch.load(proj_head_path, map_location="cpu"))
                logging.info(f"加载 ActionProjectionHead 权重: {proj_head_path}")
            else:
                logging.warning(f"未找到 ActionProjectionHead 权重: {proj_head_path}, 使用随机初始化")

        # 设置设备
        device = args.pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
        proj_head = proj_head.to(device)
        proj_head.eval()

        # 包装 policy, 在输出时应用 projection
        original_infer = policy.infer

        def patched_infer(obs, *, noise=None):
            result = original_infer(obs, noise=noise)
            # result["actions"] shape: [action_horizon, model_action_dim]
            if "actions" in result:
                actions = torch.from_numpy(result["actions"]).to(device)
                with torch.no_grad():
                    projected_actions = proj_head.project_to_actual(actions.unsqueeze(0))  # [1, H, actual_dim]
                result["actions"] = projected_actions.squeeze(0).cpu().numpy()  # [H, actual_dim]
                logging.debug(f"ActionProjectionHead: {actions.shape} → {result['actions'].shape}")
            return result

        policy.infer = patched_infer
        logging.info(f"ActionProjectionHead 已启用, 输出维度: {args.action_dim}")

    return policy


def main(args: Args) -> None:
    # 显示 PyTorch 设备信息
    try:
        import torch
        if args.pytorch_device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.pytorch_device
        logging.info(f"PyTorch 推理设备: {device}")
        if "cuda" in device and torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            # logging.info(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    except ImportError:
        logging.warning("未安装 PyTorch, 将使用 JAX 模式")

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
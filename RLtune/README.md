# ACoT-VLA RL Fine-tuning (GRPO)

基于 **GRPO (Group Relative Policy Optimization)** 的强化学习微调框架，参考 [SimpleVLA-RL](https://github.com/OpenDriveLab/SimpleVLA-RL) 设计，适配 ACoT-VLA 的 JAX/Flax flow-matching 架构。

## 核心思想

### 为什么用 GRPO？

1. **无需 Critic 网络**: GRPO 通过组内相对排名估计优势，省去价值网络的训练开销
2. **适合 flow matching**: 不需要显式 log-probability，直接用优势加权 flow matching 损失
3. **简单有效**: 仅需二值奖励（成功/失败），无需复杂奖励塑形

### 与 SimpleVLA-RL 的关键差异

| 特性 | SimpleVLA-RL | ACoT-VLA RL |
|------|-------------|-------------|
| 模型架构 | OpenVLA (自回归 token) | Pi0/ACoT-VLA (flow matching) |
| 策略损失 | PPO 裁剪 log-prob 比率 | 优势加权 MSE 损失 |
| 框架 | PyTorch + veRL + Ray | JAX/Flax |
| 动作空间 | 离散 token | 连续 ODE 去噪 |

### 数学公式

标准 flow matching 损失:
$$L_{fm} = \mathbb{E}\left[\|v_\theta(x_t, t) - u_t\|^2\right]$$

GRPO 优势加权损失:
$$L_{rl} = \mathbb{E}\left[A_i \cdot \|v_\theta(x_t, t) - u_t\|^2\right]$$

其中 $A_i$ 是 GRPO 组内归一化优势:
$$A_i = \frac{R_i - \mu_{group}}{\sigma_{group} + \epsilon}$$

## 文件结构

```
RLtune/
├── __init__.py          # 包入口
├── rl_config.py         # 配置数据类
├── grpo_algo.py         # GRPO 算法核心
├── env_runner.py        # 环境 rollout 收集
├── reward_manager.py    # 奖励计算
├── train.py             # 主训练脚本
├── run_rl_train.sh      # 启动脚本
└── README.md            # 本文档
```

## 快速开始

### 1. 环境准备

确保已安装 SRB 环境:
```bash
pip install srb
```

### 2. 启动策略服务器

在终端 1 启动模型服务:
```bash
python -m examples.srb.serve --config_name srb_train --checkpoint_dir path/to/sft_ckpt
```

### 3. 运行 RL 训练

在终端 2 启动 RL 训练:
```bash
# 基本用法
bash RLtune/run_rl_train.sh srb_train path/to/sft_checkpoint

# 自定义超参数
python -m RLtune.train \
    --config_name srb_train \
    --checkpoint_dir path/to/sft_checkpoint \
    --n_samples 8 \
    --total_epochs 100 \
    --learning_rate 5e-6 \
    --sampling_temperature 1.6
```

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_samples` | 8 | 每个任务的采样轨迹数（组大小） |
| `clip_ratio_high` | 0.28 | PPO 上裁剪比率（非对称） |
| `clip_ratio_low` | 0.2 | PPO 下裁剪比率 |
| `reward_coef` | 5.0 | 奖励缩放系数 |
| `sampling_temperature` | 1.6 | 采样温度（高温促进探索） |
| `accuracy_lower_bound` | 0.1 | 准确率过滤下界 |
| `accuracy_upper_bound` | 0.9 | 准确率过滤上界 |
| `learning_rate` | 5e-6 | 学习率（低于 SFT） |
| `total_epochs` | 100 | 训练轮数 |

## 训练流程

```
for epoch in range(total_epochs):
    1. 收集 Rollout: 在环境中运行策略，获取轨迹
    2. 计算奖励: 二值成功/失败奖励
    3. 准确率过滤: 只保留中等难度任务
    4. 计算优势: GRPO 组内归一化
    5. 更新策略: 优势加权 flow matching 损失
    6. 日志记录和检查点保存
```

## 扩展指南

### 添加新环境

1. 在 `env_runner.py` 中实现新的 `_create_env` 方法
2. 更新 `EnvConfig` 以支持新环境参数

### 自定义奖励

继承 `DenseRewardManager` 并重写 `compute_rewards` 方法:
```python
class MyRewardManager(DenseRewardManager):
    def compute_rewards(self, trajectories):
        # 自定义奖励逻辑
        pass
```

### 使用不同模型

修改 `train.py` 中的 `init_train_state` 函数以加载不同模型架构。


让我先了解一下项目的 RL 微调流程和配置。 


---

## 新环境 RL 微调完整流程

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL 微调流程 (GRPO)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    WebSocket     ┌──────────────────┐        │
│  │  环境 (Env)   │ ◄────────────►  │  Policy Server    │        │
│  │  你的新环境    │    obs/action   │  (模型推理)       │        │
│  └──────┬───────┘                  └──────────────────┘        │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ EnvRunner    │───►│ RewardManager│───►│ GRPO Algo    │      │
│  │ 收集轨迹       │    │ 计算奖励      │    │ 计算优势      │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                 │               │
│                                                 ▼               │
│                                        ┌──────────────┐        │
│                                        │ Policy Update│        │
│                                        │ 优势加权损失   │        │
│                                        └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 需要修改/创建的组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 1. 训练配置 | config.py | 添加新环境的 `TrainConfig` |
| 2. 数据变换 | `src/openpi/policies/你的环境_policy.py` | 定义输入/输出变换 |
| 3. norm_stats | `assets/你的配置名/` | 计算归一化统计量 |
| 4. 环境 Runner | env_runner.py | 适配新环境的 `_create_env` |
| 5. 奖励函数 | reward_manager.py | 可能需要自定义奖励 |
| 6. Policy Server | `examples/你的环境/serve.py` | 启动模型服务 |

---

### Step 1: 准备数据集 (LeRobot 格式)

你的数据集需要是 LeRobot 格式，结构如下：

```
dataset/你的数据集/
├── meta/
│   └── info.json           # 数据集元信息
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

每个 parquet 文件包含：
- `observation.images.image_base` — 基座相机图像
- `observation.images.image_wrist` — 腕部相机图像 (可选)
- `observation.state` — 本体感受状态 (关节角度等)
- `action` — 动作标签
- `episode_index` — episode 编号
- `frame_index` — 帧编号

### Step 2: 创建数据变换 (Policy Transform)

创建 `src/openpi/policies/你的环境_policy.py`：

```python
"""数据变换 for 你的环境."""

import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model

class YourEnvInputs(_transforms.DataTransformFn):
    """将环境观测转换为模型输入."""
    
    def __init__(self, action_dim: int, model_type: _model.ModelType,
                 observation_keys: tuple[str, ...] = ("observation.state",),
                 image_keys: tuple[str, ...] = ("observation.images.image_base",)):
        self.action_dim = action_dim
        self.model_type = model_type
        self.observation_keys = observation_keys
        self.image_keys = image_keys
    
    def __call__(self, x: dict) -> dict:
        # 处理 state
        state = np.concatenate([np.asarray(x[k]).flatten() for k in self.observation_keys])
        # 填充或截断到模型期望的维度
        if len(state) < 32:  # 模型内部状态维度
            state = np.concatenate([state, np.zeros(32 - len(state))])
        else:
            state = state[:32]
        
        result = {"state": state.astype(np.float32)}
        
        # 处理图像
        for i, img_key in enumerate(self.image_keys):
            if img_key in x:
                result[f"image_{i}"] = np.asarray(x[img_key])
        
        # 处理 action (训练时)
        if "action" in x:
            action = np.asarray(x["action"]).flatten()
            if len(action) < self.action_dim:
                action = np.concatenate([action, np.zeros(self.action_dim - len(action))])
            else:
                action = action[:self.action_dim]
            result["actions"] = action.astype(np.float32)
        
        return result


class YourEnvOutputs(_transforms.DataTransformFn):
    """将模型输出转换为环境动作."""
    
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
    
    def __call__(self, x: dict) -> dict:
        actions = np.asarray(x["actions"])
        # 截断到环境期望的维度
        if actions.ndim == 2:
            actions = actions[:, :self.action_dim]
        else:
            actions = actions[:self.action_dim]
        return {"actions": actions}
```

### Step 3: 添加训练配置

在 config.py 的 `_CONFIGS` 列表中添加：

```python
TrainConfig(
    name="your_env_train",  # 配置名称，CLI 使用
    model=pi0_config.Pi0Config(
        pi05=True,  # 使用 PI0.5 (quantile 归一化)
        action_horizon=16,
        action_dim=7,  # 你的环境动作维度
        paligemma_variant="gemma_2b_lora",  # 或 "gemma_4_e2b" 如果用 Gemma 4
        action_expert_variant="gemma_300m_lora",
    ),
    data=SRBDataConfig(  # 或自定义 DataConfigFactory
        repo_id="你的数据集名称",  # dataset/ 下的文件夹名
        base_config=DataConfig(
            prompt_from_task=True,
            action_sequence_keys=("action",),
        ),
        action_dim=7,
        observation_keys=("observation.state",),
        image_keys=("observation.images.image_base",),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"  # 预训练权重
    ),
    num_train_steps=20_000,
    batch_size=32,
    exp_name="your_experiment",
    wandb_enabled=True,
)
```

### Step 4: 计算 norm_stats

```bash
# 计算归一化统计量
uv run python scripts/compute_norm_stats.py --config-name your_env_train
```

这会生成 `assets/your_env_train/你的数据集名称/norm_stats.json`。

### Step 5: SFT 预训练 (先用演示数据微调)

```bash
# 单卡
uv run python scripts/train.py --config_name your_env_train

# 多卡
bash scripts/train.sh your_env_train your_experiment
```

### Step 6: 适配环境 Runner

修改 env_runner.py 中的 `_create_env` 方法，或创建新的环境适配器：

```python
def _create_env(self, seed: int = 0):
    """创建你的环境."""
    import gymnasium  # 或你的环境库
    
    # 方式1: 直接使用 Gymnasium 环境
    env = gymnasium.make(
        "YourEnv-v1",
        render_mode="rgb_array" if self.env_config.enable_cameras else None,
        seed=seed,
    )
    
    # 方式2: 使用自定义环境
    # from your_env import YourEnv
    # env = YourEnv(seed=seed, headless=self.env_config.headless)
    
    task_description = self.env_config.prompt
    return env, task_description
```

**关键**: 环境需要实现标准 Gymnasium 接口：
- `env.reset()` → `(obs, info)`
- `env.step(action)` → `(obs, reward, terminated, truncated, info)`
- `info` 中需要包含 `"success"` 或 `"is_success"` 字段

### Step 7: 启动 Policy Server

创建 `examples/你的环境/serve.py`：

```python
import dataclasses
import tyro
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server_sample as wps_sample

@dataclasses.dataclass
class Args:
    config_name: str = "your_env_train"
    checkpoint_dir: str = "checkpoints/your_env_train/your_experiment"
    host: str = "0.0.0.0"
    port: int = 8899
    default_prompt: str = "完成任务"

def main(args: Args):
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    server = wps_sample.WebsocketPolicyServerSample(
        policy=policy, host=args.host, port=args.port,
    )
    server.serve_forever()

if __name__ == "__main__":
    main(tyro.cli(Args))
```

### Step 8: 运行 RL 微调

```bash
# 终端 1: 启动 Policy Server
python -m examples.your_env.serve \
    --config_name your_env_train \
    --checkpoint_dir checkpoints/your_env_train/your_experiment

# 终端 2: 运行 RL 训练
python -m RLtune.train_pytorch \
    --config_name your_env_train \
    --checkpoint_dir checkpoints/your_env_train/your_experiment \
    --env_id "你的环境ID" \
    --default_prompt "完成任务" \
    --n_samples 8 \
    --total_epochs 100 \
    --learning_rate 5e-6

# 或使用启动脚本
bash RLtune/run_rl_train_pytorch.sh your_env_train \
    checkpoints/your_env_train/your_experiment
```

### Step 9: 自定义奖励 (可选)

如果需要更复杂的奖励函数，修改 reward_manager.py：

```python
class YourEnvRewardManager:
    def compute_rewards(self, trajectories):
        rewards = []
        for traj in trajectories:
            # 自定义奖励逻辑
            if traj.success:
                reward = 1.0
                # 可选: 长度奖励 (越短越好)
                reward += 0.1 * (1.0 - traj.episode_length / self.max_steps)
            else:
                reward = 0.0
            rewards.append(reward)
        return np.array(rewards), {"success_rate": np.mean([t.success for t in trajectories])}
```

## 加载数据集

```bash
mkdir -p ~/.cache/huggingface/lerobot && ln -s /media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa/dataset/srb_dataset ~/.cache/huggingface/lerobot/srb_dataset && echo "符号链接创建成功" && ls -la ~/.cache/huggingface/lerobot/srb_dataset
```

---

## 如果使用 Gemma 4 E2B

Gemma 4 只是改了 backbone，流程完全一样：

```python
# 在 config.py 中修改模型配置
model=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_4_e2b",  # ← 改这里
    action_expert_variant="gemma_300m_lora",
    # 如果需要指定本地模型路径
    # gemma4_model_path="./models/gemma-4-E2B",
)
```

**norm_stats 不变** — 它只依赖数据集，不依赖模型。

## 双卡

FSDP 双卡显存共享已全部实现完毕：

1. **train_offline.py**：
   - `wrap_model_fsdp()` — FULL_SHARD 策略 + bf16 混合精度
   - `save_fsdp_checkpoint()` / `load_fsdp_checkpoint()` — 完整状态收集/加载
   - 模型在 CPU 上构建，FSDP 自动分片到 2 张卡

2. **train_pytorch.py**：
   - `get_model()` 自动解包 FSDP 包装器

### 启动方式

```bash
torchrun --nproc_per_node=2 RLtune/train_offline.py [原有参数...]
```

FSDP 会自动将模型参数、梯度、优化器状态分片到两张卡上，每张卡只需约 50% 的显存。 


### Jax 版本 train_offline

```bash
cd /root/SpaceRobot/ACoT-VLA_saa
.venv/bin/python -m RLtune.train_offline_jax --config-name srb_train_tracking --total-epochs 20 --num-train-steps-per-epoch 10
```

---

## GRPO 路线

我会把“离线加权模仿”和“真正的 on-policy GRPO”拆开说明，并给出按你是否能稳定跑 SRB 环境来选择的下一步路线。

结论先说清楚：当前实现适合当作“离线、按相对回报重加权的行为克隆”基线；要升级成真正 GRPO，首要阻塞项不是再调权重，而是收集当前策略 rollout 并得到每个动作块的可计算 `log_prob`。



**差别是什么**

当前目标是：

$$
w_i=\frac{\exp(\mathrm{clip}(A_i/T))}{\operatorname{mean}_j\exp(\mathrm{clip}(A_j/T))},\qquad
L=\operatorname{mean}_i[w_i L_{\mathrm{FM}}(o_i,a_i)]
$$

其中 $(o_i,a_i,R_i)$ 来自固定离线数据集。它的含义是：更认真模仿高回报 demo，弱化低回报 demo。负 advantage 只会降低样本权重，不会像 PPO 那样直接约束“把该动作概率压低”。

标准 GRPO/PPO 则需要：

$$
\tau_i \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x),\qquad
r_i(\theta)=\exp(\log\pi_\theta(a_i\mid o_i)-\log\pi_{\theta_{\mathrm{old}}}(a_i\mid o_i))
$$

再优化：

$$
-\mathbb{E}\left[\min(r_iA_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A_i)\right]
+ \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

GRPO 不一定需要 critic，但必须有：

- 同一任务下、由冻结旧策略采出的多条 rollout；
- 这些 rollout 的当前环境 reward；
- 当前与旧策略对相同行动的 `log_prob`，用于 ratio 和 clip。

当前 flow-matching 模型的 `compute_loss()` 是去噪 MSE，不是 `-log_prob`。因此不能把 `old_loss / new_loss` 当 PPO ratio；那只是启发式，不是标准 GRPO。

**下一步建议**

优先做一次在线 rollout smoke test，而不是直接写 PPO loss：

1. 冻结当前策略为 `theta_old`，对每个 prompt 采样 $G=4$ 条随机 rollout。
2. 保存每个决策点的 observation、完整 action chunk、执行后的 reward、prompt、seed，以及旧策略版本。
3. 确认同一 prompt 的 4 条 rollout 有足够 reward 方差；否则 GRPO advantage 大多为零，先解决探索/奖励，而非优化器。
4. 先用这些新 rollout 做当前的有界 advantage-weighted flow-matching 更新。这仍不是 PPO，但已从“静态 demo 重加权”升级为 on-policy-ish 的 policy improvement。

要实现**真正 GRPO**，下一项核心工程是给 flow policy 增加可计算的 `log_prob(action_chunk | observation)`：

- 较严格但昂贵：把 flow ODE 当 continuous normalizing flow，通过散度积分估计 action density。
- 更实用：改为带可追踪随机转移的 diffusion/高斯 action policy，缓存每个采样步的旧 `log_prob`。
- 完成后再接入 old-policy snapshot、ratio、PPO clip、reference KL，以及每轮若干个 update epoch。

若暂时无法稳定跑 SRB 环境，应继续走离线路线，并把方法明确命名为 offline AWR / return-weighted behavior cloning；此时更值得做的是 episode 级 train/eval 切分、SFT 基线对比、奖励语义校验，以及加入少量未加权 BC anchor。

---

## 快速检查清单

- [ ] 数据集转为 LeRobot 格式
- [ ] 创建 `你的环境_policy.py` 数据变换
- [ ] 在 config.py 添加 `TrainConfig`
- [ ] 运行 compute_norm_stats.py
- [ ] SFT 预训练得到初始 checkpoint
- [ ] 实现环境 Gymnasium 接口
- [ ] 创建 serve.py 启动脚本
- [ ] 修改 env_runner.py 的 `_create_env`
- [ ] 运行 RL 微调
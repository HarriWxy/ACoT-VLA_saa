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

---

## 修改

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#370-370), lines 370 to 500

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1180-1180), lines 1180 to 1295

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/model.py#360-360), lines 360 to 395

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#70-70), lines 70 to 125

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#330-330), lines 330 to 435

更新了待办事项列表

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#430-430), lines 430 to 493

Searched for regex `class PhysicsAwareConfig|PhysicsAwareConfig|def get_freeze_filter` (`**/src/openpi/models/pi0_config.py`), 4 results

Searched for regex `repack_transforms|data_transforms|model_transforms` (`**/RLtune/data_loader_rl.py`), 6 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/pi0_config.py#1-1), lines 1 to 220

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/data_loader_rl.py#500-500), lines 500 to 594

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/utils.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/data_loader_rl.py#150-150), lines 150 to 245

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/data_loader_rl.py#245-245), lines 245 to 390

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#115-115), lines 115 to 230

真实 batch 端到端已通过，接下来回到参数加载器。当前可疑点很局部：`flax.traverse_util` 的默认 tuple path 能保留 NNX 参数树中的整数索引，只有 `sep="/"` 会把它们强行拼成字符串；我先用 dummy PhysicsAware 参数树确认这个假设，再做兼容 loader 的小改动。

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import jax; from flax import nnx, traverse_util; from openpi.models import pi0_config; config=pi0_config.PhysicsAwareConfig(paligemma_variant="dummy", action_expert_variant="dummy", action_dim=19, action_horizon=16, physics_dim=5); model=config.create(jax.random.key(0)); params=nnx.state(model, nnx.Param); flat=traverse_util.flatten_dict(params); print("leaves", len(flat)); print("integer_paths", sum(any(isinstance(part, int) for part in path) for path in flat)); print("sample", list(flat)[:8]); rebuilt=traverse_util.unflatten_dict(flat); print("roundtrip", type(rebuilt), len(traverse_util.flatten_dict(rebuilt)))'

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Ran terminal command:  rg -n "weight_loader|nnx\.state|state\(model" src RLtune scripts | head -120

Ran terminal command:  grep -RInE "weight_loader|nnx\.state|state\(model" src RLtune scripts | head -120

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_WFiVmWN9ELYVJ3cctu0NML0b__vscode-1785496181010/content.txt)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train.py#120-120), lines 120 to 220

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/train.py#55-55), lines 55 to 140

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/pi0_test.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/model_test.py#50-50), lines 50 to 95

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import jax; from flax import nnx, traverse_util; from openpi.models import pi0_config; config=pi0_config.PhysicsAwareConfig(paligemma_variant="dummy", action_expert_variant="dummy", action_dim=19, action_horizon=16, physics_dim=5); model=nnx.eval_shape(config.create, jax.random.key(0)); params=nnx.state(model).to_pure_dict(); flat=traverse_util.flatten_dict(params); print("leaves", len(flat)); print("integer_paths", sum(any(isinstance(part, int) for part in path) for path in flat)); print("sample", list(flat)[:8]); rebuilt=traverse_util.unflatten_dict(flat); print("roundtrip", len(traverse_util.flatten_dict(rebuilt)))'

参数树确认了假设：训练契约传入的是纯 dict，PhysicsAware 有 24 条包含整数索引的路径；不带 `sep` 的 `flatten_dict` 可完整 round-trip。现在只改 `CompatibleCheckpointWeightLoader` 使用 tuple path，并立即用合成 checkpoint 验证“匹配叶加载、不匹配叶保留初始化值”。

Generating patch (13 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/weight_loaders.py)


```

```


Ran terminal command:  uv run python -c 'from unittest.mock import patch; import numpy as np; from openpi.training.weight_loaders import CompatibleCheckpointWeightLoader; reference={"blocks": {0: {"kernel": np.zeros((2, 2), dtype=np.float32), "bias": np.zeros((3,), dtype=np.float32)}, 1: {"kernel": np.zeros((2, 2), dtype=np.float32)}}}; loaded={"blocks": {0: {"kernel": np.ones((2, 2), dtype=np.float16), "bias": np.ones((4,), dtype=np.float32)}}}; with patch("openpi.training.weight_loaders.download.maybe_download", return_value="unused"), patch("openpi.training.weight_loaders._model.restore_params", return_value=loaded): result=CompatibleCheckpointWeightLoader("unused").load(reference); assert result["blocks"][0]["kernel"].shape == (2, 2); assert result["blocks"][0]["kernel"].dtype == np.float32; assert np.array_equal(result["blocks"][0]["bias"], reference["blocks"][0]["bias"]); assert np.array_equal(result["blocks"][1]["kernel"], reference["blocks"][1]["kernel"]); print("compatible loader tuple-path smoke test passed")'

Ran terminal command:  uv run python - <<'PY'
from unittest.mock import patch

import numpy as np

from openpi.training.weight_loaders import CompatibleCheckpointWeightLoader

reference = {
    "blocks": {
        0: {
            "kernel": np.zeros((2, 2), dtype=np.float32),
            "bias": np.zeros((3,), dtype=np.float32),
        },
        1: {"kernel": np.zeros((2, 2), dtype=np.float32)},
    }
}
loaded = {
    "blocks": {
        0: {
            "kernel": np.ones((2, 2), dtype=np.float16),
            "bias": np.ones((4,), dtype=np.float32),
        }
    }
}
with (
    patch("openpi.training.weight_loaders.download.maybe_download", return_value="unused"),
    patch("openpi.training.weight_loaders._model.restore_params", return_value=loaded),
):
    result = CompatibleCheckpointWeightLoader("unused").load(reference)

assert result["blocks"][0]["kernel"].shape == (2, 2)
assert result["blocks"][0]["kernel"].dtype == np.float32
assert np.array_equal(result["blocks"][0]["bias"], reference["blocks"][0]["bias"])
assert np.array_equal(result["blocks"][1]["kernel"], reference["blocks"][1]["kernel"])
print("compatible loader tuple-path smoke test passed")
PY

兼容 loader 修复后的合成测试通过：整数路径可 round-trip，shape 不匹配和缺失叶都保留初始化值，dtype 会按目标参数转换。下一步检查真实 checkpoint 的命名契约和当前 PhysicsAware 模型结构，避免“能加载”但实际只加载了极少或错误的参数。

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/physics_aware_single_step.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/pi0_config.py#205-205), lines 205 to 306

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/pi0_single_step.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/pi0.py#1-1), lines 1 to 200

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_sample)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_gemma4/srb_sample)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_gemma4/srb_sample)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_gemma4/srb_sample/rl_grpo_offline_jax)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking/norm_stats.json)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/5)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/10)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/15)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/tensorboard)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax)

Ran terminal command:  find checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params -maxdepth 4 -type f | head -40 && du -sh checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params

No matches found. Your search pattern might be excluded completely by either the search.exclude settings or .*ignore files.
If you believe that it should have results, you can check into the .*ignore files and the exclude setting (here are some excluded patterns for reference:[**/node_modules,**/bower_components,**/*.code-search]).
Then if you want to include those files you can call the tool again by setting "includeIgnoredFiles" to true.

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py#1-1), lines 1 to 180

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/rl_config.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train.py#1-1), lines 1 to 125

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline.py#1-1), lines 1 to 230

Searched for regex `def main|config_name|action_dim|action_horizon|checkpoint_dir` (`**/RLtune/train_offline_jax.py`), 16 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py#200-200), lines 200 to 360

Ran terminal command:  JAX_PLATFORMS=cpu uv run python - <<'PY'
import jax
import orbax.checkpoint as ocp
from flax import traverse_util

path = "checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params"
with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(path)
params_metadata = metadata["params"]
flat = traverse_util.flatten_dict(params_metadata)
print("checkpoint leaves:", len(flat))
for path, value in list(flat.items())[:12]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    print(path, shape, dtype)
print("contains_lora:", any("lora" in part for path in flat for part in path if isinstance(part, str)))
PY

Ran terminal command:  JAX_PLATFORMS=cpu uv run python - <<'PY'
from pathlib import Path

import orbax.checkpoint as ocp
from flax import traverse_util

path = str(Path("checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params").resolve())
with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(path)
params_metadata = metadata["params"]
flat = traverse_util.flatten_dict(params_metadata)
print("checkpoint leaves:", len(flat))
for path, value in list(flat.items())[:12]:
    print(path, getattr(value, "shape", None), getattr(value, "dtype", None))
print("contains_lora:", any("lora" in part for path in flat for part in path if isinstance(part, str)))
PY

本地第 20 步 checkpoint 的元数据可读，包含 71 个参数叶并保留 LoRA；它不是普通 base 权重。现在做一次不读取 6GB 数组的 shape/key 对照：用实际 `physics_aware_srb_train_tracking` 的抽象模型与 checkpoint metadata 计算匹配率，结果会决定是否切换 config 的初始化路径。

Ran terminal command:  JAX_PLATFORMS=cpu uv run python - <<'PY'
from pathlib import Path

import jax
from flax import nnx, traverse_util
import orbax.checkpoint as ocp

from openpi.training.config import get_config

config = get_config("physics_aware_srb_train_tracking")
abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
target = traverse_util.flatten_dict(nnx.state(abstract_model).to_pure_dict())

path = str(Path("checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params").resolve())
with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(path)["params"]
source = traverse_util.flatten_dict(metadata)
source = {key[:-1] if key[-1] == "value" else key: value for key, value in source.items()}
matching = [key for key, value in source.items() if key in target and target[key].shape == value.shape]
print("target leaves:", len(target))
print("source leaves:", len(source))
print("matching leaves:", len(matching))
print("matching ratio:", f"{len(matching) / len(target):.1%}")
print("new physics leaves:", sum(key not in source for key in target if any("physics" in str(part) for part in key)))
print("action head matches:", [key for key in matching if "action_" in "/".join(map(str, key))])
PY

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#1-1), lines 1 to 95

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize.py)

Searched for text `SRBDataConfig(` (`**/src/openpi/training/config.py`), 9 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1085-1085), lines 1085 to 1185

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1310-1310), lines 1310 to 1380

对照结果显示本地 SRB checkpoint 有 64/99 个叶与 PhysicsAware 目标模型完全匹配，物理 encoder 和新 action head 会保留初始化，LoRA/主干可复用。另一个数据端问题也可以局部解决：让 `RepackTransform` 支持“存在则保留”的可选路径，物理字段缺失时仍由 SRB 默认值兜底。

Generating patch (52 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py)


```

```

```

```


Ran terminal command:  uv run python - <<'PY'
import numpy as np

from openpi.transforms import RepackTransform

transform = RepackTransform(
    {"state": "state", "actions": "action"},
    optional={"physics_params": "physics_params", "gravity": "gravity"},
)
with_physics = transform(
    {
        "state": np.zeros(2, dtype=np.float32),
        "action": np.ones(3, dtype=np.float32),
        "gravity": np.float32(3.7),
    }
)
without_physics = transform(
    {
        "state": np.zeros(2, dtype=np.float32),
        "action": np.ones(3, dtype=np.float32),
    }
)
assert "gravity" in with_physics
assert "physics_params" not in with_physics
assert "gravity" not in without_physics
assert without_physics["actions"].shape == (3,)
print("optional repack smoke test passed")
PY

可选 repack 的行为测试通过。现在把 normalization 接到这条路径上：PhysicsAware 配置统计 `physics_params`，并恢复脚本 CLI，这样未来物理条件变化时可以直接生成对应的 5 维统计；当前固定默认值会自然得到零方差并被稳定处理。

Generating patch (13 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py)


```

```


已创建 4 个待办事项

Ran terminal command:  uv run python scripts/compute_norm_stats.py --help

CLI 解析已通过。接着用实际 `physics_aware_srb_train_tracking` 数据跑一次统计，确认可选 repack、默认物理值和 `physics_params` 统计在真实 LeRobot 数据上都能工作；这也会更新该 config 复用的 norm stats。

Ran terminal command:  JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name physics_aware_srb_train_tracking

Searched for text `class AssetsConfig` (`**/src/openpi/training/config.py`), 1 result

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#35-35), lines 35 to 70

统计脚本已经真实跑完 430 个 batch，但发现一个配置细节：新 config 明确复用了 srb_train_tracking，脚本却仍按 config 名写到了 physics_aware_srb_train_tracking，因此结果不会被读取。我会让脚本遵循 `AssetsConfig.assets_dir/asset_id`，再把统计写到实际使用的位置。

Generating patch (13 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py)


```

```


Ran terminal command:  JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name physics_aware_srb_train_tracking

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking/norm_stats.json)

真实统计已写入配置实际使用的 assets，且 `physics_params` 已被统计为 5 维；当前数据中的物理值确实几乎恒定，所以它不会创造样本间条件差异。现在做一次工作区 diff、静态诊断和最小模型/loader 测试，集中找出剩余的代码级问题。

Ran terminal command:  git diff --check && git diff -- src/openpi/training/weight_loaders.py src/openpi/transforms.py src/openpi/training/config.py scripts/compute_norm_stats.py RLtune/train_offline_jax.py src/openpi/policies/srb_policy.py src/openpi/models/physics_aware_single_step.py

Ran terminal command:  git status --short --ignored assets/srb_train_tracking/srb_tracking/norm_stats.json

Checked [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/weight_loaders.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/physics_aware_single_step.py), no problems found

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_mw0kvTQv2t0YJ1A7oRbjTUpC__vscode-1785496181069/content.txt#1-1), lines 1 to 260

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_mw0kvTQv2t0YJ1A7oRbjTUpC__vscode-1785496181069/content.txt#260-260), lines 260 to 347

Ran terminal command:  uv run ruff check src/openpi/models/physics_aware_single_step.py src/openpi/policies/srb_policy.py src/openpi/training/config.py src/openpi/training/weight_loaders.py src/openpi/transforms.py scripts/compute_norm_stats.py RLtune/train_offline_jax.py

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_kf1P6KPnRYYS1NA02OSlLD3R__vscode-1785496181074/content.txt#1-1), lines 1 to 240

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_kf1P6KPnRYYS1NA02OSlLD3R__vscode-1785496181074/content.txt#240-240), lines 240 to 520

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_kf1P6KPnRYYS1NA02OSlLD3R__vscode-1785496181074/content.txt#520-520), lines 520 to 606

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import dataclasses; import numpy as np; from openpi.training.config import get_config; from RLtune import data_loader_rl; config=dataclasses.replace(get_config("physics_aware_srb_train_tracking"), num_workers=0); loader=data_loader_rl.create_data_loader(config, sharding=None, shuffle=False, num_batches=1, skip_norm_stats=False, episode_aware=True, reward_aggregation="sum"); observation, actions=next(iter(loader)); values=np.asarray(observation.physics_params); print("physics range", values.min(axis=0), values.max(axis=0), "mean", values.mean(axis=0)); assert np.isfinite(values).all(); assert values.shape == (8, 5)'

Searched for text `class DataConfig` (`**/src/openpi/training/config.py`), 2 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#65-65), lines 65 to 145

静态检查没有发现语法或类型诊断，但真实 batch 暴露了一个可改进点：退化的 quantile stats 让固定物理值全部变成 `-1`。我会让 quantile normalization 在 `q01 == q99` 时回退到 z-score（常量输入得到 0），同时把新 config 的初始化源切到已验证存在的 SRB offline checkpoint，并清理本次改动引入的 lint 问题。

Generating patch (21 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py)


```

```


Searched for text `PhysicsAwareConfig` (`**/src/openpi/training/config.py`), 6 results

Generating patch (15 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py)


```

```


Searched for text `PhysicsAwareConfig` (`**/src/openpi/training/config.py`), 6 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1145-1145), lines 1145 to 1195

Generating patch (18 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py)

Generating patch (16 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py)

正在启动: *Connect normalization to PhysicsAware config* (3/3)


```

```


Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1158-1158), lines 1158 to 1185

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py#1228-1228), lines 1228 to 1285

Generating patch (13 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py)


```

```


Generating patch (50 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/physics_aware_single_step.py)


```

```


Generating patch (14 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy.py)


```

```


Generating patch (48 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py)


```

```

```

```


Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import dataclasses; import numpy as np; from openpi.training.config import get_config; from RLtune import data_loader_rl; config=dataclasses.replace(get_config("physics_aware_srb_train_tracking"), num_workers=0); loader=data_loader_rl.create_data_loader(config, sharding=None, shuffle=False, num_batches=1, skip_norm_stats=False, episode_aware=True, reward_aggregation="sum"); observation, actions=next(iter(loader)); values=np.asarray(observation.physics_params); assert values.shape == (8, 5); assert np.isfinite(values).all(); assert np.max(np.abs(values)) < 1e-3, values; assert actions.shape == (8, 16, 19); print("normalized physics-aware loader smoke test passed", values.shape, values[0])'

Ran terminal command:  free -h && du -sh checkpoints/srb_train_tracking/srb_sample/rl_grpo_offline_jax/20/params

归一化修复后的真实 loader 已通过，固定物理值现在接近零中心。机器可用内存充足，我接着做一次真实 6GB checkpoint 的兼容加载 smoke test：只构造抽象目标参数并比较匹配叶，不执行训练或把模型放上异构 GPU。

Ran terminal command:  JAX_PLATFORMS=cpu uv run python - <<'PY'
import gc

import jax
from flax import nnx, traverse_util

from openpi.training.config import get_config

config = get_config("physics_aware_srb_train_tracking")
abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
target = nnx.state(abstract_model).to_pure_dict()
loaded = config.weight_loader.load(target)
flat_target = traverse_util.flatten_dict(target)
flat_loaded = traverse_util.flatten_dict(loaded)
loaded_leaves = [
    key for key, value in flat_loaded.items() if not isinstance(value, jax.ShapeDtypeStruct)
]
assert len(loaded_leaves) > 0
assert len(loaded_leaves) < len(flat_target)
assert all(key in flat_target and flat_loaded[key].shape == flat_target[key].shape for key in loaded_leaves)
print("real compatible checkpoint smoke test passed", len(loaded_leaves), "/", len(flat_target), "leaves")
del loaded
_gc = gc.collect()
PY

Ran terminal command:  uv run ruff check src/openpi/models/physics_aware_single_step.py src/openpi/policies/srb_policy.py src/openpi/training/config.py src/openpi/training/weight_loaders.py src/openpi/transforms.py scripts/compute_norm_stats.py RLtune/train_offline_jax.py

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_3xqNxYPXccslp4Td30urvquT__vscode-1785496181097/content.txt#1-1), lines 1 to 260

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_3xqNxYPXccslp4Td30urvquT__vscode-1785496181097/content.txt#260-260), lines 260 to 432

真实 checkpoint 加载也通过，64/99 个参数叶被安全复用，其余保持初始化。Ruff 剩余项主要是仓库原有的导入/类型升级规则；我只自动修复两个本次直接改动模型文件的空白和导入项，不触碰配置文件里的历史 lint 噪音。

Ran terminal command:  uv run ruff check --fix src/openpi/models/physics_aware_single_step.py src/openpi/policies/srb_policy.py

Ran terminal command:  JAX_PLATFORMS=cpu uv run python - <<'PY'
import jax

from openpi.models import pi0_config
from openpi.shared import nnx_utils

config = pi0_config.PhysicsAwareConfig(
    paligemma_variant="dummy",
    action_expert_variant="dummy",
    action_dim=19,
    action_horizon=16,
    physics_dim=5,
)
model = config.create(jax.random.key(0))
observation, actions = config.fake_obs(1), config.fake_act(1)
loss = nnx_utils.module_jit(model.compute_loss)(jax.random.key(1), observation, actions)
assert loss.shape == (1, 16)
print("PhysicsAware compute_loss smoke test passed", loss.shape)
PY

Ran terminal command:  uv run ruff check RLtune/train_offline_jax.py scripts/compute_norm_stats.py

Ran terminal command:  uv run ruff check --fix RLtune/train_offline_jax.py

当前功能验证已经覆盖真实 batch、真实 checkpoint 和 dummy loss；脚本入口只剩下仓库原有的 `sys.path` bootstrap 导致的 E402。接下来运行针对模型、transform、配置和 loader 的 pytest，并用 `ruff --select E9,F` 做一轮不受历史风格项干扰的错误检查。

Ran terminal command:  find src/openpi -name '*test.py' -o -name 'test_*.py' | sort | head -80

Ran terminal command:  uv run ruff check --select E9,F src/openpi/models/physics_aware_single_step.py src/openpi/policies/srb_policy.py src/openpi/training/config.py src/openpi/training/weight_loaders.py src/openpi/transforms.py scripts/compute_norm_stats.py RLtune/train_offline_jax.py

Ran terminal command:  uv run python -m RLtune.train_offline_jax --help

Autopilot extended tool call limit

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy_test.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms_test.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize_test.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/data_loader_test.py)

Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/policies/srb_policy_test.py src/openpi/transforms_test.py src/openpi/shared/normalize_test.py

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train/srb_dataset/norm_stats.json#1-1), lines 1 to 80

Ran terminal command:  git blame -L 75,105 -- src/openpi/policies/srb_policy.py && git log -3 --oneline -- src/openpi/policies/srb_policy.py

窄测里已有的 `test_srb_inputs_flatten_and_pad_state` 失败于仓库历史中的硬编码 8 维截断（`git blame` 显示来自更早提交），并非本次 PhysicsAware 改动；其余 15 个相关测试通过。我会补充本次新增的物理默认值/可选 repack/退化归一化测试，再单独复验这些行为。

Generating patch (69 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy_test.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms_test.py)


```

```

```

```


Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/policies/srb_policy_test.py -k physics_aware src/openpi/transforms_test.py -k 'optional_fields or constant_stats'

新增测试抓到一个边界：统计 JSON 的 `float32` 舍入会让常量输入与 `mean` 相差约 `4e-7`，直接除以 epsilon 得到非零噪声。修复为对零方差维度直接输出 0，然后重跑同一组测试。

Generating patch (15 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py)


```

```


正在启动: *Run pytest and error checks* (2/3)

Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/policies/srb_policy_test.py -k physics_aware src/openpi/transforms_test.py -k 'optional_fields or constant_stats'

Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/policies/srb_policy_test.py -k physics_aware

Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/models/model_test.py src/openpi/models/pi0_test.py

物理输入和归一化边界测试现在都通过。模型基础测试仍在 CPU 上运行，我先收尾读取结果；之后再做一次最终 diff/状态检查，确保没有把生成的 assets 或无关历史改动混入提交范围。Compacted conversation

已检查终端输出

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_AUVLJ2ZO109ALcsgb2ixl87i__vscode-1785496181120/content.txt)

已检查终端输出

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_V6HX0ONYGeAVJbXn3c5iGVVn__vscode-1785496181122/content.txt#129-129), lines 129 to 129

基础模型测试尚未返回最终摘要，但没有出现新的失败输出。我同时验证真实 `srb_tracking` batch 的归一化结果，并检查工作区诊断与 diff 空白；这几项彼此独立，可以并行收尾。

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import dataclasses; import numpy as np; from openpi.training.config import get_config; from RLtune import data_loader_rl; config=dataclasses.replace(get_config("physics_aware_srb_train_tracking"), num_workers=0); loader=data_loader_rl.create_data_loader(config, sharding=None, shuffle=False, num_batches=1, skip_norm_stats=False, episode_aware=True, reward_aggregation="sum"); observation, actions=next(iter(loader)); values=np.asarray(observation.physics_params); assert values.shape == (8, 5), values.shape; assert actions.shape == (8, 16, 19), actions.shape; assert np.isfinite(values).all(); assert np.max(np.abs(values)) < 1e-5, values; print("real physics-aware loader smoke test passed", values.shape, actions.shape, values.min(), values.max())'

Ran terminal command:  git diff --check

Checked [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/physics_aware_single_step.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/weight_loaders.py), no problems found

真实 batch 仍有一个物理维度残留 `0.00024216`，说明它的统计 `std` 并非严格为零，而是由保存精度产生了极小正值；测试中的 `std=0` 分支本身已通过。我要检查该维度的实际 stats，再把判定阈值对齐到这类量化噪声范围，避免把固定默认物理量误当成有效变化。

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py#1-1), lines 1 to 220

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking/norm_stats.json#1-1), lines 1 to 120

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize.py)

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking/norm_stats.json#120-120), lines 120 to 162

Ran terminal command:  git diff -- src/openpi/transforms.py src/openpi/transforms_test.py

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import numpy as np; from openpi.shared import normalize; stats=normalize.load("assets/srb_train_tracking/srb_tracking")["physics_params"]; x=np.asarray([[9.81,0.5,5.0,1.225,0.3]], dtype=np.float32); print("x", x); print("mean", stats.mean); print("std", stats.std); print("q01", stats.q01); print("q99", stats.q99); print("x-mean", x-stats.mean); print("z", (x-stats.mean)/(stats.std+1e-6))'

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms_test.py#1-1), lines 1 to 100

统计文件已确认：`q01 == q99`，但 `roughness` 的 `std=0.000122`，正是 `float32` 下用 `E[x²] - E[x]²` 的舍入误差；输入与均值只差 `2.98e-8`。更稳妥的根修是让 `RunningStats` 用 `float64` 累积均值、平方均值和直方图输入，而不是在归一化阶段凭经验放宽阈值。先用最小运行检查确认该判断，再改统计实现并重新生成 stats。

Ran terminal command:  uv run python -c 'import numpy as np; from openpi.shared.normalize import RunningStats; stats=RunningStats(); stats.update(np.full((1000, 1), 0.3, dtype=np.float32)); print(stats.get_statistics().std)'

Searched for regex `RunningStats\(` (`**/*.py`), 4 results

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize_test.py)

Generating patch (14 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize.py)


```

```


Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/shared/normalize_test.py src/openpi/transforms_test.py -k 'normalize or constant_stats'

复现与回归测试都支持这个根因：`RunningStats` 精度修复后 4 个相关测试通过。代码行为已经正确，我再去掉不必要的行内说明，然后重新生成被 gitignore 的本地 stats 文件，让真实 loader 使用修复后的统计结果。

Generating patch (6 lines) in [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize.py)


```

```


Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/shared/normalize_test.py src/openpi/transforms_test.py -k 'normalize or constant_stats'

Ran terminal command:  JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name physics_aware_srb_train_tracking

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'import dataclasses; import numpy as np; from openpi.training.config import get_config; from RLtune import data_loader_rl; config=dataclasses.replace(get_config("physics_aware_srb_train_tracking"), num_workers=0); loader=data_loader_rl.create_data_loader(config, sharding=None, shuffle=False, num_batches=1, skip_norm_stats=False, episode_aware=True, reward_aggregation="sum"); observation, actions=next(iter(loader)); values=np.asarray(observation.physics_params); assert values.shape == (8, 5), values.shape; assert actions.shape == (8, 16, 19), actions.shape; assert np.isfinite(values).all(); assert np.max(np.abs(values)) < 1e-5, values; print("real physics-aware loader smoke test passed", values.shape, actions.shape, values.min(), values.max())'

已检查终端输出

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_PWTBsbyMi34Woa5e0VLUJuQg__vscode-1785496181143/content.txt#1-1), lines 1 to 100

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_PWTBsbyMi34Woa5e0VLUJuQg__vscode-1785496181143/content.txt#100-100), lines 100 to 129

真实 loader 已通过：`physics_params` 为 `(8, 5)` 且全为 0，actions 仍为 `(8, 16, 19)`。我现在做最终改动盘点和诊断复查；模型测试进程仍在 CPU 编译阶段，若它继续没有结束，会按已知的 CPU/JAX 长耗时单独记录。

Ran terminal command:  git status --short && git diff --stat && git diff --check

Ran terminal command:  JAX_PLATFORMS=cpu uv run python -c 'from openpi.training.config import get_config; config=get_config("physics_aware_srb_train_tracking"); assert config.model.action_dim == 19; assert config.model.action_horizon == 16; assert config.model.physics_dim == 5; print("physics-aware config smoke test passed", config.name, config.model.action_dim, config.model.action_horizon, config.data.repo_id)'

Checked [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/shared/normalize.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/transforms.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/policies/srb_policy.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/config.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/models/physics_aware_single_step.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/RLtune/train_offline_jax.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/scripts/compute_norm_stats.py), [](file:///root/SpaceRobot/ACoT-VLA_saa/src/openpi/training/weight_loaders.py), no problems found

已检查终端输出

Read [](file:///root/.vscode-server/data/User/workspaceStorage/4c642311fa37b880f414d8da9e68930c/GitHub.copilot-chat/chat-session-resources/eb643887-821e-442e-86ca-9a20f151a8e0/call_WByFJPAAGYkcYO6utgoAKoZQ__vscode-1785496181149/content.txt#100-100), lines 100 to 129

Ran terminal command:  ps -eo pid,etime,stat,cmd | rg 'pytest|model_test|pi0_test'

Ran terminal command:  ps -eo pid,etime,stat,cmd | grep -E 'pytest|model_test|pi0_test' | grep -v grep

Ran terminal command:  ps -o pid,etime,pcpu,pmem,stat,wchan:32,cmd -p 1246817,1246820

后台模型回归已运行约 24 分钟，子进程持续占用约 4 个 CPU 核但没有推进到新的测试结果；这不是本次改动所需的窄验证。我会停止这个过宽的 CPU 测试进程，保留已经通过的 PhysicsAware dummy loss、配置、loader 和针对性测试结果，避免留下占用资源的后台任务。

已终止的终端

Ran terminal command:  JAX_PLATFORMS=cpu uv run pytest --strict-markers src/openpi/shared/normalize_test.py src/openpi/transforms_test.py

Ran terminal command:  uv run ruff check src/openpi/shared/normalize.py

Read [](file:///root/SpaceRobot/ACoT-VLA_saa/assets/srb_train_tracking/srb_tracking/norm_stats.json#130-130), lines 130 to 162

最终数据变换回归为 15/15 通过，Ruff 对 normalize.py 也通过；重新生成的 stats 中五个物理维度 `std` 全为 0，真实 loader 已验证零中心。最后确认后台进程已清理，并保留工作区状态供你查看。

Ran terminal command:  ps -eo pid,etime,stat,cmd | grep -E 'pytest.*model_test|pytest.*pi0_test' | grep -v grep || true

Ran terminal command:  git diff --check && git status --short

本次 PhysicsAware SRB 离线 GRPO 接入已完成并完成针对性验证：配置、数据 transform、兼容 checkpoint、物理参数归一化和真实 batch 通路均已接通；仅保留一个历史 state shape 测试失败和一个未完成的长时 CPU 模型回归说明。Autopilot extended tool call limit



已完成 `physics_aware_srb_train_tracking` 接入：

- PhysicsAware 模型、SRB transform、物理默认参数、可选字段 repack 已接通。
- action 契约保持 `19` 维、`16` horizon。
- 新增兼容 checkpoint loader，支持加载现有 SRB checkpoint。
- `RunningStats` 改用 `float64`，修复固定物理参数的伪方差。
- 真实 loader 验证通过：physics `(8, 5)` 全为 0，actions `(8, 16, 19)`。
- transform/normalize 测试 `15/15` 通过，配置 smoke test 和诊断检查通过。

相关改动见 config.py、srb_policy.py 和 normalize.py。

备注：历史 `test_srb_inputs_flatten_and_pad_state` 仍存在 `(32,)` 与 `(8,)` 的既有失败；长时间 CPU 模型回归因运行约 24 分钟无结果已停止，实际训练尚未执行。

已完成: *Add tests for new features* (2/2)

已进行更改。

需要check 一下 normalize

# Gfootball Multi-Agent VLA Training & Inference

基于 ACoT-VLA 框架的 Google Research Football 多智能体视觉-语言-动作 (VLA) 模型训练与推理方案。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACoT-VLA Multi-Agent Framework               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Agent 0     │    │  Agent 1     │    │  Agent N     │      │
│  │  (Player 0)  │    │  (Player 1)  │    │  (Player N)  │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Observation Encoder                     │       │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────────┐ │       │
│  │  │ Image   │  │ State (115) │  │ Prompt (Role)   │ │       │
│  │  │ 72×96×3 │  │ simple115   │  │ "left team GK"  │ │       │
│  │  └────┬────┘  └──────┬──────┘  └────────┬────────┘ │       │
│  └───────┼───────────────┼──────────────────┼──────────┘       │
│          │               │                  │                   │
│          ▼               ▼                  ▼                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              ACoT-VLA / PI0-FAST Model               │       │
│  │                                                      │       │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │       │
│  │  │ SigLIP   │  │ PaliGemma    │  │ Action Expert │  │       │
│  │  │ Vision   │→ │ LLM          │→ │ (Gemma 300M)  │  │       │
│  │  │ Encoder  │  │ (Gemma 2B)   │  │               │  │       │
│  │  └──────────┘  └──────────────┘  └───────────────┘  │       │
│  │                                                      │       │
│  │  ┌──────────────────────────────────────────────┐   │       │
│  │  │ Action Reasoning (Optional)                  │   │       │
│  │  │ • Explicit: Cross-attention (fine ↔ coarse)  │   │       │
│  │  │ • Implicit: LLM hidden state extraction      │   │       │
│  │  └──────────────────────────────────────────────┘   │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│                  Action Distribution                            │
│                  (19 discrete actions)                          │
│                                                                 │
│  Actions:                                                       │
│  0=idle, 1-8=move, 9=long_pass, 10=high_pass, 11=short_pass   │
│  12=shot, 13=sprint, 14-15=release, 16=slide, 17-18=dribble   │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
gfootball/
├── README_gfootball_vla.md          # 本文档
├── train.py                         # (空) 原始训练占位
├── train_gfootball.sh               # VLA 训练启动脚本
├── convert_gfootball_data_to_lerobot.py  # 数据采集与格式转换
├── serve_gfootball.py               # 多智能体推理与评估

src/openpi/policies/
├── gfootball_policy.py              # Gfootball 数据转换策略
│   ├── GfootballInputs              # 输入转换 (obs → model input)
│   ├── GfootballOutputs             # 输出转换 (model output → action)
│   ├── GfootballMultiAgentInputs    # 多智能体批量处理
│   ├── ACOTGfootballConfig          # ACoT-VLA gfootball 专用配置
│   └── build_simple115_obs()        # 观测编码工具函数

src/openpi/training/config.py        # 已注册的训练配置
├── pi0_fast_gfootball_5v5           # PI0-FAST + LoRA (5v5)
├── acot_gfootball_5v5_reasoning     # ACoT-VLA + 推理 (5v5)
├── pi0_fast_gfootball_11v11         # PI0-FAST 全量微调 (11v11)
└── debug_gfootball                  # 调试配置
```

## 快速开始

### 1. 环境准备

```bash
# 安装 ACoT-VLA (在项目根目录)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 安装 gfootball 环境依赖
pip install gfootball pygame numpy
```

### 2. 数据采集

使用现有策略（随机/启发式/MAPPO）采集训练数据：

```bash
cd gfootball

# 使用启发式策略采集 5v5 数据
python convert_gfootball_data_to_lerobot.py \
    --output_dir ../dataset/gfootball_lerobot \
    --n_episodes 1000 \
    --scenario football_5v5_malib \
    --policy heuristic \
    --team_id 0 \
    --seed 42

# 使用随机策略采集（快速测试）
python convert_gfootball_data_to_lerobot.py \
    --output_dir ../dataset/gfootball_lerobot \
    --n_episodes 100 \
    --scenario football_5v5_malib \
    --policy random

# 使用 MAPPO 预训练策略采集（更高质量数据）
python convert_gfootball_data_to_lerobot.py \
    --output_dir ../dataset/gfootball_lerobot \
    --n_episodes 500 \
    --scenario football_5v5_malib \
    --policy mappo \
    --policy_path /home/omnisky/Desktop/algos/gfootball/agents/football_5v5_mappo
```

数据格式说明：
```
dataset/gfootball_lerobot/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── episodes.jsonl         # 每集元数据
│   ├── episodes_stats.jsonl   # 每集统计量
│   └── tasks.jsonl            # 任务描述
├── data/chunk-000/
│   └── episode_XXXXXX.hdf5    # 每集数据
└── ../assets/gfootball_train/gfootball_dataset/
    └── norm_stats.json        # 归一化统计量
```

### 3. 计算归一化统计量

```bash
cd /path/to/ACoT-VLA_saa

uv run python scripts/compute_norm_stats.py --config-name pi0_fast_gfootball_5v5
```

### 4. 训练

#### 方案 A: PI0-FAST + LoRA（推荐，适合离散动作）

```bash
cd gfootball

# 使用训练脚本
bash train_gfootball.sh pi0_fast_gfootball_5v5 gfootball_5v5_exp

# 或直接运行
cd ..
uv run python scripts/train.py --config pi0_fast_gfootball_5v5 --exp-name gfootball_5v5_exp
```

#### 方案 B: ACoT-VLA + 推理（适合需要战术推理的场景）

```bash
# 显式+隐式动作推理
bash train_gfootball.sh acot_gfootball_5v5_reasoning gfootball_5v5_reasoning_exp
```

#### 方案 C: 11v11 全量微调

```bash
bash train_gfootball.sh pi0_fast_gfootball_11v11 gfootball_11v11_exp
```

### 5. 评估与推理

```bash
cd gfootball

# 运行 50 局比赛（VLA vs 启发式策略）
python serve_gfootball.py \
    --config pi0_fast_gfootball_5v5 \
    --checkpoint_dir ../checkpoints/pi0_fast_gfootball_5v5/gfootball_5v5_exp/30000 \
    --scenario football_5v5_malib \
    --n_episodes 50 \
    --opponent_policy heuristic \
    --save_results results.json

# 运行并渲染
python serve_gfootball.py \
    --config pi0_fast_gfootball_5v5 \
    --checkpoint_dir ../checkpoints/pi0_fast_gfootball_5v5/gfootball_5v5_exp/30000 \
    --render

# 启动 WebSocket 推理服务器
python serve_gfootball.py \
    --config pi0_fast_gfootball_5v5 \
    --checkpoint_dir ../checkpoints/pi0_fast_gfootball_5v5/gfootball_5v5_exp/30000 \
    --serve --port 8000
```

## 训练配置详解

| 配置名 | 模型 | 动作空间 | 场景 | 特点 |
|--------|------|----------|------|------|
| `pi0_fast_gfootball_5v5` | PI0-FAST + LoRA | Discrete(19) | 5v5 | 推荐，高效离散动作处理 |
| `acot_gfootball_5v5_reasoning` | ACoT-VLA + LoRA | Discrete(19) | 5v5 | 显式+隐式战术推理 |
| `pi0_fast_gfootball_11v11` | PI0-FAST 全量 | Discrete(19) | 11v11 | 大规模场景 |
| `debug_gfootball` | PI0-FAST dummy | Discrete(19) | - | 快速调试 |

## 观测空间

每个智能体获得独立的观测字典：

```python
observation = {
    "obs": {
        "ball":              [x, y, z],        # 球位置
        "ball_direction":    [x, y, z],        # 球移动方向
        "ball_owned_team":   {-1, 0, 1},       # 球权
        "left_team":         [[x,y], ...],     # 左队位置 (11个)
        "right_team":        [[x,y], ...],     # 右队位置 (11个)
        "active":            int,              # 当前控制球员
        "game_mode":         int,              # 比赛模式
        "score":             [left, right],    # 比分
        "steps_left":        int,              # 剩余步数
    },
    "controlled_player_index": int  # 全局球员ID
}
```

VLA 模型输入：
- **Image**: 渲染帧 (72×96×3) → 缩放至 224×224
- **State**: simple115_v2 向量 (115维)
- **Prompt**: 角色描述 (如 "left team goalkeeper: defend the goal")

## 动作空间

19 个离散足球动作：

| ID | 动作 | 类别 |
|----|------|------|
| 0 | idle | 静止 |
| 1-8 | left ~ bottom_left | 8方向移动 |
| 9 | long_pass | 长传 |
| 10 | high_pass | 高空传球 |
| 11 | short_pass | 短传 |
| 12 | shot | 射门 |
| 13 | sprint | 冲刺 |
| 14 | release_move | 停止移动 |
| 15 | release_sprint | 停止冲刺 |
| 16 | slide | 铲球 |
| 17 | dribble | 带球 |
| 18 | release_dribble | 停止带球 |

## 多智能体设计

### 训练阶段
- 每个球员的 (观测, 动作) 对独立作为训练样本
- 使用角色相关的 prompt 区分不同位置的球员
- 共享同一个 VLA 模型参数

### 推理阶段
- 每个球员独立查询 VLA 模型获取动作
- 支持批量推理以提高效率
- 可与任意对手策略（随机/启发式/MAPPO/SAC）对战

### 角色 Prompt 设计

```python
# 守门员
"left team goalkeeper: defend the goal and clear the ball"

# 后卫
"left team center_back: defend and build up play from the back"

# 中场
"left team central_midfielder: control the midfield and distribute the ball"

# 前锋
"left team striker: score goals and lead the attack"
```

## 高级用法

### 自定义训练配置

在 `src/openpi/training/config.py` 的 `_CONFIGS` 列表中添加新配置：

```python
TrainConfig(
    name="my_gfootball_config",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=19,
        action_horizon=1,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=GfootballDataConfig(
        repo_id="my_gfootball_dataset",
        base_config=DataConfig(prompt_from_task=True),
        action_dim=19,
        n_agents=4,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    num_train_steps=50_000,
    batch_size=32,
)
```

### 自定义奖励引导的数据采集

修改 `convert_gfootball_data_to_lerobot.py` 中的奖励函数：

```python
def custom_reward(obs, action, prev_obs):
    """自定义奖励函数，引导数据采集策略"""
    reward = 0.0
    # 进球奖励
    if obs["score"][0] > prev_obs["score"][0]:
        reward += 10.0
    # 控球奖励
    if obs["ball_owned_team"] == 0:
        reward += 0.5
    # 射门奖励
    if action == 12:  # shot
        reward += 0.3
    return reward
```

### 远程推理服务

```bash
# 启动服务器
python serve_gfootball.py \
    --config pi0_fast_gfootball_5v5 \
    --checkpoint_dir ./checkpoints/... \
    --serve --port 8000

# 客户端连接 (Python)
import websockets
import asyncio
import json

async def query_actions(observations):
    async with websockets.connect("ws://localhost:8000") as ws:
        await ws.send(json.dumps({
            "type": "get_actions",
            "observations": observations,
            "task_instruction": "attack and score"
        }))
        response = json.loads(await ws.recv())
        return response["actions"]
```

## 故障排除

### 常见问题

1. **`GIT_LFS_SKIP_SMUDGE=1` 必须设置**
   ```bash
   export GIT_LFS_SKIP_SMUDGE=1
   uv sync
   ```

2. **GPU 内存不足**
   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # 降低内存使用
   ```

3. **gfootball 导入失败**
   ```bash
   pip install gfootball
   # 或
   pip install --no-cache-dir gfootball
   ```

4. **数据集路径问题**
   确保 `assets/gfootball_train/gfootball_dataset/norm_stats.json` 存在

## 参考

- [Google Research Football](https://github.com/google-research/football)
- [ACoT-VLA](./AGENTS.md)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

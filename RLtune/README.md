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

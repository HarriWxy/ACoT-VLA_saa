# Space Robotics Bench

这个目录把 ACoT-VLA / OpenPI 的标准 VLA 推理与训练流程接到了 Space Robotics Bench。

## 已接入的内容

- `src/openpi/policies/srb_policy.py`
  - 把 SRB 的 flat observation dict 转成模型输入。
  - 兼容 `state` / `proprio`，也兼容视觉任务里的 `image_cam_base` / `image_cam_wrist`。
  - 如果当前 SRB 任务没有相机观测，会自动补零图像，所以非视觉任务也能跑通同一套接口。
- `src/openpi/training/config.py`
  - 增加了 `SRBDataConfig`。
  - 提供了两个模板 config：`pi05_srb` 和 `pi0_fast_srb`。
- `examples/srb/serve.py`
  - 简化版 websocket policy server，避免直接处理 `scripts/serve_policy.py` 的 union CLI。
- `examples/srb/main.py`
  - 直接在 SRB 环境里 rollout，并通过 websocket 请求 policy action chunk。

## 推理

前提：

- 你运行这个脚本的 Python 环境必须已经能正常 import `srb`、`isaaclab`、`isaacsim`。
- 同一个环境里还要能 import 当前仓库的 `openpi` 和 `openpi_client`。
- 推荐直接在 SRB 的 Isaac Sim 环境里执行 `uv pip install -e /media/omnisky/sda/algos/R2A/Algos/ACoT-VLA_saa`。

启动 policy server：

```bash
uv run python examples/srb/serve.py \
  --config-name pi05_srb \
  --checkpoint-dir checkpoints/pi05_srb/<EXP_NAME>/<STEP>
```

启动 SRB rollout：

```bash
uv run python examples/srb/main.py \
  --env-id srb/sample_collection_visual \
  --prompt "collect the sample" \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0
```

说明：

- 对 manipulation visual 任务，SRB 会返回 `image_cam_base` 和 `image_cam_wrist`，adapter 会把它们映射到模型的 base / wrist image slots。
- 对非 visual 任务，adapter 会自动补零图像；这能跑通接口，但效果会明显依赖你训练时是否也是无视觉输入。
- 当前脚本把 SRB 环境固定成 `num_envs=1`，因为 OpenPI 的 policy server 接口是单条 observation 推理。

## 训练

### 1. 数据格式

训练仍然走 OpenPI 原本的 LeRobot 数据管线。你的 SRB 数据集建议保存这些 key：

- `proprio`: 机器人本体低维状态，推荐优先保留。
- `state`: 任务相关状态。
- `state_dyn`: 可选，动态状态。
- `proprio_dyn`: 可选，动态本体状态。
- `image_cam_base`: 可选，SRB base camera RGB。
- `image_cam_wrist`: 可选，SRB wrist camera RGB。
- `actions`: 环境动作，形状应为 `(action_dim,)`。
- `task`: 任务文本。配合 `prompt_from_task=True` 自动生成 prompt。

当前模板 config 默认：

- 输出动作维度 `action_dim=7`
- 状态拼接顺序 `("proprio", "state")`
- 图像 key `("image_cam_base", "image_cam_wrist")`

如果你的 SRB 任务动作维度不是 7，或者你想把 `state_dyn` / `proprio_dyn` 也拼进去，需要在 `SRBDataConfig` 里改这些字段。

### 2. 归一化统计

```bash
uv run python scripts/compute_norm_stats.py --config-name pi05_srb
```

如果你把数据集 repo id 改成自己的，需要同时覆盖对应字段，例如：

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_srb \
  --data.repo-id your_hf_username/srb_manipulation
```

### 3. 启动训练

```bash
uv run python scripts/train.py \
  pi05_srb \
  --exp-name srb_pi05 \
  --data.repo-id your_hf_username/srb_manipulation \
  --overwrite
```

也可以使用 FAST 模型：

```bash
uv run python scripts/train.py \
  pi0_fast_srb \
  --exp-name srb_fast \
  --data.repo-id your_hf_username/srb_manipulation \
  --overwrite
```

## 重要限制

- 这次接入的是标准 VLA 训练 / 推理路径，不是 ACoT-VLA 的 `coarse_actions` 三路训练路径。
- 当前 `SRBInputs` 会把选中的状态向量裁到模型 `action_dim`，再做 zero padding。
- 对 `pi05_srb`，模型内部 state 维度是 32。如果你拼接后的 SRB 状态超过 32 维，超出的部分会被截断。
- 如果你不想接受截断，可以在 `SRBDataConfig(..., strict_state_dim=True)` 下显式报错，然后自己重新选 state keys。

## 推荐做法

- manipulation 任务优先用 visual 版本环境，例如 `*_visual`，这样训练和推理都能得到稳定的 base / wrist 视角。
- 状态优先级建议是 `proprio` > `state` > `proprio_dyn` > `state_dyn`。
- 如果你后面要把 SRB 接到 ACoT-VLA 的 coarse action 训练，需要继续扩展 data loader，让 batch 从 `(obs, actions)` 变成 `(obs, actions, coarse_actions)`。
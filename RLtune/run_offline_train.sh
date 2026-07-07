#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Offline RL fine-tuning for ACoT-VLA using GRPO
# ──────────────────────────────────────────────────────────────
#
# This script trains with pre-collected offline data from
# dataset/srb_tracking instead of online environment rollouts.
#
# Usage:
#   # Single GPU:
#   bash RLtune/run_offline_train.sh
#
#   # Multi-GPU (4x):
#   bash RLtune/run_offline_train.sh 4
#
#   # Custom checkpoint dir:
#   CKPT_DIR=checkpoints/my_sft bash RLtune/run_offline_train.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurable parameters ──
NPROC="${1:-1}"                                    # Number of GPUs
CKPT_DIR="${CKPT_DIR:-checkpoints/srb_train_gemma4}"  # SFT checkpoint directory
DATA_DIR="${DATA_DIR:-dataset/srb_tracking}"        # Offline data directory
CONFIG_NAME="${CONFIG_NAME:-srb_train_gemma4}"      # Base training config
SEED="${SEED:-42}"                                  # Random seed
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"                 # Training epochs
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"            # Steps per epoch
LR="${LR:-5e-6}"                                    # Learning rate
N_SAMPLES="${N_SAMPLES:-8}"                         # Episodes per GRPO group
FRAMES_PER_EP="${FRAMES_PER_EP:-2}"                 # Frames sampled per episode per step
REWARD_AGG="${REWARD_AGG:-sum}"                     # Reward aggregation: sum | mean | last | binary_sum
FILTER_ACC="${FILTER_ACC:-true}"                    # Enable accuracy filtering

# ── XLA / CUDA settings ──
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TOKENIZERS_PARALLELISM=false

# ── Build filter flag ──
FILTER_FLAG=""
if [ "${FILTER_ACC}" = "true" ]; then
    FILTER_FLAG="--filter-by-accuracy"
else
    FILTER_FLAG="--no-filter-by-accuracy"
fi

# ── Common args ──
COMMON_ARGS=(
    --data-dir "${DATA_DIR}"
    --checkpoint-dir "${CKPT_DIR}"
    --config-name "${CONFIG_NAME}"
    --seed "${SEED}"
    --total-epochs "${TOTAL_EPOCHS}"
    --num-train-steps-per-epoch "${STEPS_PER_EPOCH}"
    --learning-rate "${LR}"
    --n-samples "${N_SAMPLES}"
    --frames-per-episode "${FRAMES_PER_EP}"
    --reward-aggregation "${REWARD_AGG}"
    --action-dim 37
    --action-horizon 16
    ${FILTER_FLAG}
)

# ── Launch ──
echo "╔══════════════════════════════════════════════════╗"
echo "║  Offline RL Fine-tuning (GRPO)                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Data:       ${DATA_DIR}"
echo "║  Checkpoint: ${CKPT_DIR}"
echo "║  GPUs:       ${NPROC}"
echo "║  Epochs:     ${TOTAL_EPOCHS} x ${STEPS_PER_EPOCH} steps"
echo "║  LR:         ${LR}"
echo "║  Group size: ${N_SAMPLES} episodes"
echo "║  Reward agg: ${REWARD_AGG}"
echo "╚══════════════════════════════════════════════════╝"

cd "$(dirname "$0")/.."

if [ "${NPROC}" -gt 1 ]; then
    echo "Launching with torchrun (${NPROC} GPUs)..."
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
        -m RLtune.train_offline \
        "${COMMON_ARGS[@]}"
else
    echo "Launching single GPU..."
    python -m RLtune.train_offline \
        "${COMMON_ARGS[@]}"
fi

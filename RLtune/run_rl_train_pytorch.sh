#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PyTorch RL fine-tuning launch script for ACoT-VLA
#
# Usage:
#   # Single GPU:
#   bash RLtune/run_rl_train_pytorch.sh srb_train path/to/sft_checkpoint
#
#   # Multi-GPU (DDP), auto-detect GPUs:
#   bash RLtune/run_rl_train_pytorch.sh srb_train path/to/sft_checkpoint --multi_gpu
#
#   # Specify number of GPUs:
#   bash RLtune/run_rl_train_pytorch.sh srb_train path/to/sft_checkpoint --multi_gpu --num_gpus 4
#
#   # With custom hyperparameters:
#   bash RLtune/run_rl_train_pytorch.sh srb_train path/to/sft_checkpoint \
#       --n_samples 16 --learning_rate 1e-5 --total_epochs 50
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Default configuration ──
CONFIG_NAME="${1:-srb_train}"
CHECKPOINT_DIR="${2:-}"
shift 2 2>/dev/null || true

# ── Parse extra flags ──
MULTI_GPU=false
NUM_GPUS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi_gpu)
            MULTI_GPU=true
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ── Environment variables ──
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Detect GPUs ──
if [ -z "${NUM_GPUS}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# ── Print header ──
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      ACoT-VLA RL Fine-tuning (GRPO) — PyTorch              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config:     ${CONFIG_NAME}"
echo "║  Checkpoint: ${CHECKPOINT_DIR:-'auto'}"
echo "║  Multi-GPU:  ${MULTI_GPU} (${NUM_GPUS} GPUs)"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Build command ──
if [ "${MULTI_GPU}" = true ] && [ "${NUM_GPUS}" -gt 1 ]; then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS}"
    CMD+=" -m RLtune.train_pytorch"
else
    CMD="python -m RLtune.train_pytorch"
fi

CMD+=" --config_name ${CONFIG_NAME}"

if [ -n "${CHECKPOINT_DIR}" ]; then
    CMD+=" --checkpoint_dir ${CHECKPOINT_DIR}"
fi

# Pass through additional arguments
if [ -n "${EXTRA_ARGS}" ]; then
    CMD+=" ${EXTRA_ARGS}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

# ── Execute ──
eval ${CMD}

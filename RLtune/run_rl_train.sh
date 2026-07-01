#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# RL fine-tuning launch script for ACoT-VLA
#
# Usage:
#   bash RLtune/run_rl_train.sh [CONFIG_NAME] [CHECKPOINT_DIR] [OPTIONS]
#
# Examples:
#   # Basic usage with SRB config
#   bash RLtune/run_rl_train.sh srb_train path/to/sft_checkpoint
#
#   # With custom hyperparameters
#   bash RLtune/run_rl_train.sh srb_train path/to/sft_checkpoint \
#       --n_samples 16 --learning_rate 1e-5 --total_epochs 50
#
#   # Resume from checkpoint
#   bash RLtune/run_rl_train.sh srb_train path/to/rl_checkpoint
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Default configuration ──
CONFIG_NAME="${1:-srb_train}"
CHECKPOINT_DIR="${2:-}"
shift 2 2>/dev/null || true

# ── Environment variables ──
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TOKENIZERS_PARALLELISM=true

# ── Run RL training ──
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ACoT-VLA RL Fine-tuning (GRPO)                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config:     ${CONFIG_NAME}"
echo "║  Checkpoint: ${CHECKPOINT_DIR:-'auto'}"
echo "╚══════════════════════════════════════════════════════════════╝"

# Build command
CMD="python -m RLtune.train"
CMD+=" --config_name ${CONFIG_NAME}"

if [ -n "${CHECKPOINT_DIR}" ]; then
    CMD+=" --checkpoint_dir ${CHECKPOINT_DIR}"
fi

# Pass through any additional arguments
if [ $# -gt 0 ]; then
    CMD+=" $@"
fi

echo "Running: ${CMD}"
exec ${CMD}

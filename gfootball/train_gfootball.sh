#!/bin/bash
# Gfootball multi-agent VLA training script.
#
# Usage:
#   # Step 1: Collect data
#   python convert_gfootball_data_to_lerobot.py \
#       --output_dir ./dataset/gfootball_lerobot \
#       --n_episodes 1000 --scenario football_5v5_malib --policy heuristic
#
#   # Step 2: Compute normalization stats
#   uv run python scripts/compute_norm_stats.py --config-name pi0_fast_gfootball_5v5
#
#   # Step 3: Train (this script)
#   bash train_gfootball.sh pi0_fast_gfootball_5v5 gfootball_5v5_exp
#
#   # Step 4: Evaluate
#   python serve_gfootball.py \
#       --config pi0_fast_gfootball_5v5 \
#       --checkpoint_dir ./checkpoints/pi0_fast_gfootball_5v5/gfootball_5v5_exp/30000 \
#       --n_episodes 50 --render

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
CONFIG_NAME="${1:?Usage: $0 <CONFIG_NAME> <EXP_NAME>}"
EXP_NAME="${2:-gfootball_vla}"

# GPU configuration
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Memory configuration
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRAC:-0.85}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_ALLOCATOR:-platform}"

# JAX configuration
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu,gpu}"

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train.py"

# ── Validation ─────────────────────────────────────────────────────────
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Train script not found at ${TRAIN_SCRIPT}"
    echo "Make sure you're running this from the gfootball/ directory"
    exit 1
fi

# ── Info ───────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Gfootball Multi-Agent VLA Training                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config:    ${CONFIG_NAME}"
echo "║  Experiment: ${EXP_NAME}"
echo "║  GPU:       ${CUDA_VISIBLE_DEVICES}"
echo "║  JAX Mem:   ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Run Training ───────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

# Override exp_name via CLI
uv run python "${TRAIN_SCRIPT}" \
    --config "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    "$@"

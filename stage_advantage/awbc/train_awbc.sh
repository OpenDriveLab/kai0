#!/bin/bash
###############################################################################
# train_awbc.sh
#
# Train a policy with Advantage-Weighted Behavior Cloning (AWBC) using
# advantage-estimator-labeled data. The data must have task_index per frame and
# meta/tasks.jsonl mapping task_index -> prompt string (from Stage 0 + Stage 2).
#
# Configs (see src/openpi/training/config.py):
#   pi05_flatten_fold_awbc
#   pi05_tee_shirt_sort_awbc
#   pi05_hang_cloth_awbc
#
# Prerequisites:
#   - Complete Stage 0 (GT labeling) and Stage 2 (advantage estimation on data),
#     then run gt_label.py with --advantage-source absolute_advantage to produce
#     the "advantage" dataset with task_index and tasks.jsonl.
#   - Set repo_id in the AWBC config to the path of that dataset
#     (e.g. <path_to_repo_root>/data/FlattenFold/advantage).
#   - Run compute_norm_states_fast.py for the chosen config before training.
#   - Set weight_loader in config to your π₀.5 base checkpoint.
#
# Usage:
#   RUNNAME=pi05_flatten_fold_awbc RUNTIME=run1 bash stage_advantage/awbc/train_awbc.sh
###############################################################################
set -xe
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${PROJECT_ROOT}"

source .venv/bin/activate

# ─── Training config name ────────────────────────────────────────────────────
# RUNNAME must be one of: pi05_flatten_fold_awbc, pi05_tee_shirt_sort_awbc, pi05_hang_cloth_awbc
CFG=${RUNNAME:-pi05_flatten_fold_awbc}

# ─── Validate required environment variables ─────────────────────────────────
if [ -z "${RUNNAME+x}" ]; then
    echo "[WARNING] RUNNAME is not set, using default: ${CFG}"
    export RUNNAME=${CFG}
else
    echo "RUNNAME is set to: ${RUNNAME}"
fi

if [ -z "${RUNTIME+x}" ]; then
    echo "[ERROR] RUNTIME is not set. Please set RUNTIME for experiment output directory."
    echo "  Example: RUNTIME=run1 bash stage_advantage/awbc/train_awbc.sh"
    exit 1
else
    echo "RUNTIME is set to: ${RUNTIME}"
fi

# ─── Output directories ─────────────────────────────────────────────────────
OUTPUT_DIR="./experiment/${RUNNAME}"
LOG_OUTPUT_DIR="${OUTPUT_DIR}/log"
mkdir -p "${OUTPUT_DIR}" "${LOG_OUTPUT_DIR}"

export WANDB_MODE=${WANDB_MODE:-offline}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

# ─── Launch JAX training ────────────────────────────────────────────────────
echo "Launching AWBC training (JAX)..."
uv run scripts/train.py ${CFG} \
    --exp_name=${RUNTIME} \
    2>&1 | tee "${LOG_OUTPUT_DIR}/${RUNTIME}.log"

echo "============================================================"
echo "  AWBC training finished. Checkpoints: ${OUTPUT_DIR}/${RUNTIME}/"
echo "============================================================"

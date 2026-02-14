#!/bin/bash
###############################################################################
# train_estimator.sh
###########################################################
set -xe
set -o pipefail

# ─── Navigate to project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${PROJECT_ROOT}"

source .venv/bin/activate

# ─── Training config name ────────────────────────────────────────────────────
# RUNNAME must be one of: ADVANTAGE_TORCH_PI06_FLATTEN_FOLD, ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD
# Default to ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD if RUNNAME is not set
CFG=${RUNNAME:-ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD}

# ─── DDP environment variables ───────────────────────────────────────────────
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-12345}

# ─── Validate required environment variables ─────────────────────────────────
if [ -z "${RUNNAME+x}" ]; then
    echo "[WARNING] RUNNAME is not set, using default: ${CFG}"
    export RUNNAME=${CFG}
else
    echo "RUNNAME is set to: ${RUNNAME}"
fi

if [ -z "${RUNTIME+x}" ]; then
    echo "[ERROR] RUNTIME is not set. Please set RUNTIME for experiment output directory."
    echo "  Example: RUNTIME=run1 bash train_estimator.sh"
    exit 1
else
    echo "RUNTIME is set to: ${RUNTIME}"
fi

# ─── Create output directories ───────────────────────────────────────────────
OUTPUT_DIR="./experiment/${RUNNAME}"
LOG_OUTPUT_DIR="${OUTPUT_DIR}/log"
mkdir -p "${OUTPUT_DIR}" "${LOG_OUTPUT_DIR}"

# Set to "offline" for offline logging; remove or set to "online" for cloud sync
export WANDB_MODE=${WANDB_MODE:-offline}

if [ "${NPROC_PER_NODE}" -gt 1 ] || [ "${WORLD_SIZE}" -gt 1 ]; then
    # Multi-GPU / Multi-Node training via torchrun
    echo "Launching DDP training with torchrun..."
    uv run torchrun \
        --nnodes=${WORLD_SIZE} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        scripts/train_pytorch.py ${CFG} \
        --exp_name=${RUNTIME} \
        --save_interval 10000 \
        2>&1 | tee "${LOG_OUTPUT_DIR}/${RUNTIME}.log"
else
    # Single-GPU training
    echo "Launching single-GPU training..."
    uv run python scripts/train_pytorch.py ${CFG} \
        --exp_name=${RUNTIME} \
        --save_interval 10000 \
        2>&1 | tee "${LOG_OUTPUT_DIR}/${RUNTIME}.log"
fi

#!/bin/bash
###############################################################################
# eval.sh
#
# Use a trained Advantage Estimator to label a dataset with predicted
# advantage values (relative_advantage, absolute_value, absolute_advantage).
#
# This script calls eval.py, which:
#   1. Loads a trained Advantage Estimator checkpoint
#   2. Iterates over all episodes in the LeRobot dataset
#   3. Reads video frames from three camera views (top, left, right)
#   4. Runs batched GPU inference to predict advantage values per frame
#   5. Writes results as new parquet files with advantage columns appended
#
# The output parquets are saved under:
#   <repo_id>/data_<model_name>_<ckpt_steps>/chunk-*/episode_*.parquet
#
# Prerequisites:
#   - A trained Advantage Estimator checkpoint (from Stage 1)
#   - Update MODELS_CONFIG_MAP in eval.py with the correct checkpoint paths
#
# Usage:
#   bash eval.sh <model_type> <model_name> <repo_id>
#
# Examples:
#   bash eval.sh Flatten-Fold KAI0 /path/to/dataset
#   bash eval.sh Flatten-Fold PI06 /path/to/dataset
#
# Arguments:
#   model_type : Flatten-Fold / demo_A / demo_B
#   model_name : PI06 (single-timestep) / KAI0 (two-timestep stage-level)
#   repo_id    : Path to the LeRobot dataset to evaluate
###############################################################################
set -xe
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

# ─── Conda / venv activation ─────────────────────────────────────────────────
source /cpfs01/shared/smch/miniconda3/etc/profile.d/conda.sh
conda activate uv_py311
source .venv/bin/activate

export TZ='Asia/Shanghai'

# ─── Other environment variables ──────────────────────────────────────────────
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export WANDB_MODE=offline

# ─── Parse arguments ─────────────────────────────────────────────────────────
MODEL_TYPE=${1:?"Usage: bash eval.sh <model_type> <model_name> <repo_id>"}
MODEL_NAME=${2:?"Usage: bash eval.sh <model_type> <model_name> <repo_id>"}
REPO_ID=${3:?"Usage: bash eval.sh <model_type> <model_name> <repo_id>"}

echo "============================================================"
echo "  Advantage Estimator Evaluation"
echo "  Model type:  ${MODEL_TYPE}"
echo "  Model name:  ${MODEL_NAME}"
echo "  Dataset:     ${REPO_ID}"
echo "============================================================"

uv run python "${SCRIPT_DIR}/eval.py" "${MODEL_TYPE}" "${MODEL_NAME}" "${REPO_ID}"

echo "============================================================"
echo "  Evaluation complete!"
echo "  Results saved under: ${REPO_ID}/data_${MODEL_NAME}_*/"
echo "============================================================"
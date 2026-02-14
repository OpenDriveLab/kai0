#!/bin/bash
###############################################################################
# Prepare advantage-labeled datasets for training the Advantage Estimator.
###############################################################################
set -xe
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Source dataset path (modify this to your dataset) ────────────────────────
DATA_PATH="Path/to/your/dataset"

# ─── Output base directory ────────────────────────────────────────────────────
base_name=$(basename "$DATA_PATH")
dir_name=$(dirname "$DATA_PATH")/${base_name}_advantage_data

# ─── Helper function: prepare dataset and run labeling ────────────────────────
prepare_and_label() {
    local data_subdir=$1      # source data subfolder name (e.g. data_PI06_100000 or data_KAI0_100000)
    local output_name=$2      # output dataset name suffix
    local extra_args=$3       # extra arguments for gt_label.py
    local target_path="${dir_name}/${output_name}"

    echo "============================================================"
    echo "  Preparing: ${output_name}"
    echo "  Source:    ${DATA_PATH}/${data_subdir}"
    echo "  Target:    ${target_path}"
    echo "============================================================"

    mkdir -p "${target_path}"

    # Symlink videos (shared, read-only)
    ln -sfn "${DATA_PATH}/videos" "${target_path}/videos"

    # Copy norm_stats and meta (will be modified by gt_label.py)
    cp -f "${DATA_PATH}/norm_stats.json" "${target_path}/norm_stats.json"
    cp -rf "${DATA_PATH}/meta" "${target_path}/meta"

    # Copy data parquets into the standard "data" directory
    if [ -d "${target_path}/data" ]; then
        rm -rf "${target_path}/data"
    fi
    cp -r "${DATA_PATH}/${data_subdir}" "${target_path}/data"

    # Run gt_label.py to assign task_index and update tasks.jsonl
    python "${SCRIPT_DIR}/gt_label.py" "${target_path}" \
        --threshold 30 \
        --chunk-size 50 \
        --discretion-type binary \
        --advantage-source absolute_advantage \
        ${extra_args}

    echo "  Done: ${output_name}"
    echo ""
}

# ─── Dataset variants (only PI06 and KAI0) ─────────────────────────────────────
# PI06: single-timestep / timeline-style labeling (1 stage)
prepare_and_label "data_1T_TL_100000" "${base_name}_PI06_binary" ""

# KAI0: two-stage, stage-level labeling
prepare_and_label "data_2T_SL_100000" "${base_name}_KAI0_abs_binary" "--stage-nums 2"

echo "============================================================"
echo "  All datasets labeled successfully!"
echo ""
echo "  Output directory: ${dir_name}"
echo ""
echo "  Next step: update the training config in config.py with"
echo "  the target dataset path, then run train_estimator.sh"
echo "============================================================"

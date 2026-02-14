## Annotation: Stage 0–2 (Labeling, Estimator Training, Eval)

This directory contains **Stage 0** (GT labeling with `gt_label.py` / `gt_labeling.sh`), **Stage 1** (advantage estimator training via `scripts/train_pytorch.py`), and **Stage 2** (advantage estimation on new data via `eval.py`). All commands below assume you are at the **repository root** unless noted. Full pipeline and options are in the [parent README](../README.md).

### Quick Start

```bash
# Step 1: Label a dataset with advantage-based task_index (GT labels from progress)
# Edit DATA_PATH in gt_labeling.sh, then from repo root:
bash stage_advantage/annotation/gt_labeling.sh

# Step 2: Train the Advantage Estimator (update config.py repo_id / pytorch_weight_path first)
# From repo root:
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1 --save_interval 10000
# Or: uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_PI06_FLATTEN_FOLD --exp_name=run1 --save_interval 10000

# Step 3: Evaluate the trained estimator on new data (PI06 or KAI0)
# From repo root:
uv run python stage_advantage/annotation/eval.py Flatten-Fold KAI0 /path/to/dataset

# Step 4: Use the advantage-labeled data for AWBC (Stage 3)
# After Stage 2, run gt_labeling.sh with DATA_PATH = eval repo (or gt_label.py --advantage-source absolute_advantage).
# Then from repo root:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
```

### File Descriptions

| File | Stage | Description |
|---|---|---|
| `gt_label.py` | 0 | Core script: computes advantage from progress/absolute_advantage and assigns `task_index` to parquet frames |
| `gt_labeling.sh` | 0 | Batch labeling: prepares dataset dirs and runs `gt_label.py` (only .sh in this dir) |
| `eval.py` | 2 | Evaluates a trained estimator on a dataset, writing predicted advantages to new parquets |
| `evaluator.py` | 2 | `SimpleValueEvaluator`: batched GPU inference with parallel video loading and prefetching |

For Stage 0 parameters, Stage 1 config fields, Stage 2 `MODELS_CONFIG_MAP`, and end-to-end AWBC order, see the [parent README](../README.md).

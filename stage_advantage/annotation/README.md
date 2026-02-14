## Annotation: GT Data Labeling, Advantage Estimator Training & Evaluation

This directory handles **Stage 0** (GT data labeling), **Stage 1** (advantage estimator training), and **Stage 2** (advantage estimation on new data).

### Quick Start

```bash
# Step 1: Label a dataset with advantage-based task_index (GT labels from progress)
bash gt_labeling.sh

# Step 2: Train the Advantage Estimator (update config.py paths first!)
# Use ADVANTAGE_TORCH_PI06_FLATTEN_FOLD or ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 bash train_estimator.sh

# Step 3: Evaluate the trained estimator on new data (PI06 or KAI0)
bash eval.sh Flatten-Fold KAI0 /path/to/dataset

# Step 4: Use the advantage-labeled data for AWBC (Stage 3)
# Run gt_label.py with --advantage-source absolute_advantage on the Stage 2 output,
# then: XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
```

### File Descriptions

| File | Stage | Description |
|---|---|---|
| `gt_label.py` | 0 | Core script: computes advantage from progress and assigns `task_index` to parquet frames |
| `gt_labeling.sh` | 0 | Batch labeling script: prepares dataset directories and runs `gt_label.py` |
| `train_estimator.sh` | 1 | Launches PyTorch training of the Advantage Estimator (single/multi-GPU) |
| `eval.py` | 2 | Evaluates a trained estimator on a dataset, writing predicted advantages to new parquets |
| `eval.sh` | 2 | Shell wrapper for `eval.py` with environment setup |
| `evaluator.py` | 2 | `SimpleValueEvaluator` class: batched GPU inference with parallel video loading and prefetching |

See the [parent README](../README.md) for the full pipeline overview.

## Annotation: Steps 1–3 (Estimator Training, Eval, Discretize)

This directory contains **Step 1** (advantage estimator training via `scripts/train_pytorch.py`), **Step 2** (advantage prediction on data via `eval.py`), and **Step 3** (discretize advantages into positive/negative via `discretize_advantage.py`). All commands below assume you are at the **repository root** unless noted. Full pipeline and options are in the [parent README](../README.md).

### Quick Start

```bash
# Step 1: Train the Advantage Estimator (update config.py repo_id / pytorch_weight_path first)
uv run python scripts/train_pytorch.py ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD --exp_name=run1 --save_interval 10000

# Step 2: Predict advantages on a dataset (update MODELS_CONFIG_MAP in eval.py first)
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset

# Step 3: Discretize advantages into positive/negative task_index labels
# Edit DATA_PATH in discretize_advantage.sh, then:
bash stage_advantage/annotation/discretize_advantage.sh

# Step 4: AWBC training (see awbc/README.md)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
```

### File Descriptions

| File | Step | Description |
|---|---|---|
| `discretize_advantage.py` | 3 | Reads advantage columns, bins into positive/negative `task_index`, writes `meta/tasks.jsonl` |
| `discretize_advantage.sh` | 3 | Batch wrapper: prepares dataset dirs and runs `discretize_advantage.py` for PI06/KAI0 variants |
| `eval.py` | 2 | Predicts advantage values on a dataset using a trained estimator |
| `evaluator.py` | 2 | `SimpleValueEvaluator`: batched GPU inference with parallel video loading and prefetching |

Step 1 training commands and Step 0 (manual annotation) are documented in the [parent README](../README.md).

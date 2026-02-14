# Stage Advantage Pipeline

This module implements a two-stage pipeline for training an **Advantage Estimator** and using it in **Advantage-Weighted Behavior Cloning (AWBC)**.

## Pipeline Overview

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Stage 0: GT Labeling (annotation/gt_labeling.sh + gt_label.py)         │
 │  Compute advantage from progress and assign task_index labels           │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Stage 1: Train Advantage Estimator (annotation/train_estimator.sh)     │
 │  Fine-tune pi0 model to predict advantage from observations             │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Stage 2: Advantage Estimation on New Data (annotation/eval.py)         │
 │  Use trained estimator to label new datasets with advantage values      │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Stage 3: AWBC Training (awbc/train_awbc.sh)                            │
 │  Train policy with advantage-weighted behavior cloning                   │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 0: GT Data Labeling

**Goal**: Compute advantage values from raw trajectory progress and label each frame with a discretized `task_index`.

**Script**: `annotation/gt_labeling.sh` (calls `annotation/gt_label.py`)

### How it works

1. **Prepare dataset directory**: Copy/link the raw dataset (parquet + videos + meta) into a new working directory with standard LeRobot layout.
2. **Compute advantage**: For each frame `i`, the advantage is defined as:
   ```
   advantage[i] = progress[i + chunk_size] - progress[i]
   ```
   where `chunk_size` defaults to 50 frames. For frames near the end of an episode, a normalized extrapolation is used.
3. **Discretize into task_index**: Based on the advantage distribution across the entire dataset:
   - **Binary mode** (`--discretion-type binary`): Frames in the top `threshold%` get `task_index=1`, the rest get `task_index=0`.
   - **N-slices mode** (`--discretion-type n_slices`): Frames are divided into `n` equal-percentile bins, each assigned `task_index` from `0` to `n-1`.
4. **Stage-aware labeling** (`--stage-nums > 1`): Divides frames by their `stage_progress_gt` value into stages, then computes independent percentile boundaries per stage.
5. **Write back**: Updates `task_index` column in each parquet file and writes `meta/tasks.jsonl`.

### Required Source Data Columns

The source parquet files must contain these columns for the full pipeline to work:

| Column | Required By | Description |
|---|---|---|
| `progress` / `absolute_advantage` / `relative_advantage` | `gt_label.py` | Used to compute advantage values |
| `stage_progress_gt` | `AdvantageLerobotDataset` | Stage progress ground truth (0-1), used for random timestep comparison |
| `progress_gt` | Training config repack_transforms | Progress ground truth, mapped as model input |
| `observation.state` | Training config | Robot state |
| `action` | Training config | Robot action sequence |
| `episode_index`, `frame_index` | LeRobot format | Standard metadata |

### Usage

```bash
cd stage_advantage/annotation

# Example: binary labeling using absolute_advantage as the advantage source
python gt_label.py <dataset_path> \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage

# Example: 2-stage binary labeling
python gt_label.py <dataset_path> \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage \
    --stage-nums 2

# Dry run (only print statistics, do not modify files)
python gt_label.py <dataset_path> --dry-run
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--threshold` | 70.0 | Top percentile for positive advantage (binary mode) |
| `--chunk-size` | 50 | Number of frames to look ahead for progress diff |
| `--discretion-type` | `binary` | `binary` or `n_slices` |
| `--n-slices` | 10 | Number of slices (only for `n_slices` mode) |
| `--advantage-source` | `progress` | `progress`, `absolute_advantage`, or `relative_advantage` |
| `--stage-nums` | 1 | Number of stages to divide data by `stage_progress_gt` |
| `--dry-run` | false | Only compute and print statistics without modifying files |

See `gt_labeling.sh` for batch labeling examples across multiple dataset variants.

---

## Stage 1: Train Advantage Estimator

**Goal**: Fine-tune a pi0-based model to predict advantage values from observations (images + state), producing a learned Advantage Estimator.

**Script**: `annotation/train_estimator.sh`

**Configs**: `ADVANTAGE_TORCH_PI06_FLATTEN_FOLD` or `ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD` (defined in `src/openpi/training/config.py`)

### How it works

1. The training uses `scripts/train_pytorch.py`, which supports single-GPU and multi-GPU (DDP) training via `torchrun`.
2. The model architecture is `AdvantageEstimator` (defined in `src/openpi/models_pytorch/pi0_pytorch.py`), initialized from a pre-trained pi0.5 checkpoint (`pytorch_weight_path`).
3. The model is trained to regress advantage/progress values:
   - `loss_value_weight=1.0` (value prediction loss is active)
   - `loss_action_weight=0.0` (action prediction loss is disabled)
4. `skip_norm_stats=True` since the advantage estimator does not require normalization statistics.
5. Data is loaded via `AdvantageLerobotDataset` which:
   - Reads `task_index` to get the task prompt string
   - Samples a random same-episode comparison frame (prefixed with `his_-100_`)
   - Computes `progress = stage_progress_gt - his_-100_stage_progress_gt` as the regression target

### Before Training

1. **Complete Stage 0** to get a labeled dataset.
2. **Update config.py** with the correct paths:

```python
# In src/openpi/training/config.py, find ADVANTAGE_TORCH_PI06_FLATTEN_FOLD or ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD:
TrainConfig(
    name="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",  # or ADVANTAGE_TORCH_PI06_FLATTEN_FOLD
    data=LerobotAgilexDataConfig(
        repo_id="<your_labeled_dataset_path>",          # <-- update this
        assets=AssetsConfig(
            assets_dir="<your_labeled_dataset_path>/assets",  # <-- update this
            asset_id="<your_dataset_name>",              # <-- update this
        ),
    ),
    pytorch_weight_path="<path_to_pi05_base_checkpoint>",  # <-- update this
    ...
)
```

### Usage

```bash
# Single GPU (KAI0 or PI06)
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 \
    bash stage_advantage/annotation/train_estimator.sh
RUNNAME=ADVANTAGE_TORCH_PI06_FLATTEN_FOLD RUNTIME=run1 \
    bash stage_advantage/annotation/train_estimator.sh

# Multi-GPU (8 GPUs on a single node)
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 NPROC_PER_NODE=8 \
    bash stage_advantage/annotation/train_estimator.sh

# Multi-Node (2 nodes x 8 GPUs)
# On node 0 (master):
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 \
    WORLD_SIZE=2 RANK=0 NPROC_PER_NODE=8 \
    MASTER_ADDR=<master_ip> MASTER_PORT=12345 \
    bash stage_advantage/annotation/train_estimator.sh

# On node 1:
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 \
    WORLD_SIZE=2 RANK=1 NPROC_PER_NODE=8 \
    MASTER_ADDR=<master_ip> MASTER_PORT=12345 \
    bash stage_advantage/annotation/train_estimator.sh

# Resume from a previous checkpoint
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD RUNTIME=run1 RESUME=1 \
    bash stage_advantage/annotation/train_estimator.sh
```

### Training Outputs

```
experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/   # or ADVANTAGE_TORCH_PI06_FLATTEN_FOLD
  ├── <exp_name>/
  │     ├── 10000/           # checkpoint at step 10000
  │     │     ├── model.safetensors
  │     │     ├── optimizer.pt
  │     │     ├── metadata.pt
  │     │     └── assets/
  │     ├── 20000/           # checkpoint at step 20000
  │     └── ...
  └── log/
        └── <exp_name>.log
```

---

## Stage 2: Advantage Estimation on New Data

**Goal**: Use the trained Advantage Estimator to label new/unseen datasets with predicted advantage values.

**Script**: `annotation/eval.sh` (calls `annotation/eval.py`, which uses `annotation/evaluator.py`)

### How it works

1. Loads a trained Advantage Estimator checkpoint (from Stage 1).
2. Iterates over all episodes in the target LeRobot dataset.
3. For each episode, reads video frames from three camera views (top_head, hand_left, hand_right).
4. Runs batched GPU inference with parallel data prefetching to predict per-frame advantage values.
5. Writes results as new parquet files with advantage columns appended:
   - `relative_advantage`: Predicted progress difference between frame n and frame n+50 (2-timestep mode only).
   - `absolute_value`: Predicted cumulative progress from the initial frame to frame n.
   - `absolute_advantage`: Difference of absolute values between frame n+50 and frame n, clipped to [-1, 1].

### Model Variants

| Variant | Description |
|---|---|
| `PI06` | Single-timestep (absolute value only) |
| `KAI0` | Two-timestep, stage-level progress (relative + absolute advantage) |

### Before Evaluation

1. **Complete Stage 1** to get a trained Advantage Estimator checkpoint.
2. **Update `MODELS_CONFIG_MAP`** in `eval.py` with the correct `ckpt_dir` and `ckpt_steps` for your trained model.

### Usage

```bash
# Evaluate using the Flatten-Fold KAI0 model on a dataset
bash stage_advantage/annotation/eval.sh Flatten-Fold KAI0 /path/to/dataset

# Evaluate using the PI06 model
bash stage_advantage/annotation/eval.sh Flatten-Fold PI06 /path/to/dataset

# Or call eval.py directly
python stage_advantage/annotation/eval.py Flatten-Fold KAI0 /path/to/dataset
```

### Evaluation Outputs

Results are saved alongside the original data directory:

```
<repo_id>/
  ├── data/                             # Original data (unchanged)
  │     chunk-000/
  │         episode_000000.parquet
  │         ...
  ├── data_KAI0_100000/                # New parquets with advantage columns (or data_PI06_100000)
  │     chunk-000/
  │         episode_000000.parquet      # = original + relative_advantage, absolute_value, absolute_advantage
  │         ...
  └── videos/                           # Shared videos (unchanged)
```

The output parquets can then be used in Stage 3 (AWBC) or fed back into Stage 0 (`gt_label.py --advantage-source absolute_advantage`) for discretized labeling.

---

## Stage 3: AWBC Training

**Goal**: Train a policy using **Advantage-Weighted Behavior Cloning (AWBC)**. The advantage labels (from Stage 0 + Stage 2) are stored as `task_index` per frame and as prompt strings in `meta/tasks.jsonl`. By setting **`prompt_from_task=True`** in the data config, each sample’s prompt is taken from that mapping, so the policy is conditioned on the advantage-derived label (e.g. high vs low advantage) and effectively does advantage-weighted behavior cloning via the language channel.

**Configs** (in `src/openpi/training/config.py`): `pi05_flatten_fold_awbc`, `pi05_tee_shirt_sort_awbc`, `pi05_hang_cloth_awbc`. Each uses `LerobotAgilexDataConfig` or `LerobotARXDataConfig` with `base_config=DataConfig(prompt_from_task=True)` and `repo_id` pointing to the **advantage** dataset (e.g. `.../data/FlattenFold/advantage`).

### What the policy sees as prompt (training)

The prompt is read from the dataset’s **`meta/tasks.jsonl`**: each frame’s `task_index` is mapped to a task string, and that string is passed to the policy as the language prompt. **`gt_label.py`** (Stage 0) writes these strings when it builds the advantage-labeled dataset.

- **Binary mode** (typical): `task_index=0` → `"<task>, Advantage: negative"`, `task_index=1` → `"<task>, Advantage: positive"`. The `<task>` text is set in `gt_label.py` (e.g. `"fold the cloth"` for FlattenFold).
- **n_slices mode**: `task_index=i` → `"<task>, Advantage: {i}"`.

So during AWBC training the model is conditioned on prompts that explicitly include the advantage label (e.g. `"fold the cloth, Advantage: positive"` or `"fold the cloth, Advantage: negative"`).

### Inference with an AWBC-trained model

At **inference** time you must use the **same prompt format** as in training. To run the policy in the high-advantage regime, pass the **positive**-advantage prompt, e.g. `"<task>, Advantage: positive"` (with the same `<task>` wording as in your `tasks.jsonl`). Using a different format or omitting the advantage part can hurt performance, since the model was trained to condition on this exact style of prompt.

### How it works (data flow)

1. **Data**: The advantage dataset must contain `task_index` in each parquet and `meta/tasks.jsonl` mapping `task_index` → prompt string. This is produced by running Stage 2 (eval) to get advantage columns, then Stage 0 (`gt_label.py --advantage-source absolute_advantage`) to discretize into `task_index` and write `tasks.jsonl`.
2. **Config**: `prompt_from_task=True` causes the data loader to wrap the dataset with `PromptFromLeRobotTask(dataset_meta.tasks)`, which sets `prompt = tasks[task_index]` for each sample. The repack transform includes `"prompt"` so the policy receives this text as conditioning.
3. **Training**: Standard JAX training via `scripts/train.py` with the AWBC config; the policy is trained with the task-derived prompt, so the language input carries the advantage weighting.

### Before training

1. Produce the advantage dataset (Stage 0 + Stage 2) and place it at e.g. `./data/FlattenFold/advantage`.
2. In `config.py`, set **`repo_id`** to that path and **`weight_loader`** to your π₀.5 base checkpoint for the three AWBC configs you use.
3. Compute norm stats:  
   `uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc`  
   (and similarly for `pi05_tee_shirt_sort_awbc` / `pi05_hang_cloth_awbc` if needed.)

### Usage

From the repository root, run JAX training with the AWBC config and an experiment name:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tee_shirt_sort_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_hang_cloth_awbc --exp_name=run1
```

---

## Directory Structure

```
stage_advantage/
├── README.md                          # This file
├── annotation/                        # Stages 0–2: labeling & estimator training
│   ├── README.md
│   ├── gt_label.py                    # Core labeling script (progress → advantage → task_index)
│   ├── gt_labeling.sh                 # Batch labeling for PI06 / KAI0 variants
│   ├── train_estimator.sh             # Training script for the Advantage Estimator
│   ├── eval.py                        # Evaluate trained estimator on datasets
│   ├── eval.sh                        # Shell wrapper for eval.py
│   └── evaluator.py                   # SimpleValueEvaluator: batched GPU inference
└── awbc/                              # Stage 3: AWBC (see Usage above; optional train_awbc.sh)
    ├── README.md
    └── train_awbc.sh
```

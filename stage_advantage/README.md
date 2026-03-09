# Stage Advantage Pipeline

This module implements a pipeline for training an **Advantage Estimator** and using it in **Advantage-Weighted Behavior Cloning (AWBC)**.

## Pipeline Overview

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Step 0: Annotate stage_progress_gt (manual)          │
 │  Label subtask boundaries and per-frame stage progress in parquets       │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Step 1: Train Advantage Estimator (scripts/train_pytorch.py)            │
 │  Fine-tune pi0 model to predict advantage from observations              │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Step 2: Predict Advantage (annotation/eval.py)                          │
 │  Use trained estimator → parquets with absolute_advantage / relative_adv │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Step 3: Discretize Advantage (annotation/discretize_advantage.py)       │
 │  Bin advantages into positive/negative → task_index + tasks.jsonl        │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  Step 4: AWBC Training (scripts/train.py pi05_*_awbc)                    │
 │  Train policy with advantage-weighted behavior cloning (prompt_from_task) │
 └──────────────────────────────────────────────────────────────────────────┘
```

**End-to-end order for AWBC:** Step 0 → Step 1 → Step 2 → Step 3 → Step 4.

**Pre-annotated data:** The released dataset includes **`Task_A/advantage/`**, a fully annotated advantage dataset that can be used **directly for AWBC training** (Step 4) without running Step 0–3. It is available in both the [Hugging Face](https://huggingface.co/datasets/OpenDriveLab-org/Kai0) and [ModelScope](https://www.modelscope.cn/datasets/OpenDriveLab/Kai0) dataset repos. After downloading (e.g. via `scripts/download_dataset.py`), set the AWBC config `repo_id` to the local path (e.g. `<repo_root>/data/Task_A/advantage`) and run training.

---

## Step 0: Annotate `stage_progress_gt` (manual)

**Goal**: Provide per-frame **stage progress ground truth** (`stage_progress_gt`) in the parquet files. This is the supervision signal used by the Advantage Estimator in Step 1.

The procedure is:

1. **Mark episode boundaries and subtask timestamps.** For each episode, annotate:
   - The start and end timestamps of the episode.
   - The split timestamps between subtasks (e.g. for a 2-stage task: the time that separates stage 1 from stage 2).

2. **Compute per-frame `stage_progress_gt`.** For each subtask segment, linearly interpolate progress from 0 to 1 within that segment. Frames in stage `k` (of `K` total stages) get:
   ```
   stage_progress_gt = k/K + (1/K) * (frame_position_within_stage / segment_length)
   ```
   So `stage_progress_gt` ranges from 0.0 (start of the first stage) to 1.0 (end of the last stage).

3. **Write `stage_progress_gt` into the parquet files.** Add the `stage_progress_gt` column to each episode's parquet alongside `observation.state`, `action`, etc.

The resulting dataset (with `stage_progress_gt`) is the input for Step 1 (advantage estimator training).

---

## Step 1: Train Advantage Estimator

**Goal**: Fine-tune a pi0-based model to predict advantage values from observations (images + state), producing a learned Advantage Estimator.

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

1. **Complete Step 0** to get a dataset with `stage_progress_gt`.
2. **Update config.py** with the correct paths:

```python
TrainConfig(
    name="ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD",  # or ADVANTAGE_TORCH_PI06_FLATTEN_FOLD
    data=LerobotAgilexDataConfig(
        repo_id="<your_labeled_dataset_path>",          # <-- update this
        assets=AssetsConfig(
            assets_dir="<your_labeled_dataset_path>/assets",
            asset_id="<your_dataset_name>",
        ),
    ),
    pytorch_weight_path="<path_to_pi05_base_checkpoint>",  # <-- update this
    ...
)
```

### Usage

From the **repository root** with venv activated:

```bash
source .venv/bin/activate
export WANDB_MODE=${WANDB_MODE:-offline}
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD   # or ADVANTAGE_TORCH_PI06_FLATTEN_FOLD
RUNTIME=run1
mkdir -p "./experiment/${RUNNAME}/log"

# Single-GPU
uv run python scripts/train_pytorch.py ${RUNNAME} --exp_name=${RUNTIME} --save_interval 10000 \
    2>&1 | tee "./experiment/${RUNNAME}/log/${RUNTIME}.log"

# Multi-GPU (e.g. 8 GPUs)
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py ${RUNNAME} \
    --exp_name=${RUNTIME} --save_interval 10000 \
    2>&1 | tee "./experiment/${RUNNAME}/log/${RUNTIME}.log"

# Resume from latest checkpoint
uv run python scripts/train_pytorch.py ${RUNNAME} --exp_name=${RUNTIME} --resume
```

### Training Outputs

```
experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/
  ├── <exp_name>/
  │     ├── 10000/           # checkpoint at step 10000
  │     │     ├── model.safetensors
  │     │     ├── optimizer.pt
  │     │     ├── metadata.pt
  │     │     └── assets/
  │     ├── 20000/
  │     └── ...
  └── log/
        └── <exp_name>.log
```

---

## Step 2: Predict Advantage (Advantage Estimation on Data)

**Goal**: Use the trained Advantage Estimator (from Step 1) to label a dataset with predicted advantage values (`absolute_advantage`, `relative_advantage`, `absolute_value`).

**Script**: `annotation/eval.py` (uses `annotation/evaluator.py`)

### How it works

1. Loads a trained Advantage Estimator checkpoint (from Step 1).
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

1. **Complete Step 1** to get a trained Advantage Estimator checkpoint.
2. **Update `MODELS_CONFIG_MAP`** in `eval.py` with the correct `ckpt_dir` and `ckpt_steps` for your trained model.

### Usage

From the **repository root** with venv activated:

```bash
source .venv/bin/activate
uv run python stage_advantage/annotation/eval.py <model_type> <model_name> <repo_id>
```

Examples:

```bash
uv run python stage_advantage/annotation/eval.py Task-A KAI0 /path/to/dataset
uv run python stage_advantage/annotation/eval.py Task-A PI06 /path/to/dataset
```

`<model_type>` is a key in `eval.py`'s `MODELS_CONFIG_MAP` (e.g. `Task-A`); `<model_name>` is `PI06` or `KAI0`; `<repo_id>` is the path to the LeRobot dataset.

### Evaluation Outputs

Results are saved alongside the original data directory:

```
<repo_id>/
  ├── data/                             # Original data (unchanged)
  │     chunk-000/
  │         episode_000000.parquet
  ├── data_KAI0_100000/                 # New parquets with advantage columns
  │     chunk-000/
  │         episode_000000.parquet      # = original + relative_advantage, absolute_value, absolute_advantage
  └── videos/                           # Shared videos (unchanged)
```

---

## Step 3: Discretize Advantage

**Goal**: Take the predicted advantages from Step 2 (`absolute_advantage` or `relative_advantage`) and discretize them into binary (positive / negative) or n-slice `task_index` labels. This produces the dataset format needed for AWBC (Step 4): each frame gets a `task_index`, and `meta/tasks.jsonl` maps each `task_index` to a prompt string.

**Script**: `annotation/discretize_advantage.py` (batch wrapper: `annotation/discretize_advantage.sh`)

### How it works

1. **Prepare dataset directory**: Copy/link the Step 2 output (parquet with advantage columns + videos + meta) into a new working directory with standard LeRobot layout.
2. **Read advantage values**: For each frame, read the advantage value from the specified source column (`absolute_advantage` or `relative_advantage`).
3. **Discretize into task_index**: Based on the advantage distribution across the entire dataset:
   - **Binary mode** (`--discretion-type binary`): Frames in the top `threshold%` get `task_index=1` (positive), the rest get `task_index=0` (negative).
   - **N-slices mode** (`--discretion-type n_slices`): Frames are divided into `n` equal-percentile bins, each assigned `task_index` from `0` to `n-1`.
4. **Stage-aware labeling** (`--stage-nums > 1`): Divides frames by their `stage_progress_gt` value into stages, then computes independent percentile boundaries per stage.
5. **Write back**: Updates `task_index` column in each parquet file and writes `meta/tasks.jsonl`.

### Usage

```bash
cd stage_advantage/annotation

# Binary labeling using absolute_advantage
python discretize_advantage.py <dataset_path> \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage

# 2-stage binary labeling
python discretize_advantage.py <dataset_path> \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --advantage-source absolute_advantage \
    --stage-nums 2

# Dry run (only print statistics, do not modify files)
python discretize_advantage.py <dataset_path> --dry-run
```

For batch labeling across PI06 and KAI0 variants, see `discretize_advantage.sh`:

```bash
# Edit DATA_PATH in discretize_advantage.sh first (point to your Step 2 output repo)
bash stage_advantage/annotation/discretize_advantage.sh
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--threshold` | 70.0 | Top percentile for positive advantage (binary mode) |
| `--chunk-size` | 50 | Chunk size (kept for compatibility; not used by current advantage sources) |
| `--discretion-type` | `binary` | `binary` or `n_slices` |
| `--n-slices` | 10 | Number of slices (only for `n_slices` mode) |
| `--advantage-source` | `absolute_advantage` | `absolute_advantage` or `relative_advantage` |
| `--stage-nums` | 1 | Number of stages to divide data by `stage_progress_gt` |
| `--dry-run` | false | Only compute and print statistics without modifying files |

---

## Step 4: AWBC Training

**Goal**: Train a policy using **Advantage-Weighted Behavior Cloning (AWBC)**. The advantage labels (from Step 3) are stored as `task_index` per frame and as prompt strings in `meta/tasks.jsonl`. By setting **`prompt_from_task=True`** in the data config, each sample's prompt is taken from that mapping, so the policy is conditioned on the advantage-derived label (e.g. high vs low advantage) and effectively does advantage-weighted behavior cloning via the language channel.

**Configs** (in `src/openpi/training/config.py`): `pi05_flatten_fold_awbc`, `pi05_tee_shirt_sort_awbc`, `pi05_hang_cloth_awbc`. Each uses `LerobotAgilexDataConfig` or `LerobotARXDataConfig` with `base_config=DataConfig(prompt_from_task=True)` and `repo_id` pointing to the **advantage** dataset (e.g. `.../data/Task_A/advantage`).

### What the policy sees as prompt (training)

The prompt is read from the dataset's **`meta/tasks.jsonl`**: each frame's `task_index` is mapped to a task string, and that string is passed to the policy as the language prompt. **`discretize_advantage.py`** (Step 3) writes these strings when it builds the advantage-labeled dataset.

- **Binary mode** (typical): `task_index=0` → `"<task>, Advantage: negative"`, `task_index=1` → `"<task>, Advantage: positive"` (e.g. `"fold the cloth, Advantage: positive"`). The `<task>` text is defined in `annotation/discretize_advantage.py`.
- **n_slices mode**: `task_index=i` → `"<task>, Advantage: {i}"`.

### Inference with an AWBC-trained model

At **inference** time you must use the **same prompt format** as in training. To run the policy in the high-advantage regime, pass the **positive**-advantage prompt, e.g. `"<task>, Advantage: positive"` (with the same `<task>` wording as in your `tasks.jsonl`). Using a different format or omitting the advantage part can hurt performance, since the model was trained to condition on this exact style of prompt.

**Where to set the prompt when deploying:** The language prompt is set in the **inference code** (e.g. the `lang_embeddings` variable in the Agilex inference scripts). See the [train_deploy_alignment/inference README](../train_deploy_alignment/inference/README.md) and [Agilex README — Prompt and AWBC](../train_deploy_alignment/inference/agilex/README.md#prompt-and-awbc-important) for how to configure it.

### Before training

1. **Produce the advantage dataset:** Run Step 2 (eval) to get `data_PI06_100000/` or `data_KAI0_100000/` with advantage columns. Then run Step 3 (`discretize_advantage.sh` or `discretize_advantage.py --advantage-source absolute_advantage`); the script outputs a directory with `data/` (parquets with `task_index`), `meta/tasks.jsonl`, and `videos`. Use that directory as the advantage dataset (e.g. copy or link it to `./data/Task_A/advantage`).
2. In `config.py`, set **`repo_id`** to that advantage dataset path and **`weight_loader`** to your π₀.5 base checkpoint for the AWBC config(s) you use.
3. **Compute norm stats:**  
   `uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc`  
   (and similarly for `pi05_tee_shirt_sort_awbc` / `pi05_hang_cloth_awbc` if needed.)

### Usage

See [awbc/README.md](awbc/README.md) for the full command with env vars and log redirection.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tee_shirt_sort_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_hang_cloth_awbc --exp_name=run1
```

---

## Required Source Data Columns

The source parquet files must contain these columns for the full pipeline to work:

| Column | Required By | Description |
|---|---|---|
| `stage_progress_gt` | Step 0 output, Step 1 training | Stage progress ground truth (0–1), annotated manually |
| `absolute_advantage` / `relative_advantage` | Step 2 output → Step 3 input | Predicted by the advantage estimator |
| `observation.state` | Training configs | Robot state |
| `action` | Training configs | Robot action sequence |
| `episode_index`, `frame_index` | LeRobot format | Standard metadata |

---

## Directory Structure

```
stage_advantage/
├── README.md                          # This file
├── annotation/                        # Steps 0–3: labeling, estimator, eval, discretize
│   ├── README.md
│   ├── discretize_advantage.py        # Step 3: discretize advantage → task_index (positive/negative)
│   ├── discretize_advantage.sh        # Step 3: batch wrapper for PI06/KAI0 variants
│   ├── eval.py                        # Step 2: predict advantages with trained estimator
│   └── evaluator.py                   # SimpleValueEvaluator: batched GPU inference
└── awbc/                              # Step 4: AWBC
    └── README.md                      # Training commands (env + log redirection in README)
```

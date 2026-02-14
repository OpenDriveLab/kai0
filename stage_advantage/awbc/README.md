# Stage 3: AWBC (Advantage-Weighted Behavior Cloning)

Train a policy on **advantage-labeled** data so that the prompt conditions the policy on the advantage bin (e.g. high vs low advantage). This is implemented by setting **`prompt_from_task=True`** in the data config: each sample’s `task_index` is mapped to a prompt string via `meta/tasks.jsonl`, and that prompt is fed to the policy as language conditioning. Full pipeline (Stage 0 → 1 → 2 → 0 → 3) is in the [parent README](../README.md).

## Configs

All three are defined in `src/openpi/training/config.py`:

| Config name | Task | Data config |
|-------------|------|-------------|
| `pi05_flatten_fold_awbc` | FlattenFold | `LerobotAgilexDataConfig`, `repo_id=.../data/FlattenFold/advantage` |
| `pi05_tee_shirt_sort_awbc` | TeeShirtSort | `LerobotAgilexDataConfig`, `repo_id=.../data/TeeShirtSort/advantage` |
| `pi05_hang_cloth_awbc` | HangCloth | `LerobotARXDataConfig`, `repo_id=.../data/HangCloth/advantage` |

Each uses `base_config=DataConfig(prompt_from_task=True)` so that the dataset’s `task_index` column and `meta/tasks.jsonl` supply the prompt (advantage-derived label) per frame.

## Prerequisites

1. **Advantage dataset**  
   The data must have `task_index` in each parquet and `meta/tasks.jsonl` (prompt strings per `task_index`). To build it:
   - Run **Stage 2** (eval) on your dataset → get `data_PI06_100000/` or `data_KAI0_100000/` with advantage columns.
   - Run **Stage 0** on that output: `gt_label.py --advantage-source absolute_advantage` (or `gt_labeling.sh` with `DATA_PATH` = the eval repo). The resulting directory (with `data/`, `meta/tasks.jsonl`, `videos/`) is your advantage dataset.
   - Place or link it at e.g. `./data/FlattenFold/advantage` and set `repo_id` in config to that path.

2. **Config paths**  
   In `src/openpi/training/config.py`, for the AWBC config(s) you use:
   - Set **`repo_id`** to the **absolute path** of the advantage dataset (e.g. `<path_to_repo_root>/data/FlattenFold/advantage`).
   - Set **`weight_loader`** to your **π₀.5 base checkpoint** path.

3. **Norm stats**  
   From the repo root, run:
   ```bash
   uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_awbc
   ```
   (Repeat for `pi05_tee_shirt_sort_awbc` or `pi05_hang_cloth_awbc` if you train those.)

## Usage

From the **repository root**, the core training command is:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tee_shirt_sort_awbc --exp_name=run1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_hang_cloth_awbc --exp_name=run1
```

Checkpoints and logs are written under `experiment/<config_name>/<exp_name>/` (e.g. `experiment/pi05_flatten_fold_awbc/run1/`).

For a ready-to-use script with environment setup (venv activation, `XLA_PYTHON_CLIENT_MEM_FRACTION`, `WANDB_MODE`) and automatic log management, see **`train_awbc.sh`**:

```bash
RUNNAME=pi05_flatten_fold_awbc RUNTIME=run1 bash stage_advantage/awbc/train_awbc.sh
```

The shell script handles output directory creation and log redirection (via `tee`) automatically.

## Prompt format (training and inference)

During **training**, the prompt is taken from **`meta/tasks.jsonl`**: each sample’s `task_index` is mapped to a string (written by `gt_label.py` when creating the advantage dataset).

- **Binary mode**: `task_index=0` → `"<task>, Advantage: negative"`, `task_index=1` → `"<task>, Advantage: positive"` (e.g. `"fold the cloth, Advantage: positive"`). The `<task>` text is defined in `annotation/gt_label.py`.
- **n_slices mode**: `task_index=i` → `"<task>, Advantage: {i}"`.

At **inference**, use the **same format** so the model sees the conditioning it was trained on. To get high-advantage behavior, pass the **positive**-advantage prompt, e.g. `"<task>, Advantage: positive"` (with the same `<task>` wording as in your `tasks.jsonl`). Using a different prompt format or omitting the advantage part can hurt performance.



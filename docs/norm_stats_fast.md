## Fast normalization stats computation (`compute_norm_states_fast.py`)

This script provides a **fast path** to compute normalization statistics for Kai0 configs by
directly reading local parquet files instead of going through the full data loader. It produces
`norm_stats` that are **compatible with the original openpi pipeline** (same `RunningStats`
implementation and batching scheme).

---

### When to use this script

- You have already **downloaded the dataset locally** (e.g. under `./data`, see
  [`docs/dataset.md`](./dataset.md#step-1-download-the-dataset)).
- You have a **training config** in `src/openpi/training/config.py` (e.g.
  `pi05_flatten_fold_normal`) and you want to compute `norm_stats` before running
  `scripts/train.py`.
- You prefer a **simpler / faster** pipeline compared to the original `compute_norm_stats.py`
  while keeping numerically compatible statistics.

---

### Script entry point

The script lives at:

- `scripts/compute_norm_states_fast.py`

Main entry:

- `main(config_name: str, base_dir: str | None = None, max_frames: int | None = None)`

CLI is handled via [`tyro`](https://github.com/brentyi/tyro), so you call it from the repo root as:

```bash
uv run python scripts/compute_norm_states_fast.py --config-name <config_name> [--base-dir <path>] [--max-frames N]
```

---

### Arguments

- **`--config-name`** (`str`, required)
  - Name of the TrainConfig defined in `src/openpi/training/config.py`, e.g.:
    - `pi05_flatten_fold_normal`
    - `pi05_tee_shirt_sort_normal`
    - `pi05_hang_cloth_normal`
  - Internally resolved via `_config.get_config(config_name)`.

- **`--base-dir`** (`str`, optional)
  - Base directory containing the parquet data for this config.
  - If omitted, the script will read it from `config.data`:
    - `data_config = config.data.create(config.assets_dirs, config.model)`
    - `base_dir` defaults to `data_config.repo_id`
  - This means you can either:
    - Set `repo_id` in the config to your local dataset path (e.g.
      `<path_to_repo_root>/data/FlattenFold/base`), or
    - Keep `repo_id` as-is and pass `--base-dir` explicitly to point to your local copy.

- **`--max-frames`** (`int`, optional)
  - If set, stops after processing at most `max_frames` frames across all parquet files.
  - Useful for **quick sanity checks** or debugging smaller subsets.

---

### What the script does

1. **Load config**
   - Uses `_config.get_config(config_name)` to get the `TrainConfig`.
   - Calls `config.data.create(config.assets_dirs, config.model)` to build a data config.
   - Reads `action_dim` from `config.model.action_dim`.

2. **Resolve input data directory**
   - If `base_dir` is not provided:
     - Uses `data_config.repo_id` as the base directory.
     - Prints a message like:
       - `Auto-detected base directory from config: <base_dir>`
   - Verifies that the directory exists.

3. **Scan parquet files**
   - Recursively walks `base_dir` and collects all files ending with `.parquet`.
   - Sorts them lexicographically for **deterministic ordering** (matches dataset order).

4. **Read and process data**
   - For each parquet file:
     - Loads it with `pandas.read_parquet`.
     - Expects columns:
       - `observation.state`
       - `action`
     - For each row:
       - Extracts `state` and `action` arrays.
       - Applies:
         - `process_state(state, action_dim)`
         - `process_actions(actions, action_dim)`
       - These helpers:
         - **Pad** to `action_dim` (if dimension is smaller).
         - **Clip abnormal values** outside \([-π, π]\) to 0 (for robustness, consistent with `FakeInputs` logic).
     - Accumulates processed arrays into:
       - `collected_data["state"]`
       - `collected_data["actions"]`
     - Maintains a running `total_frames` counter and respects `max_frames` if provided.

5. **Concatenate and pad**
   - Concatenates all collected batches per key:
     - `all_data["state"]`, `all_data["actions"]`
   - Ensures the last dimension matches `action_dim` (pads with zeros if needed).

6. **Compute statistics with `RunningStats`**
   - Initializes one `normalize.RunningStats()` per key (`state`, `actions`).
   - Feeds data in **batches of 32** to match the original implementation’s floating-point
     accumulation behavior.
   - For each key, computes:
     - `mean`, `std`, `q01`, `q99`, etc.

7. **Save `norm_stats`**
   - Collects results into a dict `norm_stats`.
   - Saves them with `openpi.shared.normalize.save` to:
     - `output_path = config.assets_dirs / data_config.repo_id`
   - Prints the output path and a success message:
     - `✅ Normalization stats saved to <output_path>`

> **Note:** The save logic mirrors the original openpi `compute_norm_stats.py` behavior so that
> training code can load `norm_stats` transparently.

---

### Typical workflow with Kai0 configs

1. **Download dataset**
   - Follow [`docs/dataset.md`](./dataset.md#step-1-download-the-dataset) to download the Kai0
     dataset under `./data` at the repo root.

2. **Set config paths**
   - Edit `src/openpi/training/config.py` for the normal π₀.5 configs (see README `Preparation`):
     - `repo_id` → absolute path to the dataset subset, e.g.
       `<path_to_repo_root>/data/FlattenFold/base`
     - `weight_loader` → path to the π₀.5 base checkpoint (e.g. Kai0 best model per task).

3. **Compute normalization stats**
   - From the repo root:

```bash
uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal
```

4. **Train**
   - Then run JAX training with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_flatten_fold_normal --exp_name=<your_experiment_name>
```

The training code will pick up the normalization statistics saved by this script and use them
for input normalization, in the same way as the original openpi pipeline.


# Data Augmentation

Utilities for augmenting and converting robot datasets: time scaling (frame extraction), space mirroring (left/right flip + merge), and converting HDF5 episode data into LeRobot-compatible datasets.

## Contents

| Script | Location | Description |
|--------|----------|-------------|
| **time_scaling.py** | `data_augment/time_scaling.py` | Extract every Nth frame from a LeRobot dataset (time-scale). Requires **lerobot** in the environment. |
| **space_mirroring.py** | `data_augment/space_mirroring.py` | Create mirrored dataset (swap left/right arms, flip videos) and/or merge datasets. |
| **convert_h5_lerobot.py** | `data_augment/utils/convert_h5_lerobot.py` | Convert HDF5 + videos → LeRobot format. Requires **mini_lerobot** (see below). |
| **merge_lerobot.py** | `data_augment/utils/merge_lerobot.py` | Merge LeRobot datasets (also used internally by space_mirroring and time_scaling). |
| **features.json** | `data_augment/utils/features.json` | Default feature schema for LeRobot output (used by convert_h5_lerobot). |

---

## Time scaling (`time_scaling.py`)

Extract every Nth frame from a LeRobot dataset (e.g. `extraction_factor=2` keeps frames 0, 2, 4, …), producing a shorter, “faster” episode. Optionally merge the extracted dataset with other sources, or use **split mode** to extract a portion of the data and merge with the rest.

**Requirement:** The **lerobot** package must be installed in the environment (e.g. `pip install lerobot` or your project’s env).

**Run from repository root:**

```bash
python train_deploy_alignment/data_augment/time_scaling.py --src_path <source_lerobot_path> --tgt_path <target_path> --repo_id <repo_id> [options]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--src_path` | Path to source LeRobot dataset. |
| `--tgt_path` | Path to target (extracted) dataset. |
| `--repo_id` | Repository ID for the new dataset. |

**Optional arguments:**

| Argument | Default | Description |
|----------|--------|-------------|
| `--extraction_factor` | 2 | Keep every Nth frame (e.g. 2 → frames 0, 2, 4, …). |
| `--force` | false | Overwrite target if it already exists. |
| `--merge_src_paths` | — | Space-separated paths of additional datasets to merge with the extracted one. |
| `--merge_tgt_path` | `<tgt_path>_merged` | Output path for the merged dataset. |
| `--merge_repo_id` | `<repo_id>_merged` | Repository ID for the merged dataset. |
| `--merge_force` | false | Force merge even if conflicts exist. |
| `--split_ratio` | — | Float in (0, 1). Split mode: extract this fraction of data (time-scaled), keep the rest, then merge. Final dataset has `repo_id` + `_time_scaling`. |

**Examples:**

```bash
# Basic: extract every 2nd frame
python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path /path/to/source --tgt_path /path/to/extracted --repo_id extracted_dataset \
  --extraction_factor 2

# With merge: extract then merge with another dataset
python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path /path/to/source --tgt_path /path/to/extracted --repo_id extracted_dataset \
  --merge_src_paths /path/to/other_dataset --merge_tgt_path /path/to/merged --merge_repo_id merged_dataset

# Split mode: extract 30% of data (every 2nd frame), keep 70% original, merge into one dataset
python train_deploy_alignment/data_augment/time_scaling.py \
  --src_path /path/to/source --tgt_path /path/to/final --repo_id my_dataset \
  --split_ratio 0.3 --extraction_factor 2
```

---

## Space mirroring (`space_mirroring.py`)

Create a left/right mirrored version of a LeRobot dataset (swap left/right arm state and action dimensions, horizontally flip images/videos), and optionally merge original + mirrored into one dataset.

**Commands:** `create-mirror` (only mirror), `merge` (only merge), `full` (mirror then merge in one go).

**Run from repository root:**

```bash
python train_deploy_alignment/data_augment/space_mirroring.py <command> [options]
```

### Command: `full` (mirror + merge)

One-shot: create mirrored dataset, then merge with original.

```bash
python train_deploy_alignment/data_augment/space_mirroring.py full \
  --src-path /path/to/original \
  --mirror-path /path/to/mirrored \
  --merge-path /path/to/merged \
  --repo-id my_dataset \
  [--fps 30] [--robot-type agilex] [--left-dim 7] [--right-dim 7] [--num-workers 4] [--features-json /path/to/features.json] [--force]
```

| Argument | Default | Description |
|----------|--------|-------------|
| `--src-path` | required | Source (original) LeRobot dataset path. |
| `--mirror-path` | required | Output path for the mirrored dataset. |
| `--merge-path` | required | Output path for the merged (original + mirrored) dataset. |
| `--repo-id` | required | Dataset repo_id. |
| `--fps` | 30 | Video FPS. |
| `--robot-type` | agilex | Robot type (e.g. agilex, arx). |
| `--left-dim` | 7 | Left arm state/action dimension. |
| `--right-dim` | 7 | Right arm state/action dimension. |
| `--num-workers` | 4 | Parallel workers. |
| `--features-json` | — | Path to features.json (optional). |
| `--force` | false | Force merge if output exists. |

### Command: `create-mirror` (mirror only)

```bash
python train_deploy_alignment/data_augment/space_mirroring.py create-mirror \
  --src-path /path/to/source --tgt-path /path/to/mirrored \
  [--left-dim 7] [--right-dim 7] [--num-workers 4]
```

### Command: `merge` (merge only)

```bash
python train_deploy_alignment/data_augment/space_mirroring.py merge \
  --src-paths /path/to/ds1 /path/to/ds2 --tgt-path /path/to/merged \
  --repo-id merged_dataset [--fps 30] [--robot-type agilex] [--features-json ...] [--force]
```

---

## Converting HDF5 to LeRobot (`convert_h5_lerobot.py`)

`convert_h5_lerobot.py` reads HDF5 episode files and existing per-camera videos, and writes a LeRobot-compatible dataset (parquet + video chunks + metadata) using **mini_lerobot**.

### 1. Mini LeRobot dependency

The script uses the local **mini_lerobot** package (lightweight LeRobot-compatible builder) and an **interface** module that lives next to it. Both must be available when running the script.

**Option A: Editable install + PYTHONPATH (recommended)**

From the repository root:

```bash
# Install mini_lerobot in editable mode (from data_augment/utils/mini_lerobot)
uv pip install -e train_deploy_alignment/data_augment/utils/mini_lerobot
# or: pip install -e train_deploy_alignment/data_augment/utils/mini_lerobot
```

Then run the converter with the **mini_lerobot** directory on `PYTHONPATH` so that `import interface` resolves to `utils/mini_lerobot/interface.py`:

```bash
cd train_deploy_alignment/data_augment/utils
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini_lerobot"
python convert_h5_lerobot.py --help
```

**Option B: Run from inside `mini_lerobot`**

```bash
cd train_deploy_alignment/data_augment/utils/mini_lerobot
uv pip install -e .   # or pip install -e .
# Run the script from utils with PYTHONPATH including mini_lerobot so "interface" is found
PYTHONPATH="$(pwd):$PYTHONPATH" python ../convert_h5_lerobot.py --help
```

**Dependencies installed with mini_lerobot** (from `utils/mini_lerobot/pyproject.toml`): `numpy`, `tqdm`, `pyarrow`, `av`, `tyro`, `h5py`, `mcap`, `opencv-python`, etc. No need to install them separately if you install mini_lerobot with `-e`.

### 2. Input layout expected by the script

- **`data_dir`**: Directory containing one subdir per `repo_id`, each with:
  - `*.hdf5` — episode files (e.g. `observations/qpos`, `observations/images/...`).
  - `video/cam_high/`, `video/cam_left_wrist/`, `video/cam_right_wrist/` — per-episode `.mp4` files named like the HDF5 stem (e.g. `episode_0.mp4`).
- The script validates that each episode’s video frame count matches the HDF5 length; episodes with missing or invalid videos are skipped.

### 3. Usage

From `train_deploy_alignment/data_augment/utils` (with `PYTHONPATH` set as above):

```bash
python convert_h5_lerobot.py <data_dir> <save_dir> <repo_ids> [options]
```

Examples:

```bash
# Single repo_id
python convert_h5_lerobot.py /path/to/raw_data /path/to/output my_repo_id

# Multiple repo_ids (comma-separated or repeated)
python convert_h5_lerobot.py /path/to/raw_data /path/to/output repo1 repo2

# With optional args
python convert_h5_lerobot.py /path/to/raw_data /path/to/output my_repo \
  --prompt "fold the cloth" \
  --save-repoid my_lerobot \
  --max-workers 8 \
  --features-json ./features.json
```

**Main arguments (tyro):**

| Argument | Description |
|----------|-------------|
| `data_dir` | Root directory containing `<repo_id>` subdirs with `.hdf5` and `video/` layout. |
| `save_dir` | Root directory for output; script writes `<save_dir>/<task>/<save_repoid>/`. |
| `repo_ids` | One or more subdir names under `data_dir` to convert. |
| `--prompt` | Task prompt string (default inferred from `data_dir.name`, e.g. "fold the cloth"). |
| `--save-repoid` | Output dataset repo id (default derived from `data_dir` name + `_lerobot`). |
| `--max-workers` | Parallel workers (default 8). |
| `--overwrite` | Overwrite existing output directory. |
| `--only-sync` | Only validate/sync, do not build dataset. |
| `--features-json` | Path to feature schema JSON (default: `utils/features.json`). |

Output is a LeRobot-style dataset under `<save_dir>/<task>/<save_repoid>/` (e.g. `data/chunk-*/`, `meta/`, `videos/`), suitable for use with this repo’s training pipelines.

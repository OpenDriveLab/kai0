---
license: cc-by-nc-sa-4.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: FlattenFold/base/data/chunk-000/episode_000000.parquet
---
# KAI0
<div align="center">
  <a href="https://github.com/OpenDriveLab/kai0">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub Badge">
  </a>
  <a href="https://huggingface.co/OpenDriveLab-org/Kai0">
    <img src="https://img.shields.io/badge/Model-grey?logo=Model" alt="Model Badge">
  </a>
  <a href="https://mmlab.hk/research/kai0">
    <img src="https://img.shields.io/badge/Research_Blog-grey?style=flat" alt="Research Blog Badge">
  </a>
</div>

# TODO
- [ ] The advantage label will be coming soon.

## Contents
- [About the Dataset](#about-the-dataset)
- [Step 1: Download the Dataset](#step-1-download-the-dataset)
- [Step 2: Load the Dataset](#step-2-load-the-dataset)
- [Dataset Structure](#dataset-structure)
    - [Folder hierarchy](#folder-hierarchy)
    - [Details](#details)
- [License and Citation](#license-and-citation)

## [About the Dataset](#contents)
- **~134 hours** real world scenarios 
- **Main Tasks**
    - ***FlattenFold***  
      - Single task
      - Initial state: T-shirts are randomly tossed onto the table, presenting random crumpled configurations
      - Manipulation task: Operate the robotic arm to unfold the garment, then fold it
    - ***HangCloth***
      - Single task
      - Initial state: Hanger is randomly placed, garment is randomly positioned on the table
      - Manipulation task: Operate the robotic arm to thread the hanger through the garment, then hang it on the rod
    - ***TeeShirtSort***
      - Garment classification and arrangement task
      - Initial state: Randomly pick a garment from the laundry basket
      - Classification: Determine whether the garment is a T-shirt or a dress shirt
      - Manipulation task:
        - If it is a T-shirt, fold the garment
        - If it is a dress shirt, expose the collar, then push it to one side of the table
- **Count of the dataset** 

    | Task | Base (episodes count/hours) | DAgger (episodes count/hours) | Total(episodes count/hours) |
    |------|-----------------------------|-------------------------------|-----------------------------|
    | FlattenFold | 3,055/~42 hours             | 3,457/ ~13 Hours              | 6,512 /~55 hours            | 
    | HangCloth | 6954/~61 hours              | 686/~12 hours                 | 7640/~73 hours              |
    | TeeShirtSort | 5988/~31 hours              | 769/~22 hours                 | 6757/~53 hours              |
    | **Total** | **15,997/~134 hours**       | **4,912/~47 hours**           | **20,909/~181 hours**       |

## Step 1: Download the Dataset

**Recommended (one command to `./data`):** From the **repository root** of [kai0](https://github.com/OpenDriveLab/kai0), run:

```bash
pip install huggingface_hub
python scripts/download_dataset.py
```

The dataset is saved under `./data` (FlattenFold, HangCloth, TeeShirtSort). Training and evaluation scripts expect this path by default.

**Optional:** Download only specific tasks or to a custom directory:

```bash
python scripts/download_dataset.py --tasks FlattenFold HangCloth --local-dir /path/to/output
```

**Manual download (Hugging Face):**

```bash
# Full dataset to a directory of your choice
hf download OpenDriveLab-org/Kai0 --repo-type dataset --local-dir /path/to/output
```

Or in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="OpenDriveLab-org/Kai0",
    repo_type="dataset",
    local_dir="/path/to/output",
)
```

---

## Step 2: Load the Dataset

This dataset is in [LeRobot](https://github.com/huggingface/lerobot) format (v2.1).

### LeRobot &lt; 0.4.0

| Version                | Import |
|------------------------|--------|
| ≤ 0.1.0                | `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset` |
| &gt; 0.1.0 and &lt; 0.4.0 | `from lerobot.datasets.lerobot_dataset import LeRobotDataset` |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # adjust by version

# After running scripts/download_dataset.py (default ./data)
dataset = LeRobotDataset("path/to/kai0/repo/data/FlattenFold/base")  # or local path to a task subset
```

### LeRobot ≥ 0.4.0

Migrate v2.1 → v3.0 first: [LeRobot dataset v3 migration](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=OpenDriveLab-org/Kai0
```

## Dataset Structure

### Folder hierarchy

Under each task directory, data is partitioned into two subsets: **base** and **dagger**.
- **base** — Original demonstration trajectories for garment manipulation.
- **dagger** — On-policy recovery trajectories from iterative DAgger (failure-recovery modes).
```text
Kai0-data/
├── FlattenFold/
│   ├── base/
│   │   ├── data/
│   │   │   ├── chunk-000/
│   │   │   │   ├── episode_000000.parquet
│   │   │   │   ├── episode_000001.parquet
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── videos/
│   │   │   ├── chunk-000/
│   │   │   │   ├── observation.images.hand_left/
│   │   │   │   │   ├── episode_000000.mp4
│   │   │   │   │   ├── episode_000001.mp4
│   │   │   │   │   └── ...
│   │   │   │   ├── observation.images.hand_right/
│   │   │   │   │   ├── episode_000000.mp4
│   │   │   │   │   ├── episode_000001.mp4
│   │   │   │   │   └── ...
│   │   │   │   ├── observation.images.top_head/
│   │   │   │   │   ├── episode_000000.mp4
│   │   │   │   │   ├── episode_000001.mp4
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── meta/
│   │       ├── info.json
│   │       ├── episodes.jsonl
│   │       ├── tasks.jsonl
│   │       └── episodes_stats.jsonl
│   └── dagger/
├── HangCloth/
│   ├── base/
│   └── dagger/
├── TeeShirtSort/
│   ├── base/
│   └── dagger/
└── README.md
```

### Details
#### info.json
the basic struct of the [info.json](#meta/info.json)
```json
{
    "codebase_version": "v2.1",
    "robot_type": "agilex",
    "total_episodes": ...,  # the total episodes in the dataset
    "total_frames": ...,    # The total number of video frames in any single camera perspective
    "total_tasks": ...,     # Total number of tasks
    "total_videos": ...,    # The total number of videos from all camera perspectives in the dataset
    "total_chunks": ...,    # The number of chunks in the dataset
    "chunks_size": ...,     # The max number of episodes in a chunk
    "fps": ...,             # Video frame rate per second
    "splits": {             # how to split the dataset
        "train": ...       
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.top_head": {   # the camera perspective
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.hand_left": {   # the camera perspective
            ...
        },
        "observation.images.hand_right": {   # the camera perspective
            ...
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": null
        },
        "action": {
            "dtype": "float32",
            "shape": [
                14
            ],
            "names": null
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```

#### [Parquet file format](#contents)
| Field Name | shape | Meaning |
|------------|-------------|-------------|
| observation.state | [N, 14] |left `[:, :6]`, right `[:, 7:13]`, joint angle<br> left`[:, 6]`, right `[:, 13]` , gripper open range|
| action | [N, 14]  |left `[:, :6]`, right `[:, 7:13]`, joint angle<br>left`[:, 6]`, right `[:, 13]` , gripper open range |
| timestamp | [N, 1] | Time elapsed since the start of the episode (in seconds) |
| frame_index | [N, 1] | Index of this frame within the current episode (0-indexed) |
| episode_index | [N, 1] | Index of the episode this frame belongs to |
| index | [N, 1] | Global unique index across all frames in the dataset |
| task_index | [N, 1] | Index identifying the task type being performed |

### tasks.jsonl

Task language prompts (natural language instructions). Each entry maps a `task_index` to its description for language-conditioned policy training.

# License and Citation
All the data and code within this repo are under [](). Please consider citing our project if it helps your research.

```BibTeX
@misc{,
  title={},
  author={},
  howpublished={\url{}},
  year={}
}
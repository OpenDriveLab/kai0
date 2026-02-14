# χ₀

<div id="top" align="center">

[![Blog Page](https://img.shields.io/badge/Blog_Page-green)](https://mmlab.hk/research/kai0)
[![arXiv](https://img.shields.io/badge/arXiv-2602.09021-b31b1b)](https://arxiv.org/abs/2602.09021)
[![Kai0 Data](https://img.shields.io/badge/huggingface-Kai0_Data-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/OpenDriveLab-org/Kai0)
[![Kai0 Model](https://img.shields.io/badge/huggingface-Kai0_Model-orange?logo=huggingface&logoColor=white)](https://huggingface.co/OpenDriveLab-org/Kai0)
[![ModelScope Data](https://img.shields.io/badge/ModelScope-Kai0_Data-purple)](https://www.modelscope.cn/datasets/OpenDriveLab/Kai0)
[![ModelScope Model](https://img.shields.io/badge/ModelScope-Kai0_Model-purple)](https://www.modelscope.cn/models/OpenDriveLab/Kai0)

<!-- [![Repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/OpenDriveLab/kai0) -->


</div>

χ₀ (**kai0**) is a resource-efficient framework for achieving production-level robustness in robotic manipulation by taming distributional inconsistencies.
<!-- This repository is built on top of [openpi](https://github.com/Physical-Intelligence/openpi), the open-source models and packages for robotics published by the [Physical Intelligence team](https://www.physicalintelligence.company/). -->

χ₀ addresses the systematic distributional shift among the human demonstration distribution ($P_\text{train}$), the inductive bias learned by the policy ($Q_\text{model}$), and the test-time execution distribution ($P_\text{test}$) through three technical modules:

- **[Model Arithmetic](#model-arithmetic)**: A weight-space merging strategy that combines models trained on different data subsets, efficiently capturing diverse knowledge without architectural complexity. **[Released]**
- **[Stage Advantage](#stage-advantage-coming-soon)**: A stage-aware advantage estimator that provides stable, dense progress signals for policy training. **[Coming Soon]**
- **[Train-Deploy Alignment](#train-deploy-alignment-coming-soon)**: Bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. **[Coming Soon]**

χ₀ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation — flattening, folding, and hanging — surpassing the state-of-the-art $\pi_{0.5}$ baseline by approximately 250% in success rate, with `only 20 hours of data and 8 A100 GPUs`.

<!-- [[Paper]](https://github.com/OpenDriveLab/kai0) [[Blog]](https://mmlab.hk/research/kai0) -->

https://github.com/user-attachments/assets/3f5f0c48-ff3f-4b9b-985b-59ad0b2ea97c

## Table of Contents

- [Update](#update)
- [Acknowledgement](#acknowledgement)
- [Requirements](#requirements)
  - [Compute](#compute)
  - [Hardware](#hardware)
- [Installation](#installation)
- [Preparation](#preparation)
  - [1. Download the dataset](#1-download-the-dataset)
  - [2. Download checkpoints (optional, for testing)](#2-download-checkpoints-optional-for-testing)
  - [3. Fine-tune with normal π₀.₅](#3-fine-tune-with-normal-π₀.₅)
- [Project Overview](#project-overview)
- [Modules Overview and To-Do List](#modules-overview-and-to-do-list)
- [Model Arithmetic](#model-arithmetic)
  - [Workflow](#workflow)
  - [Quick Start](#quick-start)
- [Stage Advantage (Coming Soon)](#stage-advantage-coming-soon)
- [Train-Deploy Alignment (Coming Soon)](#train-deploy-alignment-coming-soon)
- [Citation](#licenseandcitation)
- [Troubleshooting](#troubleshooting)
- [Links and Community](#links-and-community)

## Update

- [Feb 10 2026] Initial release of the **Model Arithmetic** module with support for both JAX and PyTorch checkpoints (not tested thoroughly).
- [Feb 10 2026] χ₀ paper released.

## Acknowledgement

This repository is built on top of [openpi](https://github.com/Physical-Intelligence/openpi) by [Physical Intelligence](https://www.physicalintelligence.company/). We sincerely thank the Physical Intelligence team for open-sourcing their excellent π₀ and π₀.₅ models and the openpi codebase, which made this work possible. The base model training, inference pipeline, and data processing utilities all originate from openpi. Please refer to the [openpi README](https://github.com/Physical-Intelligence/openpi) for details on the base models, fine-tuning, and inference.

## Requirements

### Compute

χ₀ shares the same system requirements as openpi. You will need an NVIDIA GPU with at least the following specifications:

| Mode               | Memory Required | Example GPU           |
| ------------------ | --------------- | --------------------- |
| Inference          | > 8 GB          | RTX 4090              |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090 (not tested) |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100    |

For Model Arithmetic (mixing checkpoints), GPU memory requirements depend on the model size and number of checkpoints being mixed. A single A100 (80GB) is sufficient for most use cases.

Non-edge components (e.g., Policy Training, Model Arithmetic) have been tested on Ubuntu 22.04.

### Hardware

For real-robot deployment (dual-arm setup, cameras, and table layout), see **[Hardware Setup & 3D Print Files](setup/README.md)**. That document covers supported platforms (Agilex Piper for FlattenFold / TeeShirtSort, ARX X5 for HangCloth), Intel RealSense D435i camera placement, 3D-printed grippers and mounts with usage notes, and inference host GPU (RTX 4090 in Ubuntu 20.04).

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:OpenDriveLab/kai0.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

Follow the [openpi installation instructions](https://github.com/Physical-Intelligence/openpi#installation) to set up the base environment with [uv](https://docs.astral.sh/uv/):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

For PyTorch checkpoint mixing (not tested thoroughly), ensure `safetensors` is installed:

```bash
uv pip install safetensors
```

## Preparation

### 1. Download the dataset

Download the Kai0 dataset so it is available under `./data` for training and evaluation. From the repository root, run:

```bash
python scripts/download_dataset.py
```

This fetches the full dataset from [Hugging Face](https://huggingface.co/datasets/OpenDriveLab-org/Kai0) into `./data` (FlattenFold, HangCloth, TeeShirtSort). To download only specific tasks or use a custom path, see the [dataset docs](docs/dataset.md#step-1-download-the-dataset).

### 2. Download checkpoints (optional, for testing)

We provide **one best model per task** (FlattenFold, HangCloth, TeeShirtSort) in the [Kai0 repo on Hugging Face](https://huggingface.co/OpenDriveLab-org/Kai0/tree/main).

From the repository root, you can download all best-model checkpoints to `./checkpoints` with:

```bash
python scripts/download_checkpoints.py
```

To download only specific tasks or use a custom path, run:

```bash
python scripts/download_checkpoints.py --tasks FlattenFold HangCloth --local-dir ./my_checkpoints
```

After download, set `weight_loader` in the training config to the path of the corresponding checkpoint directory (see step 3 below). You can also use openpi’s pretrained π₀.5 checkpoint instead.

### 3. Fine-tune with normal π₀.₅

After the dataset is in `./data`, you can run **normal π₀.₅ full fine-tuning** on it, then use the resulting checkpoints for [Model Arithmetic](#model-arithmetic).

**Set paths in config**

Edit [`src/openpi/training/config.py`](src/openpi/training/config.py) (around lines 1173–1226) for the task(s) you need:

- **`repo_id`**: set to the **absolute path** to the dataset subset, e.g. `<path_to_repo_root>/data/FlattenFold/base`, `<path_to_repo_root>/data/TeeShirtSort/base`, or `<path_to_repo_root>/data/HangCloth/base`.
- **`weight_loader`**: set to the path of your **π₀.₅ base checkpoint** — either the best model you downloaded in step 2 above, or openpi’s pretrained π₀.₅ checkpoint.

Config names to use: e.g. `pi05_flatten_fold_normal`

**Compute normalization stats with our optimized scripts**

```bash
uv run python scripts/compute_norm_states_fast.py --config-name <config_name>
```

Example: `uv run python scripts/compute_norm_states_fast.py --config-name pi05_flatten_fold_normal`

**Start training**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=<your_experiment_name>
```

Example: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_flatten_fold_normal --exp_name=flatten_fold_run1`

Checkpoints are written to the config’s checkpoint directory. You can then use your checkpoints as inputs to **model arithmetic** (see [Model Arithmetic](#model-arithmetic)).

## Project Overview

```
+-----------------------------------------------------------------------------------------------+
|                                    kai0 Framework Overview                                    |
|   Built on openpi: full-param finetuning of pi0/pi0.5 + server/client inference               |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|   Main Pipeline:                                                                              |
|                                                                                               |
|   +----------------+     +----------------+     +----------------+     +----------------+     |
|   |Data Processing |     |Model Finetuning|     |Model Arithmetic|     |Infer. & Deploy |     |
|   | augment,mirror |---->| pi0/pi0.5 full |---->| ckpt merging,  |---->| server/client, |     |
|   | scale, merge   |     | param training |     | weight optimize|     | DAgger, smooth |     |
|   | train_deploy_  |     | openpi train   |     | model_         |     | train_deploy_  |     |
|   | alignment/data |     | scripts        |     | arithmetic/    |     | alignment/     |     |
|   +----------------+     +--------^-------+     +----------------+     +----------------+     |
|                                   |                                                           |
|                                   | advantage labels enable                                   |
|                                   | advantage-weighted regression                             |
|                                   |                                                           |
|   Stage Advantage Pipeline:       |                                                           |
|                                   |                                                           |
|   +----------------+     +--------+-------+     +----------------+                            |
|   | GT Data        |     | Train          |     | Adv. Labelling |                            |
|   | Labelling      |---->| Adv. Estimator |---->| (prediction)   |                            |
|   | stage_adv./    |     | stage_adv./    |     | stage_adv./    |                            |
|   +----------------+     +----------------+     +----------------+                            |
|                                                                                               |
+-----------------------------------------------------------------------------------------------+
```

## Modules Overview and to-do list

<!-- | Module                  | Description                                                        | Status       |
| ----------------------- | ------------------------------------------------------------------ | ------------ |
| Model Arithmetic        | Weight-space merging of multiple trained checkpoints                | Released     |
| Stage Advantage         | Stage-aware advantage estimation for policy training                | Coming Soon Before CNY  |
| Train-Deploy Alignment  | DAgger, spatio-temporal augmentation, and chunk-wise smoothing      | Coming Soon Before CNY | -->

- [x] kai0 oracle: training and inference code with non-advantage data of three tasks
- [x] Model Arithmetic: code of different baselines for weight-space interpolation
- [ ] Stage Advantage: code, data (advantage labels), and checkpoints — **Feb 15**
- [ ] HuggingFace & ModelScope: upload Stage Advantage data and checkpoints — **Feb 14**
- [ ] Train-Deploy Alignment — **Feb 14**

## Model Arithmetic

Model Arithmetic combines multiple trained openpi model checkpoints into a single mixed model using optimized weighted averaging. This enables efficiently aggregating knowledge from models trained on different data subsets (e.g., different object appearances, state variations) without requiring Mixture-of-Experts architectures.

Both JAX (Orbax/OCDBT) and PyTorch checkpoints (`model.safetensors`, not tested thoroughly) are supported. Six mixing methods are available: **average** (equal weight \(1/N\) per checkpoint), **inverse_loss**, **gradient_descent**, **adaptive_gradient_descent**, **greedy**, and **manual weights**.

### Workflow

The mixing process follows three steps:

1. **(Optional)** Split a LeRobot dataset into subsets and train one model per subset.
2. Dump a small validation set for weight optimization.
3. Mix the checkpoints using one of the supported methods.

### Quick Start

Taking Task C (hanging clothes) as an example:

**Step 1: Dump validation data**

```bash
python model_arithmetic/dump_data.py \
  --dataset pi05_hang_cloth \
  --output hang_cloth_val.pkl
```

**Step 2: Mix checkpoints** (example using inverse_loss — fastest method, no gradient steps)

```bash
# JAX checkpoints
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt \
  --optimize_method inverse_loss \
  --use_gpu \
  --gpu_ids "0"

# PyTorch checkpoints (not tested thoroughly)
python model_arithmetic/arithmetic_torch.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints /path/to/torch_ckpt1 /path/to/torch_ckpt2 /path/to/torch_ckpt3 \
  --output /path/to/mixed_torch_ckpt \
  --optimize_method inverse_loss
```

For gradient-based optimization, dataset splitting, and all other methods, see the full documentation in [`model_arithmetic/README.md`](model_arithmetic/README.md).

## Stage Advantage (Coming Soon)

Stage Advantage decomposes long-horizon tasks into semantic stages and provides stage-aware advantage signals for policy training. It addresses the numerical instability of prior non-stage approaches by computing advantage as progress differentials within each stage, yielding smoother and more stable supervision.

**This module is currently under refinement and will be released soon.**

## Train-Deploy Alignment (Coming Soon)

Train-Deploy Alignment bridges the distribution gap between training and real-world deployment through:
- **Spatio-temporal augmentation**: Data augmentation including space mirroring and time scaling for dual-arm setups.
- **Heuristic DAgger corrections**: Interactive on-robot data collection for iterative policy improvement.
- **Temporal chunk-wise smoothing**: Smoothed action execution to reduce jitter during deployment.

**This module is currently under refinement and will be released soon.**

## License and Citation

All assets and code in this repository are under the Apache 2.0 license unless specified otherwise. The data and checkpoint are under CC BY-NC-SA 4.0. Other modules (including PaliGemma) inherit their own distribution licenses. If you find χ₀ useful in your research, please consider citing:

```bibtex
@article{sima2026kai0,
  title={$\chi_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```

## Troubleshooting

*(Common issues and fixes will be added as we go.)*

## Links and Community

- [Paper](https://github.com/OpenDriveLab/kai0)
- [Project Blog](https://mmlab.hk/research/kai0)
- [openpi (Base Repository)](https://github.com/Physical-Intelligence/openpi)
- [UniVLA](https://github.com/OpenDriveLab/UniVLA)
- [SparseVideoNav](https://github.com/OpenDriveLab/SparseVideoNav)
<!-- - [X (Twitter)](https://x.com/OpenDriveLab/status/2003745616955142150)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7409531902761795584/) -->

Join our community for discussions, questions, and updates:

<table>
  <tr>
    <td align="center"><b>Discord</b></td>
    <td align="center"><b>Feishu</b></td>
    <td align="center"><b>WeChat Group</b></td>
    <td align="center"><b>WeChat Chonghao</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/community_qrcode/discord_qrcode.jpg" width="200"></td>
    <td align="center"><img src="assets/community_qrcode/feishu_qrcode.png" width="200"></td>
    <td align="center"><img src="assets/community_qrcode/wechat_qrcode.jpeg" width="200"></td>
    <td align="center"><img src="assets/community_qrcode/wechat_smch_qrcode.jpeg" width="200"></td>
  </tr>
</table>

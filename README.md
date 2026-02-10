# χ₀


<div>
<a href="https://mmlab.hk/research/kai0" target="_blank"><img src="https://img.shields.io/badge/Blog_Page-green" alt="Blog Page"></a>
<a href="https://github.com/OpenDriveLab/kai0"><img alt="Repo" src="https://img.shields.io/badge/github-repo-blue?logo=github"/></a>
<a href="https://arxiv.org/abs/2602.09021" target="_blank"><img src="https://img.shields.io/badge/arXiv-2602.09021-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/OpenDriveLab-org/Kai0">
  <img alt="Kai0 Data" src="https://img.shields.io/badge/huggingface-Kai0_Data-orange?logo=huggingface&logoColor=white"/>
</a>
<a href="https://www.modelscope.cn/models/OpenDriveLab/Kai0">
  <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Kai0_Data-purple"/>
</a>


</div>

χ₀ (**kai0**) is a resource-efficient framework for achieving production-level robustness in robotic manipulation by taming distributional inconsistencies. This repository is built on top of [openpi](https://github.com/Physical-Intelligence/openpi), the open-source models and packages for robotics published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

χ₀ addresses the systematic distributional shift among the human demonstration distribution ($P_\text{train}$), the inductive bias learned by the policy ($Q_\text{model}$), and the test-time execution distribution ($P_\text{test}$) through three technical modules:

- **[Model Arithmetic](#model-arithmetic)**: A weight-space merging strategy that combines models trained on different data subsets, efficiently capturing diverse knowledge without architectural complexity. **[Released]**
- **[Stage Advantage](#stage-advantage-coming-soon)**: A stage-aware advantage estimator that provides stable, dense progress signals for policy training. **[Coming Soon]**
- **[Train-Deploy Alignment](#train-deploy-alignment-coming-soon)**: Bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. **[Coming Soon]**

χ₀ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation — flattening, folding, and hanging — surpassing the state-of-the-art $\pi_{0.5}$ baseline by approximately 250% in success rate, with only 20 hours of data and 8 A100 GPUs.

[[Paper]](https://github.com/OpenDriveLab/kai0) [[Blog]](https://mmlab.hk/research/kai0)

## Updates

- [Feb 10 2026] Initial release of the **Model Arithmetic** module with support for both JAX and PyTorch checkpoints (not tested thoroughly).
- [Feb 10 2025] χ₀ paper released.

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

The repo has been tested with Ubuntu 22.04.

### Hardware

For real-robot deployment (dual-arm setup, cameras, and table layout), see **[Hardware Setup & 3D Print Files](setup/README.md)**. That document covers supported platforms (Agilex Piper for FlattenFold / TeeShirtSort, ARX X5 for HangCloth), Intel RealSense D435 camera placement, 3D-printed grippers and mounts with usage notes, and inference host GPU (RTX 4090 in Ubuntu 20.04).

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

## Modules Overview

| Module                  | Description                                                        | Status       |
| ----------------------- | ------------------------------------------------------------------ | ------------ |
| Model Arithmetic        | Weight-space merging of multiple trained checkpoints                | Released     |
| Stage Advantage         | Stage-aware advantage estimation for policy training                | Coming Soon  |
| Train-Deploy Alignment  | DAgger, spatio-temporal augmentation, and chunk-wise smoothing      | Coming Soon  |

## Model Arithmetic

Model Arithmetic combines multiple trained openpi model checkpoints into a single mixed model using optimized weighted averaging. This enables efficiently aggregating knowledge from models trained on different data subsets (e.g., different object appearances, state variations) without requiring Mixture-of-Experts architectures.

Both JAX (Orbax/OCDBT) and PyTorch (`model.safetensors`) checkpoints (not tested thoroughly) are supported. Five mixing methods are available: **inverse_loss**, **gradient_descent**, **adaptive_gradient_descent**, **greedy**, and **manual weights**.

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

## Citation

If you find χ₀ useful in your research, please consider citing:

```bibtex
@article{sima2026kai0,
  title={χ₀: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies},
  author={Yu, Checheng and Sima, Chonghao and Jiang, Gangcheng and Zhang, Hai and Mai, Haoguang and Li, Hongyang and Wang, Huijie and Chen, Jin and Wu, Kaiyang and Chen, Li and Zhao, Lirui and Shi, Modi and Luo, Ping and Bu, Qingwen and Peng, Shijia and Li, Tianyu and Yuan, Yibo},
  journal={arXiv preprint arXiv:2602.09021},
  year={2026}
}
```

## Links

- [Paper](https://github.com/OpenDriveLab/kai0)
- [Project Blog](https://mmlab.hk/research/kai0)
- [openpi (Base Repository)](https://github.com/Physical-Intelligence/openpi)
- [UniVLA](https://github.com/OpenDriveLab/UniVLA)
- [SparseVideoNav](https://github.com/OpenDriveLab/SparseVideoNav)
<!-- - [X (Twitter)](https://x.com/OpenDriveLab/status/2003745616955142150)
- [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7409531902761795584/) -->

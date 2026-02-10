## Model Arithmetic

Workflow:

0. **Optional:** Split a LeRobot dataset into subsets with `split_data.py` (train one model per subset).
1. Dump a small validation set with `dump_data.py`.
2. Mix the checkpoints into one:
   - **JAX checkpoints** (Orbax/OCDBT): use `arithmetic.py`
   - **PyTorch checkpoints** (model.safetensors): use `arithmetic_torch.py`

You need a working OpenPI environment (same as training): JAX/Flax for JAX, PyTorch + `safetensors` for PyTorch, and the `openpi` package.  
Scripts: `split_data.py`, `dump_data.py`, `arithmetic.py` (JAX), `arithmetic_torch.py` (PyTorch). Shared helpers live in `arithmetic_common.py`.

Both `arithmetic.py` and `arithmetic_torch.py` support the same methods: **average**, **inverse_loss**, **gradient_descent**, **adaptive_gradient_descent**, **greedy**, and manual **--weights**.

---

## Step 0: Split dataset (optional)

If you want to train **separate models on different data subsets** and then mix them, first split a LeRobot-format dataset into disjoint subsets by episode. Each subset is a full LeRobot dataset (e.g. for training with your existing pipeline).

```bash
python model_arithmetic/split_data.py \
  --source_path /path/to/lerobot_dataset \
  --dst_path /path/to/split_output \
  --split_num 4 \
  --seed 42
```

- **`--source_path`**: Path to the source LeRobot dataset (must contain `meta/`, `data/`, `videos/`).
- **`--dst_path`**: Output directory; subsets are written as `dst_path/split_0`, `dst_path/split_1`, ...
- **`--split_num`**: Number of subsets (default: 4).
- **`--seed`**: Random seed for shuffling episodes before splitting (default: 42).

Then train one model on each of `split_0`, `split_1`, ... and use Step 1–2 below to dump validation data and mix those checkpoints.

---

## Step 1: Dump validation data

Pick a config name (same as training, e.g. `pi05_hang_cloth`) and run:

```bash
python model_arithmetic/dump_data.py \
  --dataset pi05_hang_cloth \
  --output hang_cloth_val.pkl
```

Change `--dataset` and `--output` to your own config and file name if needed.

---

## Step 2: Mix checkpoints

- **JAX**: run `arithmetic.py`. Checkpoints are Orbax dirs (e.g. `.../90000` or `.../90000/params`). Output: `OUTPUT_DIR/0/` + `norm_stats.json`.
- **PyTorch**: run `arithmetic_torch.py`. Checkpoints are dirs containing `model.safetensors`. Output: `OUTPUT_DIR/model.safetensors` + `norm_stats.json`.

Common arguments for both scripts:

- `--config`: same config name as training (e.g. `pi05_hang_cloth`)
- `--data-path`: the `.pkl` from Step 1
- `--checkpoints`: one or more checkpoint dirs
- `--output`: directory where the mixed checkpoint will be saved

Choose one of the following methods. Examples below use `arithmetic.py` (JAX); for PyTorch use `arithmetic_torch.py` with the same `--optimize_method` and `--weights` (no `--weight_format`).

---

### Method 1: `average`

Equal weighting: with \(N\) checkpoints, each gets weight \(1/N\). No validation data or optimization; fastest option.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_avg \
  --optimize_method average \
  --use_gpu \
  --gpu_ids "0"
```

---

### Method 2: `inverse loss`

Compute each checkpoint’s loss on the validation set, then set weight proportional to \(1/\text{loss}^2\). Lower loss → higher weight. No gradient step, fast.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_inverse \
  --optimize_method inverse_loss \
  --use_gpu \
  --gpu_ids "0"
```

---

### Method 3: `gradient descent`

Optimize mixing weights by gradient descent on the mixed model’s validation loss (Adam + cosine LR). Usually gives better weights than inverse_loss. Tune `--num_iterations` and `--learning_rate` if needed.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_gd \
  --optimize_method gradient_descent \
  --num_iterations 50 \
  --learning_rate 0.05 \
  --use_gpu \
  --gpu_ids "0"
```

---

### Method 4: `adaptive gradient descent`

Same as gradient_descent but scales the gradient step by the current loss (larger loss → larger update). Can help when losses vary a lot. Same args as gradient_descent.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_adaptive \
  --optimize_method adaptive_gradient_descent \
  --num_iterations 50 \
  --learning_rate 0.05 \
  --use_gpu \
  --gpu_ids "0"
```

---

### Method 5: `greedy search`

Greedy forward selection: (1) pick the single checkpoint with lowest loss, (2) repeatedly add the checkpoint that most improves the (equal-weight) mix, (3) stop when no improvement. No continuous weights—only which checkpoints to include and equal weighting among them. No `--num_iterations` or `--learning_rate`.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_greedy \
  --optimize_method greedy \
  --use_gpu \
  --gpu_ids "0"
```

---

### Method 6: Manual weights

If you already know the weights (e.g. 0.5, 0.3, 0.2), pass them with `--weights`. They will be normalized to sum to 1. Do not set `--optimize_method`.

```bash
python model_arithmetic/arithmetic.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints \
    /path/to/ckpt_run1/90000 \
    /path/to/ckpt_run2/90000 \
    /path/to/ckpt_run3/90000 \
  --output /path/to/mixed_ckpt_manual \
  --weights 0.5 0.3 0.2 \
  --use_gpu \
  --gpu_ids "0"
```

Number of values in `--weights` must match the number of checkpoints.

---

### PyTorch checkpoints: `arithmetic_torch.py`

For OpenPI PyTorch checkpoints (each dir must contain `model.safetensors`), use `arithmetic_torch.py`. Same methods as JAX (average, inverse_loss, gradient_descent, adaptive_gradient_descent, greedy, manual `--weights`).

Example with gradient_descent:

```bash
python model_arithmetic/arithmetic_torch.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints /path/to/torch_ckpt1 /path/to/torch_ckpt2 /path/to/torch_ckpt3 \
  --output /path/to/mixed_torch_ckpt \
  --optimize_method gradient_descent \
  --num_iterations 50 \
  --learning_rate 0.05
```

Example with inverse_loss:

```bash
python model_arithmetic/arithmetic_torch.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints /path/to/torch_ckpt1 /path/to/torch_ckpt2 \
  --output /path/to/mixed_torch_ckpt \
  --optimize_method inverse_loss
```

Example with manual weights:

```bash
python model_arithmetic/arithmetic_torch.py \
  --config pi05_hang_cloth \
  --data-path hang_cloth_val.pkl \
  --checkpoints /path/to/torch_ckpt1 /path/to/torch_ckpt2 \
  --output /path/to/mixed_torch_ckpt \
  --weights 0.6 0.4
```

Requires `pip install safetensors` if not already installed.


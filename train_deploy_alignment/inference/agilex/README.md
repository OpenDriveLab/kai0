# Agilex Inference

Deploy trained OpenPi policies on Agilex dual-arm robots (Piper arms, RealSense cameras) with ROS Noetic. Inference runs by sending observations to a **policy server** (running on a GPU host) and applying the returned action chunks on the robot, with optional temporal smoothing or temporal ensembling [\[2\]](#references).

---

## Prerequisites: Piper SDK and robot setup

Before using this inference stack, follow the **official Agilex Piper SDK** setup so CAN, arms, and cameras work correctly:

- **[Piper SDK (Agilex Piper ROS SDK)](https://github.com/agilexrobotics/piper_sdk)**  
  Install dependencies, CAN tools, and the Piper SDK (e.g. `pip3 install piper_sdk` or clone and install). Use the official docs for:
  - CAN module activation (single or multiple)
  - Reading joint feedback and controlling the arm
  - Master/slave configuration for dual arms

After the Piper SDK and robot hardware are set up, install the **IPC Python environment** (below) and the rest of the inference stack (ROS Noetic, Piper ROS package, RealSense, inference scripts).

---

## IPC Python environment (one-time setup)

On the **IPC (industrial PC)**, use a **dedicated Python 3.10 environment** for inference. Do not reuse the GPU host’s training env.

### 1. Create and activate conda env

```bash
conda create -n kai0_inference python=3.10
conda activate kai0_inference
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies from requirements

From the **repository root**:

```bash
pip install -r train_deploy_alignment/inference/agilex/requirements_inference_ipc.txt
```

The file [requirements_inference_ipc.txt](requirements_inference_ipc.txt) lists versions for `pyquaternion`, `pyyaml`, `mujoco`, `dm_control`, `opencv-python`, `rospkg`, `diffusers`, etc.

### 4. Install OpenPi client (editable)

From the **repository root**:

```bash
cd packages/openpi-client
pip install -e .
cd ../..
```

If `packages/openpi-client` lives elsewhere (e.g. a separate clone), use that path. The inference scripts depend on `openpi_client` to talk to the policy server.

After this, install ROS Noetic, Piper ROS package, `piper_msgs`, and RealSense drivers as needed for your setup.

---

## Inference Setup

Inference uses two machines: a **GPU host** (e.g. 4090) that runs the policy server, and the **IPC** that runs ROS + the inference client.

### Step 1: On the GPU host — start the policy server

From the **repository root** on the GPU machine:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<train_config> --policy.dir=<checkpoint_dir> [--port=8000]
```

Use the **same training config name and checkpoint directory** as your trained model. Example:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

Default port is `8000`. The server listens on all interfaces (`0.0.0.0`) so the IPC can connect. Note the GPU host’s **IP address** (e.g. `192.168.1.10`) for the next step.

#### RTC (real-time chunking) mode

**RTC** stands for **real-time chunking** [\[1\]](#references) and enables aligning newly inferred action chunks with the previously executed prefix under latency. For **RTC inference** (`inference/agilex_inference_openpi_rtc.py` with `--rtc_mode`), the policy server must load the **RTC model** (Pi0RTC), which supports `prev_action_chunk`, `inference_delay`, and `execute_horizon` for real-time guidance. Use an **RTC training config** when starting the server:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_rtc_flatten_fold_inference --policy.dir=<path_to_jax_checkpoint> [--port=8000]
```

- **Config name:** Use an RTC config: **Pi0-based** e.g. `pi05_rtc_flatten_fold_inference` (`model=pi0_config.Pi0RTCConfig()`), or **Pi05-based** e.g. `pi05_rtc_flatten_fold_inference` (`model=pi0_config.Pi0RTCConfig(pi05=True)`). Same checkpoint weights as the non-RTC model can be used; the RTC config only changes the model class (Pi0RTC) used at serve time.
- **Checkpoint:** Must be a **JAX checkpoint** (directory containing `params/`, not `model.safetensors`). RTC is not supported for PyTorch checkpoints.
- Then on the IPC, run the RTC inference script with `--rtc_mode` (see [Inference scripts](#inference-scripts)).

### Step 2: On the IPC — set `--host` and run the robot stack

1. **Set the inference script’s policy server address**  
   When starting the inference script (see below), pass **`--host <gpu_host_ip>`** (and `--port` if you changed it). Example: `--host 192.168.1.10 --port 8000`. This makes the client on the IPC connect to the policy server on the GPU host.

2. **Start the robot stack and inference client**  
   Run roscore, CAN config, RealSense, Piper launch, then the inference script (with `--host` and `--port` as above). Full commands are in [Full setup (one-time and run-time)](#full-setup-one-time-and-run-time) below.

---

## Prompt and AWBC (important)

**You must set the language prompt in the inference code** so it matches the prompt used at training time. The policy is conditioned on this text; mismatched or missing prompts will hurt performance.

### Where to set the prompt

- **`inference/agilex_inference_openpi_temporal_smoothing.py`**  
  At the top of the file, set the global:
  ```python
  lang_embeddings = "fold the cloth"   # or your task string
  ```
  This value is sent as `"prompt"` in the inference payload.

- **`inference/agilex_inference_openpi_temporal_ensembling.py`**  
  Same: set the global `lang_embeddings` to the same task string used in training (temporal ensembling [\[2\]](#references)).

- **`inference/agilex_inference_openpi_rtc.py`**  
  Set the global `lang_embeddings` at the top of the file; it is sent as `"prompt"` in the RTC inference payload.

Other inference entrypoints (e.g. sync) also need the same prompt set in their payload (same variable name or equivalent field).

### If the model was trained with AWBC (advantage-weighted)

Models trained with **Advantage-Weighted Behavior Cloning** (see [stage advantage pipeline](../../../stage_advantage/README.md)) are conditioned on prompts that include an advantage label, e.g.:

- `"<task>, Advantage: negative"`
- `"<task>, Advantage: positive"`

At inference you must use the **same format**. For high-advantage behavior use the **positive** form, e.g.:

```python
lang_embeddings = "fold the cloth, Advantage: positive"
```

Use the same `<task>` wording as in your training config. Using a different format or omitting the advantage part can hurt performance. See [stage_advantage/README.md — Inference with an AWBC-trained model](../../../stage_advantage/README.md) for details.

---

## Full setup (one-time and run-time)

Below is the flow **on the IPC** after the policy server is already running on the GPU host (see [Inference Setup](#inference-setup) above). Replace `PKG_ROOT` and paths with your own.

### Prerequisites

- **Piper SDK and robot:** Follow the [Piper SDK](https://github.com/agilexrobotics/piper_sdk) setup first (CAN, SDK, arm control).
- **IPC Python env:** Set up the `kai0_inference` conda env (Python 3.10, PyTorch, [requirements_inference_ipc.txt](requirements_inference_ipc.txt), `openpi-client`) as in [IPC Python environment](#ipc-python-environment-one-time-setup) above.
- **ROS Noetic**, Piper ROS package (e.g. `Piper_ros_private-ros-noetic`) at a path we call `PKG_ROOT`.
- **Policy server** running on the GPU host. You need the GPU host’s **IP** and the server **port** (default 8000).
- **tmux** optional, for a multi-terminal layout.

### Steps on the IPC (manual or via tmux)

1. **Roscore**  
   ```bash
   roscore
   ```

2. **CAN config** (Piper; may need sudo)  
   ```bash
   cd $PKG_ROOT && sudo ./can_config.sh
   ```
   Use your own method to supply the sudo password if required.

3. **RealSense cameras**  
   ```bash
   cd $PKG_ROOT && roslaunch realsense2_camera multi_camera.launch
   ```

4. **Piper arms** (master/slave, mode 1, auto enable)  
   ```bash
   cd $PKG_ROOT && roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
   ```

5. **Inference node**  
   From the repo root or the directory that contains `inference/` (with the `kai0_inference` conda env active):
   ```bash
   conda activate kai0_inference
   python inference/agilex_inference_openpi_temporal_smoothing.py --host <gpu_host_ip> --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50
   ```
   **Important:** Set **`--host`** to the **IP of the GPU host** where `serve_policy` is running (e.g. `192.168.1.10`). Use `--port` equal to the port you started the server with (default 8000).

   Alternatively use `agilex_inference_openpi_temporal_ensembling.py` with `--smooth_method temporal_ensembling` [\[2\]](#references) (see script `--help`). Set **`lang_embeddings`** (or equivalent prompt) in the script as in [Prompt and AWBC](#prompt-and-awbc-important) above.

### Optional: multi-pane layout with tmux

- Session name, e.g. `piper_demo`.
- Create session and split into panes (e.g. 2×2 or tiled).
- In each pane run one of the commands above (roscore, can_config, realsense, piper, inference).
- Logs: run each command with `2>&1 | tee $LOG_DIR/<pane_name>.log` if desired.
- To stop: `tmux kill-session -t piper_demo` (or attach and Ctrl+C in each pane).

No standalone `start_demo_*.py` is provided; use the steps above and set `PKG_ROOT`, `DEMO_ROOT`, and `LOG_DIR` to your paths.

---

## Inference scripts

| Script | Description |
|--------|-------------|
| `inference/agilex_inference_openpi_temporal_smoothing.py` | Temporal smoothing; prompt set via `lang_embeddings`. |
| `inference/agilex_inference_openpi_temporal_ensembling.py` | Supports `--smooth_method naive_async` or `temporal_ensembling` [\[2\]](#references); prompt via `lang_embeddings`. |
| `inference/agilex_inference_openpi_sync.py` | Synchronous inference. |
| `inference/agilex_inference_openpi_rtc.py` | **RTC (real-time chunking):** uses `prev_action_chunk` and delay for chunk alignment [\[1\]](#references); includes temporal smoothing over chunk boundaries. **Server must be started with an RTC config** — see [RTC mode](#rtc-real-time-chunking-mode) above. Run with `--rtc_mode`. |

Set **`--host`** to the **GPU host IP** (where `serve_policy` is running) and **`--port`** to the server port (default 8000). Ensure the prompt in the script matches training (and, for AWBC, use the positive-advantage format as in [stage_advantage](../../../stage_advantage/README.md)).

---

## References

1. **Black, K., Galliker, M. Y., & Levine, S. (2025).** *Real-Time Execution of Action Chunking Flow Policies.* arXiv preprint arXiv:2506.07339.

2. **Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023).** *Learning fine-grained bimanual manipulation with low-cost hardware.* arXiv preprint arXiv:2304.13705.

BibTeX:

```bibtex
@misc{black2025realtime,
  author    = {Black, Kevin and Galliker, Manuel Y. and Levine, Sergey},
  title     = {Real-Time Execution of Action Chunking Flow Policies},
  year      = {2025},
  eprint    = {2506.07339},
  archivePrefix = {arXiv},
  primaryClass  = {cs}
}

@misc{zhao2023learning,
  author    = {Zhao, Tony Z. and Kumar, V. and Levine, Sergey and Finn, Chelsea},
  title     = {Learning fine-grained bimanual manipulation with low-cost hardware},
  year      = {2023},
  eprint    = {2304.13705},
  archivePrefix = {arXiv},
  primaryClass  = {cs}
}
```

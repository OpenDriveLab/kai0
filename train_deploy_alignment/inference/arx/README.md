# ARX Inference

Deploy trained OpenPi policies on **ARX-X5** dual-arm robots (RealSense cameras, ROS2). Inference runs by sending observations to a **policy server** (on a GPU host) and applying the returned action chunks on the robot, with optional temporal smoothing [\[2\]](#references) or **RTC (real-time chunking)** [\[1\]](#references).

---

## Prerequisites: ARX-X5 and ROS2 setup

Follow the **official ARX-X5** setup so arms and cameras work correctly:

- **[ARX_X5 (ARX Robotics)](https://github.com/ARXroboticsX/ARX_X5)**  
  Clone the repo, follow the docs for:
  - ROS2 workspace (e.g. `ROS2/X5_ws`), building packages
  - Arm control and feedback topics
  - RealSense camera setup

After the ARX-X5 and robot hardware are set up, install the **Python 3.10 inference environment** (below) and **build the bimanual package** in this repo.

---

## Python 3.10 inference environment (one-time setup)

On the machine that runs the inference client (IPC or same host as ROS2), use a **dedicated Python 3.10 environment**.

### 1. Create and activate conda env

```bash
conda create -n kai0_inference python=3.10
conda activate kai0_inference
```

### 2. Install dependencies

Install PyTorch (if needed for local use), OpenCV, NumPy, and other deps. From the **repository root** you can reuse the Agilex IPC requirements if present:

```bash
pip install -r train_deploy_alignment/inference/agilex/requirements_inference_ipc.txt
```

(or install `opencv-python`, `numpy`, `pyrealsense2`, etc. as needed.)

### 3. Install OpenPi client (editable)

From the **repository root**:

```bash
cd packages/openpi-client
pip install -e .
cd ../..
```

The inference scripts use `openpi_client` to talk to the policy server.

### 4. Build the bimanual package (ARX)

From the **ARX inference directory**:

```bash
cd train_deploy_alignment/inference/arx
./build.sh
```

This builds the **bimanual** package (e.g. `cd bimanual && ./build.sh`). Ensure your environment can load the built libraries (e.g. `LD_LIBRARY_PATH` as in [setup.sh](setup.sh)).

### 5. ROS2 and ARX messages

Source your ROS2 workspace and ensure the **arx5_arm_msg** (or equivalent) package is built and sourced so that `RobotStatus` / `RobotCmd` are available. The inference scripts import `arx5_arm_msg.msg`.

---

## Inference setup (startup sequence)

Inference uses two machines: a **GPU host** (policy server) and the **client machine** (ROS2 + inference script). Follow the order below.

### On the GPU host — start the policy server

From the **repository root** on the GPU machine:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<train_config> --policy.dir=<checkpoint_dir> [--port=8000]
```

Use the same training config and checkpoint as your trained model. For **RTC inference**, use an RTC config (e.g. `pi0_rtc_aloha_sim` or `pi05_rtc_flatten_fold_inference`); see [RTC mode](#rtc-real-time-chunking-mode) below.

### On the client — full startup steps

Before running inference (or DAGGER), **CAN must be configured and up** (per the **ARX official repo**; this repo does not provide CAN scripts), and you must **enable both master and slave arms**. Order:

1. **Ensure CAN is configured and up** (follow ARX official ARX_X5 / ARX_CAN setup).

2. **Enable both master and slave arms.** Start the master and slave controller nodes (e.g. in separate terminals, or use the DAGGER **arx_start.sh** in [dagger/arx](../../dagger/arx/README.md) to start both):

   ```bash
   ros2 launch arx_x5_controller open_remote_master.launch.py
   ros2 launch arx_x5_controller open_remote_slave.launch.py
   ```

   Wait until nodes are up (e.g. `ros2 node list` shows the arm nodes).

3. **Source ROS2** and (if needed) your conda env:

   ```bash
   source /path/to/ros2_ws/install/setup.bash
   conda activate <your_inference_env>
   ```

4. **Source ARX setup** (for `LD_LIBRARY_PATH` so bimanual libs load). From the **arx** directory:

   ```bash
   cd train_deploy_alignment/inference/arx
   source setup.sh
   ```

5. **Run the inference script** from the **arx** directory, with `--host` set to the GPU host IP:

   ```bash
   cd train_deploy_alignment/inference/arx
   source setup.sh
   cd inference
   python arx_openpi_inference_rtc.py --host <policy_server_ip> --port 8000 --rtc_mode --chunk_size 50
   ```

   Or run another script (see [Inference scripts](#inference-scripts) below). Replace `<policy_server_ip>` with your policy server IP.

---

## RTC (real-time chunking) mode

**RTC** stands for **real-time chunking** [\[1\]](#references). For RTC inference, the policy server must load the **RTC model** (Pi0RTC). Start the server with an RTC config, e.g.:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_rtc_aloha_sim --policy.dir=<path_to_jax_checkpoint> [--port=8000]
```

Then run the ARX inference script with **`--rtc_mode`** (e.g. `arx_openpi_inference_rtc.py --rtc_mode`). RTC uses JAX checkpoints only.

---

## Prompt and AWBC

Set the **language prompt** in the inference script to match training. In the scripts, set the global **`lang_embeddings`** at the top (e.g. `lang_embeddings = "hang the cloth"`). For **AWBC**-trained models, use the same advantage format as in [stage_advantage](../../../stage_advantage/README.md) (e.g. `"<task>, Advantage: positive"`).

---

## Inference scripts

Run from `train_deploy_alignment/inference/arx` after `source setup.sh`, then `cd inference`:

| Script | Description | Example command |
|--------|-------------|-----------------|
| `inference/arx_openpi_inference_rtc.py` | **RTC** [\[1\]](#references) with `--rtc_mode`; without it, same as temporal smoothing. Server must use RTC config for RTC. | `python arx_openpi_inference_rtc.py --host <IP> --port 8000 --rtc_mode --chunk_size 50` |
| `inference/arx_openpi_inference_temporal_smooth.py` | Temporal smoothing; async inference + stream buffer. | `python arx_openpi_inference_temporal_smooth.py --host <IP> --port 8000` |
| `inference/arx_openpi_inference_sync.py` | Sync: blocking infer every chunk, then execute step-by-step (like Agilex sync). | `python arx_openpi_inference_sync.py --host <IP> --port 8000` |
| `inference/arx_openpi_inference_temporal_ensembling.py` | Temporal ensembling [\[2\]](#references): `--smooth_method naive_async` or `temporal_ensembling`, `--exp_weight_m` for aggregation. | `python arx_openpi_inference_temporal_ensembling.py --host <IP> --smooth_method temporal_ensembling --exp_weight_m 0.01` |

- **`--host`**: GPU host IP. **`--port`**: server port (default 8000).
- **`lang_embeddings`**: Set in the script (or in `arx_openpi_inference_rtc` for scripts that import it) to match training.

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

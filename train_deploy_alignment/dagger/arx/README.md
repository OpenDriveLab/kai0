# ARX DAGGER (data collection with policy-in-the-loop)

DAgger data collection on the **ARX-X5** dual-arm setup: the policy runs on a remote GPU server (WebSocket); the IPC runs ROS2, ARX master/slave arms, and RealSense cameras. The collection script runs inference, lets you intervene with the master arms (press **d**), and saves episodes with optional intervention labels and videos.

See also: [DAGGER overview](../README.md) · [ARX inference](../../inference/arx/README.md)

---

## Prerequisites

- **ARX-X5 dual-arm system**: two master arms and two slave arms. **CAN** must be configured per the **official ARX-X5 / ARX_CAN** setup before running DAGGER or inference; this repo does not provide CAN activation scripts.
- **RealSense cameras** and ROS2 (e.g. same camera setup as [ARX inference](../../inference/arx/README.md)).
- **Policy server** on another machine (same as inference): start with `serve_policy`, connect IPC over network.

Python on the IPC: use a conda env with **openpi-client** and deps (see [ARX inference README](../../inference/arx/README.md)); e.g. create one with `conda create -n arx_inference python=3.10` and install dependencies there.

---

## Step 1: DAGGER setup (on the IPC, one-time)

### 1.1 Build the ROS2 workspace (X5_ws)

The ARX controller and messages live in **X5_ws**, a **colcon** (ROS2) workspace. From the **dagger/arx** directory:

```bash
cd train_deploy_alignment/dagger/arx/X5_ws
colcon build
source install/setup.bash
```

Use `source install/setup.bash` whenever you open a new terminal to run the DAGGER script or launch the controller.

### 1.2 Build the bimanual package

The DAGGER script uses the **bimanual** Python package (native .so). From the **dagger/arx** directory:

```bash
cd train_deploy_alignment/dagger/arx
./build.sh
```

This runs `cd bimanual && ./build.sh`, which does a CMake build and installs into `bimanual/api/` (e.g. `arx_x5_python/*.so`). Ensure the loader can find the libs by **sourcing `setup.sh`** before running the script (see Step 2).

---

## Step 2: DAGGER launch sequence

Same idea as inference: start the policy server on the GPU machine, then on the IPC start ROS2, enable arms, and run the DAGGER script. **CAN must already be configured and up** (per ARX official repo).

### On the inference machine (GPU server)

From the **repository root**:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir> [--port=8000]
```

Note the server IP for use on the IPC (e.g. `<policy_server_ip>`).

### On the IPC (in order)

1. **Source ROS2 and ARX workspace**:

   ```bash
   source /path/to/your/ros2_install/setup.bash
   cd train_deploy_alignment/dagger/arx/X5_ws
   source install/setup.bash
   ```

2. **Enable arms (master + slave).** Either use **arx_start.sh** or start the nodes manually:

   **Option A — arx_start.sh:**

   ```bash
   cd train_deploy_alignment/dagger/arx
   ./arx_start.sh
   ```

   **Option B — Manual:** In separate terminals:

   ```bash
   ros2 launch arx_x5_controller open_remote_master.launch.py
   ros2 launch arx_x5_controller open_remote_slave.launch.py
   ```

3. **Run the DAGGER collection script** (in another terminal):

   ```bash
   cd train_deploy_alignment/dagger/arx
   source setup.sh
   conda activate <your_inference_env>
   python arx_openpi_dagger_collect.py --host <policy_server_ip> --port 8000 --dataset_dir <path/to/dagger/data> --dataset_name <run_name>
   ```

   Replace `<policy_server_ip>`, `<path/to/dagger/data>`, and `<run_name>` with your policy server IP, dataset root path, and run name. The script will connect to the server, run homing (if `--auto_homing`), then wait for **Enter** to start inference.

---

## Usage after launch (keyboard controls)

Once the script is running:

| Key | Action |
|-----|--------|
| **Enter** | Start inference (after homing and prompt). |
| **d** (or **Ctrl+Q**) | **Enter DAgger mode.** Inference pauses; master arms are enabled and move to align with the slave arms. **Safety:** keep clear when you press **d**. Then you can teleoperate with the master arms. |
| **Space** | **Start recording** the current segment (in DAgger mode). Press Space when ready to demonstrate. |
| **s** | **Save** the current episode (HDF5 + optional videos + intervention JSON). You can press **s** anytime; save runs in the background. |
| **r** | **Exit DAgger** and resume inference. |
| **Ctrl+C** | Exit the program. |

### Typical flow

1. Press **Enter** to start inference (policy controls slave arms).
2. When you want to correct or demonstrate: press **d** → wait for master alignment (stay clear) → press **Space** to start recording → move master arms → press **s** to save when done.
3. Press **r** to resume inference. Repeat **d** / **Space** / **s** / **r** as needed.
4. Each **s** writes one episode; episode index increments automatically.

---

## Data storage

- **Root**: `--dataset_dir` (set to your dataset root, e.g. `~/data/dagger`).
- **Per run**: `{dataset_dir}/{dataset_name}/` (e.g. `~/data/dagger/my_run/`).
- **Per episode**:  
  - `episode_0.hdf5`, `episode_1.hdf5`, ... — joint observations and actions (no images in HDF5).  
  - `episode_0_intervention.json` (optional) — intervention labels per frame (0 = policy, 1 = human).  
  - Optional camera videos under a `video/` (or similar) tree when `--save_video` is set.

Episode index increments after each save. Set `--dataset_name` to group runs (e.g. by task or date).

---

## Scripts and files

| File | Purpose |
|------|---------|
| `arx_start.sh` | Enable arms only: check no master/slave running, then start master + slave nodes. Run DAGGER script separately (see README). |
| `setup.sh` | Set `LD_LIBRARY_PATH` for bimanual; source before running the script. |
| `build.sh` | Build bimanual (CMake). |
| `arx_openpi_dagger_collect.py` | Main DAGGER script (inference + intervention + save). |

---

## Quick reference (IPC launch order)

```text
1. Ensure CAN is configured and up (per ARX official repo).
2. source ROS2 + X5_ws:  source X5_ws/install/setup.bash
3. ./arx_start.sh   # enable master + slave only
4. In another terminal: source setup.sh && conda activate <your_inference_env> && python arx_openpi_dagger_collect.py --host <policy_server_ip> --port 8000 --dataset_dir <path/to/dagger/data> --dataset_name <run_name>
```

Replace `<policy_server_ip>`, `<path/to/dagger/data>`, `<run_name>`, and `<your_inference_env>` with your values. For inference-only deployment (no DAgger), see [ARX inference](../../inference/arx/README.md).

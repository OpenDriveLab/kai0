# Agilex DAgger (data collection with policy-in-the-loop)

DAgger data collection on the **Agilex** dual-arm setup: run the policy on an inference machine, and on the IPC (industrial PC) run ROS + master/slave arms and RealSense cameras. The collection script runs inference, switches to DAgger mode for human demonstration, and saves episodes under a configurable dataset name.

---

## Prerequisites

- **Agilex Piper master-slave dual-arm system**: two master arms and two slave arms, each connected to the IPC via **USB-to-CAN** (4 adapters in total).
- **RealSense cameras** and `realsense2_camera` ROS package (e.g. `multi_camera.launch`).
- **Policy server** runs on another machine (inference server) reachable over the network; same setup as [inference](../inference/agilex/).

Python environment on the IPC: use the same **kai0_inference** conda env as in inference (see [ARX inference README](../../inference/arx/README.md) for conda, `pip install openpi-client`, etc.).

---

## Step 1: DAGGER setup (on the IPC, one-time)

### 1.1 Install CAN and network tools

```bash
sudo apt update && sudo apt install can-utils ethtool
```

### 1.2 Identify CAN ports for all four arms

Connect the four USB-to-CAN adapters (two for slave arms, two for master arms). Then run:

```bash
cd train_deploy_alignment/dagger/agilex
./find_all_can_port.sh
```

This lists each CAN interface and its **USB bus-info** (e.g. `1-13:1.0`, `1-12:1.0`). Note which bus-info corresponds to which arm (left/right, master/slave) by unplugging and re-running if needed.

### 1.3 Configure and activate CAN interfaces

Edit **`activate_can_arms.sh`** and set the four USB bus-info strings to match your setup. The launch file expects these interface names:

- `can_left_slave`, `can_right_slave` — slave arms  
- `can_left_mas`, `can_right_mas` — master arms  

Example (replace with your bus-info from step 1.2):

```bash
bash ./can_activate.sh can_left_slave  1000000 "1-13:1.0"
bash ./can_activate.sh can_right_slave 1000000 "1-12:1.0"
bash ./can_activate.sh can_left_mas    1000000 "1-6:1.0"
bash ./can_activate.sh can_right_mas  1000000 "1-5:1.0"
```

Then run:

```bash
./activate_can_arms.sh
```

Check with `ip link show` that the four CAN interfaces are up with the correct names.

### 1.4 Build the workspace

From the **dagger/agilex** directory (or your catkin workspace that contains `piper` and `piper_msgs`):

```bash
cd train_deploy_alignment/dagger/agilex
catkin_make
# or: cd your_ws && catkin_make
```

Source the workspace when launching: `source devel/setup.bash` (see Step 2).

---

## Step 2: DAGGER launch sequence

Same idea as inference: start the policy server on the inference machine, then on the IPC start ROS, CAN, cameras, piper nodes, and finally the DAGGER collection script.

### On the inference machine (GPU server)

Start the policy server (from repo root):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir> [--port=8000]
```

Connect the IPC to this machine (e.g. Ethernet). Note the server IP (e.g. `192.168.1.10`).

### On the IPC (in order)

1. **Start roscore** (in a dedicated terminal):

   ```bash
   roscore
   ```

2. **Go to the DAGGER directory and activate CAN**:

   ```bash
   cd train_deploy_alignment/dagger/agilex
   ./activate_can_arms.sh
   ```

3. **Launch RealSense cameras**:

   ```bash
   roslaunch realsense2_camera multi_camera.launch
   ```

4. **Source workspace and launch piper (master + slave)**:

   ```bash
   source devel/setup.bash
   roslaunch piper start_ms_piper_new.launch
   ```

5. **Activate conda and run the DAGGER collection script**:

   ```bash
   conda activate kai0_inference
   python agilex_openpi_dagger_collect.py --host 192.168.1.10 --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50 --dataset_name flat_1101_27_100_mixed1.0
   ```

Replace `--host` with your policy server IP and `--dataset_name` with your desired dataset name.

---

## Usage after launch (keyboard controls)

Once the script is running, the robot is in **inference mode**: the policy controls the slave arms. Use the following keys to intervene and collect DAgger data.

### Entering DAgger mode (human intervention)

| Key | Action |
|-----|--------|
| **d** | **Activate DAgger mode.** Inference is paused. The script saves the **inference-phase** data (policy output so far) to the inference dataset, then **enables the master arms** and moves them to match the slave positions (safe pose first, then slave pose). **⚠️ Safety:** Keep clear of the arms when you press **d**; the masters will power on and move. After that, you can teleoperate by dragging the master arms. |

### Recording the human demonstration

| Key | Action |
|-----|--------|
| **Space** | **Start recording** the DAgger segment. Press Space after the masters have reached the slave pose and you are ready to demonstrate. From this moment, frames (observations + actions from master/slave) are collected. |
| **s** | **End and save** the current DAgger episode. Press **s** when you have finished the demonstration. The episode is written to disk (HDF5 + optional videos). |

### After saving an episode

A short prompt appears (about 10 seconds):

| Key | Action |
|-----|--------|
| **w** | **Delete** the episode you just saved (e.g. if the data quality is bad). The HDF5 and any exported videos for this episode are removed. |
| **Any other key** | **Confirm** the data is good. The episode is kept and the episode index increments. |

### Resuming inference

| Key | Action |
|-----|--------|
| **r** | **Exit DAgger mode** and **resume inference**. The policy takes over the slave arms again. You can press **d** again later to collect another DAgger segment. |

### Typical flow

1. Let the policy run (inference mode).
2. When you want to correct or demonstrate: press **d** → wait for masters to enable and move to slave pose (stay clear).
3. Press **Space** to start recording, then move the master arms to demonstrate.
4. Press **s** to end and save the episode.
5. If the capture was bad, press **w** to delete it; otherwise press another key to keep it.
6. Press **r** to return to inference, or press **Space** again (after step 5) to start a new DAgger recording without leaving DAgger mode.

---

## Dataset storage

- **Root directory**: `/home/agilex/data` (configurable via `--dataset_dir`).
- **Inference-phase data** (saved when you press **d**):  
  `{dataset_dir}/{dataset_name}_inference_hdf5/aloha_mobile_dummy/episode_*.hdf5`
- **DAgger (human) data** (saved when you press **s**):  
  `{dataset_dir}/{dataset_name}_dagger_hdf5/aloha_mobile_dummy/episode_*.hdf5`
- **Per episode** (same structure for both):
  - `episode_0.hdf5`, `episode_1.hdf5`, ... — HDF5 with `/observations/qpos`, `/observations/qvel`, `/observations/effort`, `/action`, `/base_action`
  - Optional videos under `video/{camera_name}/episode_{i}.mp4` when export is enabled.

Episode indices are independent for inference vs DAgger. Set `--dataset_name` to group runs (e.g. by task or date).

---

## Scripts and files

| File | Purpose |
|------|--------|
| `find_all_can_port.sh` | List CAN interfaces and USB bus-info (run after plugging in USB-CAN adapters). |
| `can_activate.sh` | Activate a single CAN interface by name, bitrate, and optional USB bus-info. |
| `activate_can_arms.sh` | Activate all four CAN interfaces (edit bus-info before use). |
| `agilex_openpi_dagger_collect.py` | Main DAGGER script: inference + DAgger mode and episode saving. |
| `collect_data_1010.py` | Helper module used by the collector (`save_data`, `create_video_from_images`, `CollectOperator`). |

---

## Quick reference (IPC launch order)

```text
1. roscore
2. cd train_deploy_alignment/dagger/agilex && ./activate_can_arms.sh
3. roslaunch realsense2_camera multi_camera.launch
4. source devel/setup.bash && roslaunch piper start_ms_piper_new.launch
5. conda activate kai0_inference && python agilex_openpi_dagger_collect.py --host <IP> --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50 --dataset_name <your_dataset_name>
```

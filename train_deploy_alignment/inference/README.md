# Inference

This directory contains deployment and inference code for running trained OpenPi policies on real robots. Implementations include:

- **Temporal chunk-wise smoothing** — smooth execution across action chunks (e.g. exponential decay, temporal ensembling [\[2\]](arx/README.md#references)) to reduce jerk and improve tracking.
- **RTC (real-time chunking)** — real-time execution of action-chunking flow policies [\[1\]](arx/README.md#references): the policy server streams chunks; the client executes with low latency and optional smoothing.

Details and script options (e.g. `--rtc_mode`, `--smooth_method`) are in the platform READMEs below.

Two robot platforms are supported:

| Robot | Stack | Description |
|-------|--------|-------------|
| **Agilex** | ROS Noetic, Piper arms, RealSense cameras | Dual-arm bimanual manipulation (e.g. cloth folding). OpenPi client; optional temporal smoothing / ensembling. See [agilex/README.md](agilex/README.md). |
| **ARX** | ROS 2, ARX X5 arms | Dual-arm manipulation (e.g. hang cloth). **RTC** and **temporal chunk-wise smoothing** (sync, async, ensembling). See [arx/README.md](arx/README.md). |

---

## Two-machine setup (policy server + robot)

Inference is split across two machines:

1. **GPU host (e.g. 4090)** — Run the **policy server** so the trained model is served over the network.
2. **Robot / industrial PC** — Run ROS, cameras, arms, and the **inference client** script that sends observations to the server and executes the returned actions.

**On the GPU host**, from the repo root:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<train_config_name> --policy.dir=<checkpoint_dir> [--port=8000]
```

Example (use the same config name and checkpoint path you used for training):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

Default port is `8000`. The server binds to `0.0.0.0` so the industrial PC can connect.

**On the industrial PC**, start the robot stack and inference client. In the inference script (e.g. Agilex), set **`--host`** to the **GPU host’s IP address** (so the client connects to the policy server). Use the same `--port` as on the server (default 8000). See [agilex/README.md](agilex/README.md) for the full sequence (Piper SDK setup → ROS → inference script with `--host <gpu_host_ip>`).

---

**Important:** Set the **language prompt** in the inference code to match the training config. For **AWBC**-trained models, use the same prompt format (including the positive-advantage form). See [agilex/README.md#prompt-and-awbc](agilex/README.md#prompt-and-awbc) and the [stage advantage README](../../stage_advantage/README.md).

- **Agilex:** [agilex/README.md](agilex/README.md) — Piper SDK ref, setup, serve_policy + inference flow, prompt / AWBC.  
- **ARX:** [arx/README.md](arx/README.md) — **RTC** and **temporal chunk-wise smoothing** scripts, device setup, inference flow, data collection.

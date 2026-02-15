# DAGGER (data collection with policy-in-the-loop)

This module provides **DAgger-style data collection** for robot platforms: the policy runs on a remote GPU server; the IPC runs ROS/ROS2, arm controllers, and cameras. Operators can switch into DAgger mode to demonstrate corrections, then save episodes (HDF5, optional videos, intervention labels).

---

## Platforms

| Platform | Directory | Description |
|----------|-----------|-------------|
| **ARX-X5** | [arx/](arx/) | Dual-arm ARX-X5 (ROS2, master/slave, RealSense). CAN configured per ARX official repo. |
| **Agilex** | [agilex/](agilex/) | Dual-arm Agilex Piper (ROS, master/slave, RealSense). Includes CAN scripts. |

See each platform’s README for prerequisites, one-time setup (workspace build, CAN if applicable), and launch order (policy server → CAN/arms → collection script).

---

## Common flow

1. **Policy server** (GPU machine): `serve_policy` from the repo root.
2. **IPC**: Source ROS/ROS2, ensure CAN is up (if required), enable arms (master + slave), then run the platform’s DAGGER collection script.
3. **During collection**: Start inference (Enter), enter DAgger mode (e.g. **d**), record (Space), save (s), resume (r).

Details and keybindings are in [arx/README.md](arx/README.md) and [agilex/README.md](agilex/README.md).

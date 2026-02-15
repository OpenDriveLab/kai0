# Hardware Setup & 3D Print Files

Quick reference for deploying and debugging hardware for the supported task platforms. All 3D-printed parts use English filenames in the task folders below.

---

## Table of Contents

- [1. Task_A / Task_B (Agilex Piper)](#1-task_a--task_b-agilex-piper)
- [2. Task_C (ARX X5)](#2-task_c-arx-x5)
- [3. Inference Host](#3-inference-host)

---

## 1. Task_A / Task_B (Agilex Piper)

**Directories:** `Task_A/`, `Task_B/`

### 1.1 Components

| Component | Specification |
|-----------|---------------|
| Arm | Agilex Piper |
| Cameras | Intel RealSense D435 (triple-camera setup) |
| Printed parts | Left/right wrist camera mounts, center camera mount, center camera base |

### 1.2 Task_A Layout

| Parameter | Value |
|-----------|-------|
| Center camera height (from table top) | 76 cm |
| Left secondary arm → table front edge | 18 cm |
| Right secondary arm → table front edge | 18 cm |
| Center base → left secondary arm center | 34 cm |
| Center base → right secondary arm center | 34 cm |
| Left primary arm → table front edge | 18 cm |
| Right primary arm → table front edge | 12 cm |
| Left–right primary arm center distance | 39 cm |

### 1.3 Task_B Layout (demoA-style)

| Parameter | Value |
|-----------|-------|
| Center camera height (from table top) | 93 cm |
| Left secondary arm → table front edge | 27 cm |
| Right secondary arm → table front edge | 27 cm |
| Center base → left secondary arm center | 16 cm |
| Center base → right secondary arm center | 16 cm |
| Left primary arm → table front edge | 18 cm |
| Right primary arm → table front edge | 11 cm |
| Left–right primary arm center distance | 40 cm |

### 1.4 3D Models — Usage (Task_A / Task_B)

#### Gripper (end-effector)

| File | Format | Use |
|------|--------|-----|
| `agilex-gripper.STEP` | STEP | CAD import / interchange. Use for design or export to other tools. |
| `agilex-gripper.SLDPRT` | SolidWorks | Native part. Edit or export to STEP/3MF from here. |
| `agilex-gripper.3mf` | 3MF | **Use for 3D printing.** Load in your slicer (Cura, PrusaSlicer, etc.); contains mesh and is print-ready. |
| `agilex-gripper-modified.STEP` | STEP | Modified gripper assembly. Same use as above when the modified design is required. |
| `agilex-gripper-modified.SLDPRT` | SolidWorks | Modified gripper native part. |
| `agilex-gripper-soft.SLDPRT` | SolidWorks | Soft gripper variant. Use if deploying the soft gripper; export to STEP or 3MF for printing. |

**Usage:** Mount the printed gripper on the **wrist flange of each Agilex Piper arm** (left and right). Use either the standard or modified version per your build; use the soft variant only if specified for the task.

#### Camera mounts

| File | Format | Use |
|------|--------|-----|
| `camera-bottom-mount-bracket.STEP` | STEP | Camera bottom bracket — CAD/slicer. |
| `camera-bottom-mount-bracket.SLDPRT` | SolidWorks | Same bracket, native. |
| `end-effector-camera-mount-d435i-centered.STEP` | STEP | **End-effector camera mount** for Intel RealSense D435i, optical center aligned. |

**Usage:** Print the **camera-bottom-mount-bracket** and use it at the **center camera** base (table-mounted). Print **end-effector-camera-mount-d435i-centered** and attach it to the **wrist of each arm**; mount one D435i per arm for the left/right wrist cameras. Follow the layout tables in §1.2 / §1.3 for heights and distances.

---

## 2. Task_C (ARX X5)

**Directory:** `Task_C/`

### 2.1 Components

| Component | Specification |
|-----------|---------------|
| Arm | ARX X5 |
| Cameras | Intel RealSense D435 (triple-camera setup) |
| Printed parts | Left/right secondary arm grippers, left/right wrist camera mounts, center camera mount, center camera base |

### 2.2 Layout

| Parameter | Value |
|-----------|-------|
| Center camera height (from table top) | 93 cm |
| Left secondary arm base → table front edge | 45 cm |
| Right secondary arm base → table front edge | 87 cm |
| Center base → left secondary arm center | 22 cm |
| Center base → right secondary arm center | 22 cm |
| Left primary arm → table front edge | 18 cm |
| Right primary arm → table front edge | 11 cm |
| Left–right primary arm center distance | 53 cm |

### 2.3 3D Models — Usage (Task_C)

#### Grippers (secondary arms)

| File | Format | Use |
|------|--------|-----|
| `gripper-1.STEP`, `gripper-1.SLDPRT` | STEP / SolidWorks | **Left secondary arm gripper.** Print and mount on the left secondary arm flange. |
| `gripper-2.STEP`, `gripper-2.SLDPRT` | STEP / SolidWorks | **Right secondary arm gripper.** Print and mount on the right secondary arm flange. |

**Usage:** Use STEP for slicing or CAD; use SLDPRT to edit in SolidWorks. Print one of each and install on the corresponding secondary arm.

#### Bases (table-mounted)

| File | Format | Use |
|------|--------|-----|
| `left-right-base.SLDPRT` | SolidWorks | Left and right base parts. Export to STEP or 3MF for printing if needed. |
| `center-base.SLDPRT` | SolidWorks | Center base. Same workflow. |
| `base-integrated.STEP`, `base-integrated.SLDPRT` | STEP / SolidWorks | **Integrated base** (single assembly). Use when building the unified base; print from STEP or export 3MF from SolidWorks. |
| `left-right-base-drawing.SLDDRW`, `left-right-base-drawing.PDF` | Drawing | Assembly/dimension reference for left and right bases. |
| `center-base-drawing.SLDDRW` | Drawing | Reference for center base. |
| `base-integrated-drawing.SLDDRW` | Drawing | Reference for integrated base. |

**Usage:** Print either the separate bases (left, right, center) or the **base-integrated** version. Place on the table according to §2.2. Use the PDF/SLDDRW drawings for assembly and bolt positions.

#### Camera mounts

| File | Format | Use |
|------|--------|-----|
| `arx-d435-centered.STEP`, `arx-d435-centered.SLDPRT` | STEP / SolidWorks | **ARX D435 centered mount.** Primary wrist camera mount for D435; use on left and right arms. |
| `camera-bottom-mount-bracket.STEP`, `camera-bottom-mount-bracket.SLDPRT` | STEP / SolidWorks | Camera bottom bracket for the **center (table-mounted) camera**. |
| `arx-camera-mount-v1.STEP`, `arx-camera-mount-v1.3mf` | STEP / 3MF | ARX camera mount variant 1. Use 3MF for printing. Alternative to `arx-d435-centered` if needed. |
| `arx-camera-mount-v2.STEP` | STEP | Variant 2; use in CAD or slicer. |
| `arx-camera-mount-v2-2mm.STEP` | STEP | Variant 2 with 2 mm adjustment. |
| `arx-camera-mount-v3.STEP` | STEP | Variant 3. |
| `arx-camera-mount-new-v1.STEP` | STEP | New variant 1. Choose the variant that matches your ARX arm and camera setup. |

**Usage:** Print **arx-d435-centered** (or the chosen variant) for each arm’s wrist camera, and **camera-bottom-mount-bracket** for the center camera. Mount D435 units and align with the layout in §2.2.

#### Assembly

| File | Format | Use |
|------|--------|-----|
| `mount-assembly.SLDASM` | SolidWorks assembly | Full mount assembly. Open in SolidWorks to see how grippers, bases, and camera mounts go together; use for BOM and assembly order. |

**Usage:** Reference only; do not print. Use to verify fit and assembly sequence before printing parts.

---

## 3. Inference Host

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4090 (≥ 48 GB VRAM) |

---

**File formats:** Use **STEP** for CAD exchange and most slicers; use **3MF** when provided for direct 3D printing. **SLDPRT / SLDDRW / SLDASM** are for SolidWorks only.

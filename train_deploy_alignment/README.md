# Train–Deploy Alignment

This directory contains three modules used to align training data and deployment/inference:

| Module | Description |
|--------|-------------|
| **dagger** | DAgger-style data collection (policy-in-the-loop, intervention, save). See [dagger/README.md](dagger/README.md) for ARX and Agilex. |
| **inference** | Deployment and inference code, including ARX, Agilex. |
| **data_augment** | Data augmentation and format conversion (time scaling, space mirroring, HDF5 → LeRobot). See [data_augment/README.md](data_augment/README.md). |

See each module’s README for setup and usage.

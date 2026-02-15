# Mini LeRobot

Reading and creating LeRobot-compatible datasets without heavy dependencies. Used by `../convert_h5_lerobot.py` to build LeRobot-format datasets from HDF5 + videos.

**Install (for use with convert_h5_lerobot):**

```bash
# From repo root
uv pip install -e train_deploy_alignment/data_augment/utils/mini_lerobot
```

**Running the converter:** The script also imports the `interface` module from this directory (e.g. `lazy_load_hdf5_dataset_noimg`). When running `convert_h5_lerobot.py`, add this directory to `PYTHONPATH` so `import interface` works:

```bash
cd train_deploy_alignment/data_augment/utils
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini_lerobot"
python convert_h5_lerobot.py ...
```

See [../../README.md](../../README.md) for full usage and options.

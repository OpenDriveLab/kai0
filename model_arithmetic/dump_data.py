"""
python scripts/dump_data.py \
    --dataset pi05_hang_cloth \
    --output hang_cloth_data.pkl

python scripts/dump_data.py \
    --dataset pi05_flat_fold_cloth \
    --output flat_fold_cloth_data.pkl
"""

import argparse
import pickle

from tqdm import tqdm
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.training.sharding as sharding
import jax

def main():
    parser = argparse.ArgumentParser(description="dump test data for model mixture")
    parser.add_argument("--dataset", required=True, type=str, help="Config names used to dump test data from.")
    parser.add_argument("--output", required=True, help="Output directory for dumped data.")

    args = parser.parse_args()
    datasets = args.dataset
    config = _config.get_config(datasets)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    samples_list = []
    for i, samples in enumerate(tqdm(data_loader)):
        if i >= 50:
            break
        samples_list.append(samples)
    with open(args.output, 'wb') as f:
        pickle.dump(samples_list, f)


if __name__ == "__main__":
    main()
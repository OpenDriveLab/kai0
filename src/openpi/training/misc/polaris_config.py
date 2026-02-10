"""PolaRiS baseline policy configs."""

from typing import TypeAlias

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.droid_policy as droid_policy
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


def get_polaris_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import RLDSDroidDataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        # PolaRiS DROID jointpos policies
        #
        TrainConfig(
            name="pi05_droid_jointpos_polaris",
            model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    droid_rlds_dataset.RLDSDataset(
                        name="droid",
                        version="1.0.1",
                        weight=0.9,
                        filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                    ),
                    droid_rlds_dataset.RLDSDataset(
                        name="polaris_droid_cotrain_dataset",
                        version="1.0.0",
                        weight=0.1,
                        filter_dict_path="gs://openpi-assets/droid/polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json",
                    ),
                ),
                rlds_data_dir="<path_to_droid_rlds_dataset>",
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_fast_droid_jointpos_polaris",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=10,
                max_token_len=180,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    droid_rlds_dataset.RLDSDataset(
                        name="droid",
                        version="1.0.1",
                        weight=0.9,
                        filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                    ),
                    droid_rlds_dataset.RLDSDataset(
                        name="polaris_droid_cotrain_dataset",
                        version="1.0.0",
                        weight=0.1,
                        filter_dict_path="gs://openpi-assets/droid/polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json",
                    ),
                ),
                rlds_data_dir="<path_to_droid_rlds_dataset>",
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_droid_jointpos_polaris",
            model=pi0_config.Pi0Config(
                # action_dim=8, # leave as 32 default...
                action_horizon=10,
                max_token_len=100,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    droid_rlds_dataset.RLDSDataset(
                        name="droid",
                        version="1.0.1",
                        weight=0.9,
                        filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                    ),
                    droid_rlds_dataset.RLDSDataset(
                        name="polaris_droid_cotrain_dataset",
                        version="1.0.0",
                        weight=0.1,
                        filter_dict_path="gs://openpi-assets/droid/polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json",
                    ),
                ),
                rlds_data_dir="<path_to_droid_rlds_dataset>",
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        TrainConfig(
            name="pi0_droid_jointpos_100k_polaris",
            model=pi0_config.Pi0Config(
                # action_dim=8, # leave as 32 default...
                action_horizon=10,
                max_token_len=100,
            ),
            data=RLDSDroidDataConfig(
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris/assets",
                    asset_id="droid",
                ),
                datasets=(
                    droid_rlds_dataset.RLDSDataset(
                        name="droid",
                        version="1.0.1",
                        weight=0.9,
                        filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                    ),
                    droid_rlds_dataset.RLDSDataset(
                        name="polaris_droid_cotrain_dataset",
                        version="1.0.0",
                        weight=0.1,
                        filter_dict_path="gs://openpi-assets/droid/polaris_droid_cotrain_dataset_sample_ranges_v1_0_0.json",
                    ),
                ),
                rlds_data_dir="<path_to_droid_rlds_dataset>",
                action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_100k_polaris/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            num_train_steps=1_000,
            batch_size=128,
            log_interval=100,
            save_interval=1000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        ),
        # openpi doesn't support finetuning of binning policies, so this is an inference-only config
        TrainConfig(
            name="paligemma_binning_droid_jointpos",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=600,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                    outputs=[
                        _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                        droid_policy.DroidOutputs(),
                    ],
                ),
            ),
        ),
    ]

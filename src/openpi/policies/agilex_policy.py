"""Policy transforms for the Agilex robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class AgilexInputs(transforms.DataTransformFn):
    """Inputs for the Agilex policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. For normal pi05
      training, names must be exactly the keys of required_rename_map. For advantage
      estimator, optional_rename_map keys may be included as well.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.

    required_rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    # Optional cameras for advantage-estimator training (history frames).
    optional_rename_map = {
        "his_-100_top_head": "base_-100_rgb",
        "his_-100_hand_left": "left_wrist_-100_rgb",
        "his_-100_hand_right": "right_wrist_-100_rgb",
    }

    all_rename_map = {**required_rename_map, **optional_rename_map}

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(required_rename_map.keys())
    EXTRA_CAMERAS: ClassVar[tuple[str, ...]] = tuple(optional_rename_map.keys())
    
    # if set all state to zeros
    mask_state: bool = False

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0/pi0_rtc model, not pi05/pi05_rtc or pi0-FAST
        mask_padding = self.model_type in (_model.ModelType.PI0, _model.ModelType.PI0_RTC)

        in_images = data["images"]

        if set(in_images) - set(self.EXPECTED_CAMERAS) - set(self.EXTRA_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        image_masks = {}
        for camera in self.EXPECTED_CAMERAS + self.EXTRA_CAMERAS:
            if camera in in_images:
                img = in_images[camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.all_rename_map[camera]] = img
                image_masks[self.all_rename_map[camera]] = np.True_

            elif camera not in in_images and camera in self.EXTRA_CAMERAS:
                continue  # optional camera can be skipped
            else:
                raise ValueError(f"Camera {camera} not found in data")


        # filter unnormal state / action value, set to 0
        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        # Prepare inputs dictionary
        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": masked_state,
        }

        # Add actions if present
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            actions = np.where(actions > np.pi, 0, actions)
            actions = np.where(actions < -np.pi, 0, actions)
            if mask_padding:
                # Create action mask for padding
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, self.action_dim:] = False
                inputs["action_mask"] = action_mask

            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        # Advantage-estimator optional fields: passthrough or convert to tensor
        for key in ("frame_index", "episode_length", "progress", "image_original", "episode_index"):
            if key in data:
                inputs[key] = data[key]
        
        def _to_tensor(x, default=None):
            if x is None and default is not None:
                return default
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if isinstance(x, torch.Tensor):
                return x.detach().clone()
            raise NotImplementedError(f"Unsupported type: {type(x)}")

        if "action_advantage" in data:
            inputs["action_advantage"] = _to_tensor(data["action_advantage"], default=torch.tensor(1.0))
        if "action_advantage_original" in data:
            inputs["action_advantage_original"] = _to_tensor(data["action_advantage_original"])
        return inputs


@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex policy."""

    def __call__(self, data: dict) -> dict:
        # Return the first 14 dimensions of actions (13 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :14])} 
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
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    
    # if set all state to zeros
    mask_state: bool = False

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}

        # filter unnormal state / action value, set to 0
        state = np.where(state > np.pi, 0, state)
        state = np.where(state < -np.pi, 0, state)

        # Prepare inputs dictionary
        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "image": images,
            "image_mask": image_mask,
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
            
            if self.convert_to_eef_position:
                actions[..., :14] = batch_qpos_to_eef_pos(actions[..., :14])
            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class AgilexOutputs(transforms.DataTransformFn):
    """Outputs for the Agilex policy."""

    def __call__(self, data: dict) -> dict:
        # Return the first 14 dimensions of actions (13 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :14])} 
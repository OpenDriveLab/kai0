from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import h5py
import numpy as np
import cv2

def load_hdf5_dataset(
    episode_path: str | Path,
) -> dict:
    """Load hdf5 dataset and return a dict with observations and actions"""

    with h5py.File(episode_path) as f:
        state_images_cam_high = np.array(f["observations/images/cam_high"])
        state_images_cam_left_wrist = np.array(f["observations/images/cam_left_wrist"])
        state_images_cam_right_wrist = np.array(f["observations/images/cam_right_wrist"])
        state_qpos = np.array(f["observations/qpos"])

    assert (
        state_images_cam_high.shape[0]
        == state_images_cam_left_wrist.shape[0]
        == state_images_cam_right_wrist.shape[0]
        == state_qpos.shape[0]
    )

    epi_len = state_images_cam_high.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode

def lazy_load_hdf5_dataset(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_images_cam_high = f["observations/images/cam_high"]
    state_images_cam_left_wrist = f["observations/images/cam_left_wrist"]
    state_images_cam_right_wrist = f["observations/images/cam_right_wrist"]
    state_qpos = np.array(f["observations/qpos"])

    epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": state_images_cam_high,
        "observation.images.hand_left": state_images_cam_left_wrist,
        "observation.images.hand_right": state_images_cam_right_wrist,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode, f

def lazy_load_hdf5_dataset_noimg(
    episode_path: str | Path,
) -> Tuple[Dict, h5py.File]:
    """Load hdf5 dataset and return a dict with observations and actions"""
    f = h5py.File(episode_path, 'r')

    state_qpos = np.array(f["observations/qpos"])

    epi_len = state_qpos.shape[0]
    episode = {
        "observation.state": state_qpos.reshape((epi_len, -1)),
        "observation.images.top_head": None,
        "observation.images.hand_left": None,
        "observation.images.hand_right": None,
        "action": state_qpos.reshape((epi_len, -1)),
        "epi_len": epi_len
    }
    return episode, f

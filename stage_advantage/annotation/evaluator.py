"""
Advantage Estimator inference module.

Provides SimpleValueEvaluator: a class that loads a trained Advantage Estimator
checkpoint, reads multi-view video frames, and outputs per-frame advantage
predictions via batched GPU inference with parallel data prefetching.
"""

from __future__ import annotations

import dataclasses
import os
import cv2
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import torch
import safetensors.torch
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from openpi.training import config as _config
from openpi.shared import download
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch_Custom as PI0Pytorch
import openpi.models.tokenizer as _tokenizer
from types import SimpleNamespace
from openpi.shared import image_tools


class SimpleValueEvaluator:
    """Evaluator that runs inference and returns per-frame advantage predictions."""

    def __init__(self, config_name: str, ckpt_dir: str, num_workers: int = 4):
        """
        Args:
            config_name: Training config name (must exist in config registry).
            ckpt_dir: Path to the checkpoint directory containing model.safetensors.
            num_workers: Number of parallel threads for video loading and image preprocessing.
        """
        self.config_name = config_name
        self.ckpt_dir = ckpt_dir
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()
        self.tokenizer = _tokenizer.PaligemmaTokenizer(self.config.model.max_token_len)
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        logging.info(f"Evaluator initialized on device: {self.device}, num_workers: {num_workers}")

    def __del__(self):
        """Shut down the thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def shutdown(self):
        """Explicitly shut down the thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logging.info("Thread pool shut down")

    def _load_model(self):
        """Load model config and weights from checkpoint."""
        self.config = _config.get_config(self.config_name)
        checkpoint_dir = download.maybe_download(self.ckpt_dir)

        new_model = self.config.model.__class__(**{**self.config.model.__dict__})
        self.config = dataclasses.replace(self.config, model=new_model)

        self.model = PI0Pytorch(new_model).to(self.device)
        self.model.eval()
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        logging.info(f"Loading model weights: {model_path}")
        safetensors.torch.load_model(self.model, model_path, strict=True)
        logging.info("Model loaded successfully")

    def _load_video_frames(self, video_path: str, frame_interval: int = 1) -> List[np.ndarray]:
        """
        Load frames from a video file with optional interval sampling.

        Args:
            video_path: Path to the video file.
            frame_interval: Sampling interval (1 = every frame, 2 = every other frame, etc.).

        Returns:
            List of RGB numpy arrays.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                frame_count += 1
        finally:
            cap.release()

        logging.info(
            f"Loaded {len(frames)} frames from {os.path.basename(video_path)} "
            f"(total: {frame_count}, interval: {frame_interval})"
        )
        return frames

    def _load_videos_parallel(
        self,
        video_paths: Tuple[str, str, str],
        frame_interval: int = 1
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load three video files in parallel.

        Args:
            video_paths: Tuple of (top_video, left_video, right_video) paths.
            frame_interval: Sampling interval.

        Returns:
            Tuple of frame lists for (top, left, right) views.
        """
        top_video_path, left_video_path, right_video_path = video_paths

        futures = {
            self._executor.submit(self._load_video_frames, top_video_path, frame_interval): 'top',
            self._executor.submit(self._load_video_frames, left_video_path, frame_interval): 'left',
            self._executor.submit(self._load_video_frames, right_video_path, frame_interval): 'right'
        }

        results = {}
        for future in as_completed(futures):
            video_type = futures[future]
            results[video_type] = future.result()

        return results['top'], results['left'], results['right']

    def _process_single_image(self, rgb_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single RGB image into a model-ready tensor.

        Args:
            rgb_img: RGB numpy array (H, W, 3).

        Returns:
            Tensor of shape (C, H, W), normalized to [-1, 1], resized to 224x224 with padding.
        """
        tensor = torch.from_numpy(rgb_img).float() / 255.0
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor

    def _batch_numpy_to_tensor_parallel(self, np_images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of RGB numpy images to a batched tensor in parallel.

        Args:
            np_images: List of RGB numpy arrays.

        Returns:
            Tensor of shape (batch_size, C, H, W).
        """
        futures = [self._executor.submit(self._process_single_image, img) for img in np_images]
        tensors = [future.result() for future in futures]
        return torch.stack(tensors, dim=0)

    def _batch_numpy_to_tensor(self, np_images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert a list of RGB numpy images to a batched tensor (sequential).

        Args:
            np_images: List of RGB numpy arrays.

        Returns:
            Tensor of shape (batch_size, C, H, W).
        """
        tensors = []
        for rgb_img in np_images:
            tensor = torch.from_numpy(rgb_img).float() / 255.0
            tensor = tensor * 2.0 - 1.0
            tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)
            tensor = tensor.permute(2, 0, 1)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _prepare_batch_tensors(
        self,
        top_frames: List[np.ndarray],
        left_frames: List[np.ndarray],
        right_frames: List[np.ndarray],
        batch_indices: List[int],
        future_indices: List[int],
        initial_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of tensors for current and future frames in parallel.

        Args:
            top_frames, left_frames, right_frames: Frame lists for each camera view.
            batch_indices: Indices of frames in the current batch.
            future_indices: Corresponding future frame indices.
            initial_tensors: Optional pre-computed initial frame tensors.

        Returns:
            Dict with keys 'base_top', 'base_left', 'base_right',
            'future_top', 'future_left', 'future_right'.
        """
        base_top_list = [top_frames[j] for j in batch_indices]
        base_left_list = [left_frames[j] for j in batch_indices]
        base_right_list = [right_frames[j] for j in batch_indices]
        future_top_list = [top_frames[j] for j in future_indices]
        future_left_list = [left_frames[j] for j in future_indices]
        future_right_list = [right_frames[j] for j in future_indices]

        all_lists = [
            base_top_list, base_left_list, base_right_list,
            future_top_list, future_left_list, future_right_list
        ]

        futures = [
            self._executor.submit(self._batch_numpy_to_tensor_parallel, img_list)
            for img_list in all_lists
        ]
        results = [f.result() for f in futures]

        return {
            'base_top': results[0],
            'base_left': results[1],
            'base_right': results[2],
            'future_top': results[3],
            'future_left': results[4],
            'future_right': results[5]
        }

    def evaluate_video_2timesteps_advantages(
        self,
        video_paths: Tuple[str, str, str],
        prompt: str,
        batch_size: int = 8,
        frame_interval: int = 1,
        relative_interval: int = 50,
        min_frame_index: int = None,
        max_frame_index: int = None,
        prefetch: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate advantage using two-timestep comparison (relative + absolute).

        For each frame n, computes:
          - relative_advantage: model(frame_n, frame_{n+relative_interval})
          - absolute_value: model(frame_0, frame_n)
          - absolute_advantage: absolute_value[n+interval] - absolute_value[n]

        Args:
            video_paths: Tuple of (top, left, right) video file paths.
            prompt: Task prompt string.
            batch_size: Batch size for GPU inference.
            frame_interval: Sampling interval (1 = every frame).
            relative_interval: Number of frames ahead for comparison (default: 50).
            min_frame_index: Start frame index (inclusive).
            max_frame_index: End frame index (inclusive).
            prefetch: Whether to prefetch next batch during GPU inference.

        Returns:
            List of dicts, each containing: frame_idx, future_frame_idx,
            relative_advantage, absolute_value, absolute_advantage.
        """
        if len(video_paths) != 3:
            raise ValueError("Expected 3 video paths: (top, left, right)")

        # Load video frames in parallel
        logging.info(f"Loading video frames in parallel (interval: {frame_interval}, workers: {self.num_workers})...")
        top_frames, left_frames, right_frames = self._load_videos_parallel(video_paths, frame_interval)

        # Validate consistent frame counts
        if len(left_frames) != len(top_frames) or len(right_frames) != len(top_frames):
            raise ValueError(
                f"Inconsistent frame counts: top={len(top_frames)}, "
                f"left={len(left_frames)}, right={len(right_frames)}"
            )

        top_frames = top_frames[min_frame_index:max_frame_index+1]
        left_frames = left_frames[min_frame_index:max_frame_index+1]
        right_frames = right_frames[min_frame_index:max_frame_index+1]
        num_frames = len(top_frames)
        if num_frames < 2:
            raise ValueError(f"Insufficient frames: {num_frames}, need at least 2")

        logging.info(f"Total frames after slicing: {num_frames}, relative_interval: {relative_interval}")

        # Preprocess the first frame as the initial reference (for absolute value)
        initial_futures = [
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [top_frames[0]]),
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [left_frames[0]]),
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [right_frames[0]])
        ]
        initial_top_tensor = initial_futures[0].result().to(self.device)
        initial_left_tensor = initial_futures[1].result().to(self.device)
        initial_right_tensor = initial_futures[2].result().to(self.device)

        # Tokenize prompt
        tokens, token_masks = self.tokenizer.tokenize(prompt, state=None)

        all_results = []

        logging.info(f"Starting batched evaluation (n vs n+{relative_interval}, prefetch: {prefetch})...")
        if num_frames <= 0:
            raise ValueError(f"Insufficient frames ({num_frames}), need at least {relative_interval + 1}")

        max_frame_idx = num_frames - 1
        batch_starts = list(range(0, num_frames, batch_size))

        def prepare_batch_data(batch_start: int) -> Tuple[Dict, List[int], int]:
            """Prepare tensors for a single batch."""
            end_idx = min(batch_start + batch_size, num_frames)
            current_batch_size = end_idx - batch_start
            batch_indices = list(range(batch_start, end_idx))
            future_frame_indices = [min(j + relative_interval, max_frame_idx) for j in batch_indices]

            # Collect all images and convert in parallel
            all_images = []
            for j in batch_indices:
                all_images.extend([top_frames[j], left_frames[j], right_frames[j]])
            for fidx in future_frame_indices:
                all_images.extend([top_frames[fidx], left_frames[fidx], right_frames[fidx]])

            tensor_futures = [self._executor.submit(self._process_single_image, img) for img in all_images]
            tensors = [f.result() for f in tensor_futures]

            n = current_batch_size
            base_top = torch.stack(tensors[0:n*3:3], dim=0)
            base_left = torch.stack(tensors[1:n*3:3], dim=0)
            base_right = torch.stack(tensors[2:n*3:3], dim=0)
            future_top = torch.stack(tensors[n*3::3], dim=0)
            future_left = torch.stack(tensors[n*3+1::3], dim=0)
            future_right = torch.stack(tensors[n*3+2::3], dim=0)

            return {
                'base_top': base_top,
                'base_left': base_left,
                'base_right': base_right,
                'future_top': future_top,
                'future_left': future_left,
                'future_right': future_right
            }, future_frame_indices, current_batch_size

        # Prefetch the first batch
        prefetch_future = None
        if prefetch and len(batch_starts) > 1:
            prefetch_future = self._executor.submit(prepare_batch_data, batch_starts[0])

        for batch_idx, i in enumerate(tqdm(batch_starts, desc="Evaluating")):
            # Get current batch data
            if prefetch and prefetch_future is not None:
                batch_tensors, future_frame_indices, current_batch_size = prefetch_future.result()
            else:
                batch_tensors, future_frame_indices, current_batch_size = prepare_batch_data(i)

            # Prefetch next batch while GPU is busy
            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare_batch_data, batch_starts[batch_idx + 1])

            # Move to device
            base_top_batch = batch_tensors['base_top'].to(self.device)
            base_left_batch = batch_tensors['base_left'].to(self.device)
            base_right_batch = batch_tensors['base_right'].to(self.device)
            future_top_batch = batch_tensors['future_top'].to(self.device)
            future_left_batch = batch_tensors['future_left'].to(self.device)
            future_right_batch = batch_tensors['future_right'].to(self.device)

            # Expand initial frame tensors to batch size
            initial_top_batch = initial_top_tensor.expand(current_batch_size, -1, -1, -1)
            initial_left_batch = initial_left_tensor.expand(current_batch_size, -1, -1, -1)
            initial_right_batch = initial_right_tensor.expand(current_batch_size, -1, -1, -1)

            # Build relative observation: compare frame_n (his_-100) vs frame_{n+interval} (base_0)
            relative_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_-100_rgb": base_top_batch,
                    "left_wrist_-100_rgb": base_left_batch,
                    "right_wrist_-100_rgb": base_right_batch,
                    "base_0_rgb": future_top_batch,
                    "left_wrist_0_rgb": future_left_batch,
                    "right_wrist_0_rgb": future_right_batch,
                },
                "image_masks": {}
            }

            # Build absolute observation: compare frame_0 (his_-100) vs frame_n (base_0)
            absolute_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_-100_rgb": initial_top_batch,
                    "left_wrist_-100_rgb": initial_left_batch,
                    "right_wrist_-100_rgb": initial_right_batch,
                    "base_0_rgb": base_top_batch,
                    "left_wrist_0_rgb": base_left_batch,
                    "right_wrist_0_rgb": base_right_batch,
                },
                "image_masks": {}
            }

            # Expand tokenized prompt to batch
            tokens_batch = np.tile(tokens[np.newaxis, :], (current_batch_size, 1))
            token_masks_batch = np.tile(token_masks[np.newaxis, :], (current_batch_size, 1))

            relative_observation = {
                **relative_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }
            absolute_observation = {
                **absolute_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }

            relative_observation = SimpleNamespace(**relative_observation)
            absolute_observation = SimpleNamespace(**absolute_observation)

            # Batched inference
            with torch.no_grad():
                relative_val_arr = self.model.sample_values(self.device, relative_observation)  # (batch, 1)
                absolute_val_arr = self.model.sample_values(self.device, absolute_observation)   # (batch, 1)

            # Collect per-frame results
            for j in range(current_batch_size):
                frame_idx = i + j

                # Normalize relative advantage when interval differs from expected
                if future_frame_indices[j] - frame_idx == relative_interval:
                    relative_val = float(relative_val_arr[j, 0].item())
                elif future_frame_indices[j] == frame_idx:
                    relative_val = float(0)
                else:
                    relative_val = float(relative_val_arr[j, 0].item()) / (future_frame_indices[j] - frame_idx) * relative_interval

                # First frame has zero absolute value by definition
                if frame_idx == 0:
                    absolute_val = float(0)
                else:
                    absolute_val = float(absolute_val_arr[j, 0].item())

                result = {
                    "frame_idx": frame_idx,
                    "future_frame_idx": future_frame_indices[j],
                    "relative_advantage": relative_val,
                    "absolute_value": absolute_val
                }
                all_results.append(result)

        # Compute absolute_advantage from absolute_value differences
        all_results_dict = {result["frame_idx"]: result for result in all_results}
        for result in all_results:
            frame_idx = result["frame_idx"]
            future_frame_idx = result["future_frame_idx"]
            future_result = all_results_dict.get(future_frame_idx)
            if future_frame_idx == frame_idx:
                result["absolute_advantage"] = 0.0
            elif future_frame_idx - frame_idx != relative_interval:
                result["absolute_advantage"] = (future_result["absolute_value"] - result["absolute_value"]) / (future_frame_idx - frame_idx) * relative_interval
            else:
                result["absolute_advantage"] = future_result["absolute_value"] - result["absolute_value"]

            result["absolute_advantage"] = max(-1.0, min(1.0, result["absolute_advantage"]))
            result["relative_advantage"] = max(-1.0, min(1.0, result["relative_advantage"]))

        logging.info(f"Evaluation complete, processed {len(all_results)} frames")
        return all_results

    def evaluate_video_1timestep_advantage(
        self,
        video_paths: Tuple[str, str, str],
        prompt: str,
        batch_size: int = 8,
        frame_interval: int = 1,
        relative_interval: int = 50,
        min_frame_index: int = None,
        max_frame_index: int = None,
        prefetch: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate advantage using single-timestep mode (absolute value only).

        For each frame n, computes:
          - absolute_value: model(frame_n)
          - absolute_advantage: absolute_value[n+interval] - absolute_value[n]

        Args:
            video_paths: Tuple of (top, left, right) video file paths.
            prompt: Task prompt string.
            batch_size: Batch size for GPU inference.
            frame_interval: Sampling interval (1 = every frame).
            relative_interval: Frames ahead for advantage computation (default: 50).
            min_frame_index: Start frame index (inclusive).
            max_frame_index: End frame index (inclusive).
            prefetch: Whether to prefetch next batch during GPU inference.

        Returns:
            List of dicts, each containing: frame_idx, future_frame_idx,
            absolute_value, absolute_advantage.
        """
        if len(video_paths) != 3:
            raise ValueError("Expected 3 video paths: (top, left, right)")

        # Load video frames in parallel
        logging.info(f"Loading video frames in parallel (interval: {frame_interval}, workers: {self.num_workers})...")
        top_frames, left_frames, right_frames = self._load_videos_parallel(video_paths, frame_interval)

        # Validate consistent frame counts
        if len(left_frames) != len(top_frames) or len(right_frames) != len(top_frames):
            raise ValueError(
                f"Inconsistent frame counts: top={len(top_frames)}, "
                f"left={len(left_frames)}, right={len(right_frames)}"
            )

        top_frames = top_frames[min_frame_index:max_frame_index+1]
        left_frames = left_frames[min_frame_index:max_frame_index+1]
        right_frames = right_frames[min_frame_index:max_frame_index+1]
        num_frames = len(top_frames)
        if num_frames < 2:
            raise ValueError(f"Insufficient frames: {num_frames}, need at least 2")
        logging.info(f"Total frames after slicing: {num_frames}, relative_interval: {relative_interval}")

        # Tokenize prompt
        tokens, token_masks = self.tokenizer.tokenize(prompt, state=None)

        all_results = []

        logging.info(f"Starting batched evaluation (1-timestep mode, prefetch: {prefetch})...")
        if num_frames <= 0:
            raise ValueError(f"Insufficient frames ({num_frames}), need at least {relative_interval + 1}")

        max_frame_idx = num_frames - 1
        batch_starts = list(range(0, num_frames, batch_size))

        def prepare_batch_data_1timestep(batch_start: int) -> Tuple[Dict, List[int], int]:
            """Prepare tensors for a single batch (current frames only)."""
            end_idx = min(batch_start + batch_size, num_frames)
            current_batch_size = end_idx - batch_start
            batch_indices = list(range(batch_start, end_idx))
            future_frame_indices = [min(j + relative_interval, max_frame_idx) for j in batch_indices]

            all_images = []
            for j in batch_indices:
                all_images.extend([top_frames[j], left_frames[j], right_frames[j]])

            tensor_futures = [self._executor.submit(self._process_single_image, img) for img in all_images]
            tensors = [f.result() for f in tensor_futures]

            n = current_batch_size
            base_top = torch.stack(tensors[0:n*3:3], dim=0)
            base_left = torch.stack(tensors[1:n*3:3], dim=0)
            base_right = torch.stack(tensors[2:n*3:3], dim=0)

            return {
                'base_top': base_top,
                'base_left': base_left,
                'base_right': base_right,
            }, future_frame_indices, current_batch_size

        # Prefetch the first batch
        prefetch_future = None
        if prefetch and len(batch_starts) > 1:
            prefetch_future = self._executor.submit(prepare_batch_data_1timestep, batch_starts[0])

        for batch_idx, i in enumerate(tqdm(batch_starts, desc="Evaluating")):
            if prefetch and prefetch_future is not None:
                batch_tensors, future_frame_indices, current_batch_size = prefetch_future.result()
            else:
                batch_tensors, future_frame_indices, current_batch_size = prepare_batch_data_1timestep(i)

            # Prefetch next batch
            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare_batch_data_1timestep, batch_starts[batch_idx + 1])

            # Move to device
            base_top_batch = batch_tensors['base_top'].to(self.device)
            base_left_batch = batch_tensors['base_left'].to(self.device)
            base_right_batch = batch_tensors['base_right'].to(self.device)

            absolute_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_0_rgb": base_top_batch,
                    "left_wrist_0_rgb": base_left_batch,
                    "right_wrist_0_rgb": base_right_batch,
                },
                "image_masks": {}
            }

            # Expand tokenized prompt to batch
            tokens_batch = np.tile(tokens[np.newaxis, :], (current_batch_size, 1))
            token_masks_batch = np.tile(token_masks[np.newaxis, :], (current_batch_size, 1))

            absolute_observation = {
                **absolute_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }

            absolute_observation = SimpleNamespace(**absolute_observation)

            # Batched inference
            with torch.no_grad():
                absolute_val_arr = self.model.sample_values(self.device, absolute_observation)  # (batch, 1)

            for j in range(current_batch_size):
                frame_idx = i + j
                if frame_idx == 0:
                    absolute_val = float(0)
                else:
                    absolute_val = float(absolute_val_arr[j, 0].item())

                result = {
                    "frame_idx": frame_idx,
                    "future_frame_idx": future_frame_indices[j],
                    "absolute_value": absolute_val
                }
                all_results.append(result)

        # Compute absolute_advantage from absolute_value differences
        all_results_dict = {result["frame_idx"]: result for result in all_results}
        for result in all_results:
            frame_idx = result["frame_idx"]
            future_frame_idx = result["future_frame_idx"]
            future_result = all_results_dict.get(future_frame_idx)
            if future_frame_idx == frame_idx:
                result["absolute_advantage"] = 0.0
            elif future_frame_idx - frame_idx != relative_interval:
                result["absolute_advantage"] = (future_result["absolute_value"] - result["absolute_value"]) / (future_frame_idx - frame_idx) * relative_interval
            else:
                result["absolute_advantage"] = future_result["absolute_value"] - result["absolute_value"]

            result["absolute_advantage"] = max(-1.0, min(1.0, result["absolute_advantage"]))

        logging.info(f"Evaluation complete, processed {len(all_results)} frames")
        return all_results


def main():
    """Example usage for quick testing."""
    config_name = "VALUE_TORCH_Pi05_KAI_cloth_11_15"
    ckpt_dir = "/path/to/checkpoint/100000"

    video_root = "/path/to/test_videos"
    top_video = os.path.join(video_root, "top_head.mp4")
    left_video = os.path.join(video_root, "hand_left.mp4")
    right_video = os.path.join(video_root, "hand_right.mp4")

    evaluator = SimpleValueEvaluator(
        config_name=config_name,
        ckpt_dir=ckpt_dir,
        num_workers=48,
    )

    results = evaluator.evaluate_video_2timesteps_advantages(
        video_paths=(top_video, left_video, right_video),
        prompt="Flatten and fold the cloth.",
        batch_size=8,
        frame_interval=1,
        prefetch=True,
    )

    print(f"\n=== Evaluation complete ===")
    print(f"Total results: {len(results)}")
    for res in results:
        print(
            f"frame {res['frame_idx']}, future {res['future_frame_idx']}: "
            f"relative_adv={res['relative_advantage']:.4f}, "
            f"absolute_adv={res['absolute_advantage']:.4f}, "
            f"absolute_val={res['absolute_value']:.4f}"
        )

    evaluator.shutdown()


if __name__ == "__main__":
    main()

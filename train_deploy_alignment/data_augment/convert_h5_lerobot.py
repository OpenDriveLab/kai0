import os
os.environ["SVT_LOG"] = "1"

from typing import List, Dict, Tuple
from pathlib import Path
import shutil
import logging
import time
from functools import partial
import subprocess

import numpy as np
import tyro
import av
import h5py
import json

from mini_lerobot.builder import LeRobotDatasetBuilder
from interface import lazy_load_hdf5_dataset_noimg as lazy_load_hdf5_dataset


def load_features_from_json(json_path: Path | str) -> dict:
    """Load features from a JSON file and convert to the format expected by LeRobotDatasetBuilder.
    
    Args:
        json_path: Path to the features.json file
        
    Returns:
        Dictionary of features in the format expected by LeRobotDatasetBuilder
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Features JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        features = json.load(f)
    
    # Filter out metadata fields that are automatically added by LeRobot
    metadata_fields = {"episode_index", "frame_index", "index", "task_index"}
    filtered_features = {
        k: v for k, v in features.items() 
        if k not in metadata_fields
    }
    
    # Convert shape from list to tuple for consistency with Python format
    for key, feature in filtered_features.items():
        if "shape" in feature and isinstance(feature["shape"], list):
            feature["shape"] = tuple(feature["shape"])
        # Convert None to empty list for names if needed
        if "names" in feature and feature["names"] is None:
            feature.pop("names")
    
    return filtered_features


def encode_video_frames(
        images: np.ndarray, 
        dst: Path,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
        log_level: int | None = av.logging.ERROR,
        overwrite: bool = False,
) -> bytes:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(dst)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        print(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Define video output frame size (assuming all input frames are the same size)

    dummy_image = images[0]
    height, width, _ = dummy_image.shape

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_image in images:
            # input_image = Image.open(input_data).convert("RGB")
            # input_frame = av.VideoFrame.from_image(input_image)
            input_frame = av.VideoFrame.from_ndarray(input_image, format="rgb24", channel_last=True)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def produce_episode(video_map: dict[str, Path], log_dir: Path, prompt: str):
    episode_start_time = time.time()

    # Camera name mapping: HDF5 key -> actual video subdir name
    camera_mapping = {
        "top_head": "cam_high",
        "hand_left": "cam_left_wrist",
        "hand_right": "cam_right_wrist"
    }

    try:
        episode, f = lazy_load_hdf5_dataset(log_dir)

        epi_len = episode.pop("epi_len")
        tasks = [prompt] * epi_len

        feature_data = {
            "observation.state": episode["observation.state"],
            "action": episode["action"]
        }

        # Use existing video files when present; skip re-encoding
        camera_list = [key for key in episode.keys() if key.startswith("observation.images.")]

        for camera_key in camera_list:
            video_dst = video_map[camera_key]
            hdf5_camera_name = camera_key.replace("observation.images.", "")
            video_dir_name = camera_mapping.get(hdf5_camera_name, hdf5_camera_name)

            # Path to existing video: data_dir/video/video_dir_name/episode_X.mp4
            data_dir = log_dir.parent
            episode_name = log_dir.stem

            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"

            if existing_video_path.exists():
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_video_path, video_dst)
            else:
                print(f"  Warning: {video_dir_name}: existing video not found, encoding from frames.")
                images = np.array(episode[camera_key])
                encode_video_frames(images, dst=video_dst, fps=30, overwrite=True)

        f.close()
        return feature_data, tasks

    except Exception as e:
        error_time = time.time() - episode_start_time
        print(f"  Error processing episode {log_dir.name}: {e}. Ignoring this episode.")
        print(f"  Elapsed before error: {error_time:.2f}s")

def main(
    data_dir: Path | str,
    save_dir: Path | str,
    repo_ids: List[str] | str,
    prompt: str | None = None,
    save_repoid: str | None = None,
    max_workers: int = 8,
    *,
    overwrite: bool = False,
    only_sync: bool = False,
    features_json: Path | str | None = None,
):
    
    data_dir = Path(data_dir)
    
    # Load features from JSON file
    if features_json is None:
        # Default to features.json in the same directory as this script
        features_json = Path(__file__).parent / "features.json"
    FEATURES = load_features_from_json(features_json)
    if type(repo_ids) is str:
        repo_ids = [repo_ids]
    
    task = data_dir.name.split('_')[0]
    if save_repoid is None:
        repoid = data_dir.name.split('_')
        # task = repoid[0]
        save_repoid = '_'.join(repoid[1: -1]) + '_lerobot'
        print(f"save_repoid will be set according to repo_ids: {save_repoid}")

    log_files: List[Path] = []
    for repo_id in repo_ids:
        repo_path = data_dir / repo_id
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path {repo_path} does not exist.")
        found_files = sorted(d for d in repo_path.iterdir() if not d.is_dir() and d.suffix == '.hdf5')
        log_files.extend(found_files)
    # filter invalid hdf5 files
    valid_files = []
    for file in log_files:
        data_dir = file.parent
        episode, _ = lazy_load_hdf5_dataset(file)
        epi_len = episode["epi_len"]
        episode_name = file.stem
        all_videos_exist = True
        for video_dir_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            existing_video_path = data_dir / "video" / video_dir_name / f"{episode_name}.mp4"
            if not existing_video_path.exists():
                print(f"  Warning: {existing_video_path} not found, skipping this file.")
                all_videos_exist = False
                break
            try:
                with av.open(existing_video_path, 'r') as container:
                    stream = container.streams.video[0]
                    assert stream.frames == epi_len, f"Video {existing_video_path} has {stream.frames} frames, expected {epi_len}"
            except Exception as e:
                print(f"  Invalid video file {existing_video_path}, error: {e}. Ignoring this file.")
                all_videos_exist = False
                break

        if not all_videos_exist:
            continue
        try:
            with h5py.File(file, 'r') as f:
                pass
            valid_files.append(file)
        except Exception as e:
            print(f"  Invalid file {file}, error: {e}. Ignoring this file.")

    output_path = Path(save_dir) / task / save_repoid

    if not only_sync:
        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Output path {output_path} already exists. Use --overwrite to overwrite.")
        
        builder = LeRobotDatasetBuilder(
            repo_id=save_repoid,
            fps=30,
            features=FEATURES,
            robot_type='arx' if task == 'hang' else 'agilex',
            root=output_path,
        )

        if prompt is None:
            prompt = f"{task} the cloth"
        builder.add_episodes(
            partial(produce_episode, prompt=prompt),
            valid_files,
            max_workers=max_workers,
        )
        
        builder.flush()


def add_episode(

):
    pass

if __name__ == "__main__":
    st = time.time()
    tyro.cli(main)
    print(f"Time taken: {time.time() - st} seconds")
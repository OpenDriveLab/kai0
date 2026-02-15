#!/usr/bin/env python3
"""
extract_lerobot.py

Extract (downsample) frames from a LeRobot dataset by keeping every Nth frame.
This accelerates the actions in the dataset - for example, with extraction_factor=2,
a 60-frame episode becomes a 30-frame episode.

Usage:
    python extract_lerobot.py --src_path /path/to/source --tgt_path /path/to/target \\
                              --repo_id extracted_dataset --extraction_factor 2
"""
from pathlib import Path
import shutil
import json
import argparse
from typing import Dict, Any, List
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np

# --- lerobot imports (must be available in env) ---
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import (
        load_info,
        load_episodes,
        load_episodes_stats,
        load_tasks,
    )
except Exception as e:
    raise RuntimeError("lerobot package import failed. Activate environment where lerobot is installed.") from e

# Try to import video processing libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available. Video extraction will be skipped.")

# Try to import merge function
try:
    from merge_lerobot import merge_repos
    MERGE_AVAILABLE = True
except ImportError:
    MERGE_AVAILABLE = False
    print("Warning: merge_lerobot module not available. Merge functionality will be disabled.")


def find_parquet_by_episode(src_root: Path, ep_idx: int) -> Path | None:
    """Try to locate parquet for given episode index under src_root."""
    basename = f"episode_{ep_idx:06d}.parquet"
    candidates = list(src_root.rglob(basename))
    return candidates[0] if candidates else None


def find_video_by_episode_and_key(src_root: Path, ep_idx: int, vid_key: str) -> Path | None:
    """Try to locate mp4 for given episode and video key under src_root."""
    basename = f"episode_{ep_idx:06d}.mp4"
    candidates = list(src_root.rglob(basename))
    for c in candidates:
        if vid_key in "/".join(c.parts):
            return c
    return candidates[0] if candidates else None


def extract_frames_from_video(src_video_path: Path, tgt_video_path: Path, extraction_factor: int, target_fps: float):
    """
    Extract every Nth frame from source video and write to target video.
    
    Args:
        src_video_path: Source video file path
        tgt_video_path: Target video file path
        extraction_factor: Keep every Nth frame
        target_fps: FPS for the output video
    """
    if not HAS_CV2:
        raise RuntimeError("cv2 is required for video extraction but not available")
    
    cap = cv2.VideoCapture(str(src_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    tgt_video_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(tgt_video_path), fourcc, target_fps, (width, height))
    
    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep every Nth frame (0, N, 2N, 3N, ...)
        if frame_idx % extraction_factor == 0:
            out.write(frame)
            extracted_count += 1
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return extracted_count


def extract_dataset(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    extraction_factor: int = 2,
    force: bool = False,
    merge_src_paths: List[str] = None,
    merge_tgt_path: str = None,
    merge_repo_id: str = None,
    merge_force: bool = False,
):
    """
    Extract frames from a LeRobot dataset by keeping every Nth frame.
    
    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (extracted) dataset
        repo_id: Repository ID for the new dataset
        extraction_factor: Keep every Nth frame (e.g., 2 means keep frames 0, 2, 4, ...)
        force: Force extraction even if target exists
        merge_src_paths: Optional list of additional source dataset paths to merge with extracted dataset
        merge_tgt_path: Optional path for merged dataset (if None and merge_src_paths provided, uses tgt_path)
        merge_repo_id: Optional repo_id for merged dataset (if None and merge_src_paths provided, uses repo_id)
        merge_force: Force merge even if conflicts exist
    """
    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    
    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")
    
    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")
    
    # Load source dataset info
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}")
    
    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}
    
    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    
    print(f"[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {src_info.get('total_episodes', 0)}")
    print(f"    - Total frames: {src_info.get('total_frames', 0)}")
    print(f"    - Extraction factor: {extraction_factor}")
    
    # Create target dataset structure
    print(f"[*] Creating target dataset at {tgt_root}")
    ds_target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=str(tgt_root),
        robot_type=robot_type,
        features=features
    )
    meta_target = ds_target.meta
    
    # Add tasks from source
    for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
        meta_target.add_task(task)
    print(f"[*] Added {len(src_tasks_dict)} tasks to target dataset")
    
    # Determine video keys
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    print(f"[*] Video keys to process: {video_keys}")
    
    warnings: List[str] = []
    
    # Process each episode
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
    for src_ep_idx_str, ep in tqdm(items_sorted, desc="Extracting episodes"):
        src_ep_idx = int(src_ep_idx_str)
        src_ep_length = int(ep.get("length", 0))
        ep_tasks = ep.get("tasks", [])
        
        # Calculate new episode length after extraction
        new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
        if new_ep_length == 0:
            warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
            continue
        
        # Find source parquet file
        src_chunksize = int(src_info.get("chunks_size", 1000))
        src_chunk_idx = src_ep_idx // src_chunksize
        
        try:
            src_parquet_rel = src_info["data_path"].format(
                episode_chunk=src_chunk_idx,
                episode_index=src_ep_idx
            )
            src_parquet_path = (src_root / src_parquet_rel).resolve()
            if not src_parquet_path.is_file():
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    raise FileNotFoundError
        except Exception:
            alt = find_parquet_by_episode(src_root, src_ep_idx)
            if alt:
                src_parquet_path = alt
            else:
                warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                continue
        
        # New episode index in target
        new_ep_idx = int(meta_target.info["total_episodes"])
        
        # Target paths
        tgt_chunk_idx = meta_target.get_episode_chunk(new_ep_idx)
        tgt_parquet_rel = meta_target.data_path.format(
            episode_chunk=tgt_chunk_idx,
            episode_index=new_ep_idx
        )
        tgt_parquet_path = meta_target.root / tgt_parquet_rel
        
        # --- EXTRACT & PATCH PARQUET: keep every Nth frame ---
        start_global_index = int(meta_target.info.get("total_frames", 0))
        tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.read_parquet(str(src_parquet_path))
            total_frames = len(df)
            
            # Extract every Nth frame (indices 0, N, 2N, 3N, ...)
            extracted_indices = list(range(0, total_frames, extraction_factor))
            df_extracted = df.iloc[extracted_indices].copy()
            
            # Reset pandas index immediately to avoid alignment issues
            df_extracted.reset_index(drop=True, inplace=True)
            
            n_rows = len(df_extracted)
            
            # Helper function to handle episode_index column format
            def make_episode_index_col(series, scalar_val, n):
                if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                    return pd.Series([[int(scalar_val)]] * n, index=range(n))
                else:
                    return pd.Series([int(scalar_val)] * n, index=range(n))
            
            # Update episode_index
            df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)
            
            # Update frame_index to be sequential within the episode [0, 1, 2, 3, ...]
            if "frame_index" in df_extracted.columns:
                new_frame_indices = list(range(n_rows))
                if df_extracted["frame_index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["frame_index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                    else:
                        df_extracted["frame_index"] = new_frame_indices
                else:
                    df_extracted["frame_index"] = new_frame_indices
            
            # Update timestamp to be evenly spaced according to FPS
            if "timestamp" in df_extracted.columns:
                frame_duration = 1.0 / fps
                new_timestamps = [i * frame_duration for i in range(n_rows)]
                if df_extracted["timestamp"].dtype == object and n_rows > 0:
                    first_val = df_extracted["timestamp"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                    else:
                        df_extracted["timestamp"] = new_timestamps
                else:
                    df_extracted["timestamp"] = new_timestamps
            
            # Update global index
            new_global_indices = list(range(start_global_index, start_global_index + n_rows))
            if "index" in df_extracted.columns:
                if df_extracted["index"].dtype == object and n_rows > 0:
                    first_val = df_extracted["index"].iloc[0]
                    if isinstance(first_val, (list, tuple, np.ndarray)):
                        df_extracted["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices
            else:
                df_extracted["index"] = new_global_indices
            
            # Write extracted parquet
            df_extracted.to_parquet(str(tgt_parquet_path), index=False)
            
        except Exception as e:
            warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
            continue
        
        # --- EXTRACT VIDEOS ---
        for vid_key in video_keys:
            try:
                src_vpath_rel = src_info["video_path"].format(
                    episode_chunk=src_chunk_idx,
                    video_key=vid_key,
                    episode_index=src_ep_idx
                )
                src_vpath = (src_root / src_vpath_rel).resolve()
                if not src_vpath.is_file():
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                        continue
            except Exception:
                alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                if alt_vid:
                    src_vpath = alt_vid
                else:
                    warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found, skipping")
                    continue
            
            tgt_vchunk_idx = tgt_chunk_idx
            tgt_vpath_rel = meta_target.video_path.format(
                episode_chunk=tgt_vchunk_idx,
                video_key=vid_key,
                episode_index=new_ep_idx
            )
            tgt_vpath = meta_target.root / tgt_vpath_rel
            
            try:
                if HAS_CV2:
                    extracted_count = extract_frames_from_video(
                        src_vpath, tgt_vpath, extraction_factor, float(fps)
                    )
                else:
                    warnings.append(f"cv2 not available, skipping video extraction for {vid_key}")
            except Exception as e:
                warnings.append(f"Failed to extract video {vid_key} for episode {src_ep_idx}: {e}")
        
        # Get episode stats (may use string keys)
        ep_stats = {}
        if isinstance(src_episodes_stats, dict):
            ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
        
        # Register episode in target meta
        meta_target.save_episode(
            episode_index=new_ep_idx,
            episode_length=new_ep_length,
            episode_tasks=ep_tasks,
            episode_stats=ep_stats
        )
    
    # Summary
    print("\n[+] Extraction finished!")
    print(f"    Target total_episodes: {meta_target.info.get('total_episodes')}")
    print(f"    Target total_frames  : {meta_target.info.get('total_frames')}")
    print(f"    Reduction factor     : ~{extraction_factor}x")
    
    if warnings:
        print(f"\n[!] {len(warnings)} warnings:")
        for w in warnings[:10]:  # Show first 10 warnings
            print("  -", w)
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    # Smoke test
    try:
        ds_check = LeRobotDataset(repo_id=repo_id, root=str(meta_target.root))
        print("\n[*] Smoke test succeeded!")
        print(f"    Loaded dataset: {ds_check.num_episodes} episodes, {ds_check.num_frames} frames")
    except Exception as e:
        print(f"\n[!] Warning: failed to load target dataset: {e}")
    
    # Merge with additional datasets if requested
    if merge_src_paths and len(merge_src_paths) > 0:
        if not MERGE_AVAILABLE:
            print("\n[!] Warning: merge_lerobot module not available. Skipping merge step.")
            return
        
        print("\n" + "=" * 60)
        print("Merging extracted dataset with additional sources")
        print("=" * 60)
        
        # Prepare merge source paths (include extracted dataset)
        all_merge_srcs = [tgt_path] + merge_src_paths
        merge_final_path = merge_tgt_path if merge_tgt_path else tgt_path + "_merged"
        merge_final_repo_id = merge_repo_id if merge_repo_id else repo_id + "_merged"
        
        print(f"Merge sources: {all_merge_srcs}")
        print(f"Merge target: {merge_final_path}")
        print(f"Merge repo_id: {merge_final_repo_id}")
        print()
        
        try:
            # Infer features/fps/robot_type from extracted dataset
            merge_repos(
                src_paths=all_merge_srcs,
                tgt_path=merge_final_path,
                repo_id=merge_final_repo_id,
                fps=fps,
                robot_type=robot_type,
                features=features,
                force=merge_force
            )
            print("\n[+] Merge complete!")
        except Exception as e:
            print(f"\n[!] Error during merge: {e}")
            raise
    
    return


def time_scaling_with_split(
    src_path: str,
    tgt_path: str,
    repo_id: str,
    split_ratio: float = 0.3,
    extraction_factor: int = 2,
    force: bool = False,
):
    """
    Split dataset by ratio, extract frames from one part, and merge both parts.
    
    Args:
        src_path: Path to source LeRobot dataset
        tgt_path: Path to target (final merged) dataset
        repo_id: Repository ID for the new dataset (will append _time_scaling suffix)
        split_ratio: Ratio of data to extract (e.g., 0.3 means 30% will be extracted, 70% kept original)
        extraction_factor: Extract every Nth frame for the extracted portion
        force: Force operation even if target exists
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
    
    if extraction_factor < 1:
        raise ValueError(f"extraction_factor must be >= 1, got {extraction_factor}")
    
    src_root = Path(src_path).expanduser().resolve()
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_path}")
    
    tgt_root = Path(tgt_path).expanduser().resolve()
    if tgt_root.exists() and any(tgt_root.iterdir()) and not force:
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Use --force or remove it first.")
    
    # Load source dataset
    print(f"[*] Loading source dataset from {src_root}")
    try:
        src_info = load_info(src_root)
        src_episodes = load_episodes(src_root)
        src_tasks_dict, _ = load_tasks(src_root)
    except Exception as e:
        raise RuntimeError(f"Cannot load source dataset: {e}")
    
    try:
        src_episodes_stats = load_episodes_stats(src_root)
    except Exception:
        src_episodes_stats = {}
    
    # Extract parameters from source
    fps = src_info.get("fps", 30)
    robot_type = src_info.get("robot_type", "unknown")
    features = src_info.get("features", {})
    
    total_episodes = len(src_episodes)
    extract_count = max(1, int(total_episodes * split_ratio))
    keep_count = total_episodes - extract_count
    
    print(f"[*] Source dataset info:")
    print(f"    - FPS: {fps}")
    print(f"    - Robot type: {robot_type}")
    print(f"    - Total episodes: {total_episodes}")
    print(f"    - Split ratio: {split_ratio}")
    print(f"    - Episodes to extract: {extract_count}")
    print(f"    - Episodes to keep original: {keep_count}")
    print(f"    - Extraction factor: {extraction_factor}")
    
    # Sort episodes by index
    items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
    
    # Split episodes
    extract_episodes = dict(items_sorted[:extract_count])
    keep_episodes = dict(items_sorted[extract_count:])
    
    print(f"\n[*] Split episodes:")
    print(f"    - Extracting: episodes {list(extract_episodes.keys())[0]} to {list(extract_episodes.keys())[-1]}")
    print(f"    - Keeping original: episodes {list(keep_episodes.keys())[0]} to {list(keep_episodes.keys())[-1]}")
    
    # Create temporary directories
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="time_scaling_"))
    extract_tgt_path = str(temp_dir / "extracted")
    keep_tgt_path = str(temp_dir / "kept")
    
    try:
        # Step 1: Extract frames from first portion
        print(f"\n[1/3] Extracting frames from {extract_count} episodes...")
        # Create target dataset structure for extracted portion
        ds_extract = LeRobotDataset.create(
            repo_id=repo_id + "_extracted",
            fps=int(fps),
            root=extract_tgt_path,
            robot_type=robot_type,
            features=features
        )
        meta_extract = ds_extract.meta
        
        # Add tasks
        for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_extract.add_task(task)
        
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        warnings: List[str] = []
        
        # Process extract episodes
        for src_ep_idx_str, ep in tqdm(extract_episodes.items(), desc="Extracting episodes"):
            src_ep_idx = int(src_ep_idx_str)
            src_ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])
            
            # Calculate new episode length after extraction
            new_ep_length = (src_ep_length + extraction_factor - 1) // extraction_factor
            if new_ep_length == 0:
                warnings.append(f"Episode {src_ep_idx} too short ({src_ep_length} frames), skipping")
                continue
            
            # Find source parquet file
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize
            
            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    warnings.append(f"Parquet for episode {src_ep_idx} not found, skipping")
                    continue
            
            # New episode index in target
            new_ep_idx = int(meta_extract.info["total_episodes"])
            
            # Target paths
            tgt_chunk_idx = meta_extract.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_extract.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_extract.root / tgt_parquet_rel
            
            # Extract & patch parquet
            start_global_index = int(meta_extract.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                df = pd.read_parquet(str(src_parquet_path))
                total_frames = len(df)
                
                # Extract every Nth frame
                extracted_indices = list(range(0, total_frames, extraction_factor))
                df_extracted = df.iloc[extracted_indices].copy()
                df_extracted.reset_index(drop=True, inplace=True)
                
                n_rows = len(df_extracted)
                
                def make_episode_index_col(series, scalar_val, n):
                    if len(series) > 0 and series.dtype == object and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return pd.Series([[int(scalar_val)]] * n, index=range(n))
                    else:
                        return pd.Series([int(scalar_val)] * n, index=range(n))
                
                df_extracted["episode_index"] = make_episode_index_col(df_extracted["episode_index"], new_ep_idx, n_rows)
                
                # Update frame_index
                if "frame_index" in df_extracted.columns:
                    new_frame_indices = list(range(n_rows))
                    if df_extracted["frame_index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["frame_index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["frame_index"] = [[int(x)] for x in new_frame_indices]
                        else:
                            df_extracted["frame_index"] = new_frame_indices
                    else:
                        df_extracted["frame_index"] = new_frame_indices
                
                # Update timestamp
                if "timestamp" in df_extracted.columns:
                    frame_duration = 1.0 / fps
                    new_timestamps = [i * frame_duration for i in range(n_rows)]
                    if df_extracted["timestamp"].dtype == object and n_rows > 0:
                        first_val = df_extracted["timestamp"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["timestamp"] = [[float(x)] for x in new_timestamps]
                        else:
                            df_extracted["timestamp"] = new_timestamps
                    else:
                        df_extracted["timestamp"] = new_timestamps
                
                # Update global index
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df_extracted.columns:
                    if df_extracted["index"].dtype == object and n_rows > 0:
                        first_val = df_extracted["index"].iloc[0]
                        if isinstance(first_val, (list, tuple, np.ndarray)):
                            df_extracted["index"] = [[int(x)] for x in new_global_indices]
                        else:
                            df_extracted["index"] = new_global_indices
                    else:
                        df_extracted["index"] = new_global_indices
                else:
                    df_extracted["index"] = new_global_indices
                
                df_extracted.to_parquet(str(tgt_parquet_path), index=False)
            except Exception as e:
                warnings.append(f"Failed to extract parquet for episode {src_ep_idx}: {e}")
                continue
            
            # Extract videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue
                
                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_extract.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_extract.root / tgt_vpath_rel
                
                try:
                    if HAS_CV2:
                        extract_frames_from_video(src_vpath, tgt_vpath, extraction_factor, float(fps))
                except Exception as e:
                    warnings.append(f"Failed to extract video {vid_key} for episode {src_ep_idx}: {e}")
            
            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
            
            meta_extract.save_episode(
                episode_index=new_ep_idx,
                episode_length=new_ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )
        
        if warnings:
            print(f"  [!] {len(warnings)} warnings during extraction")
        
        # Step 2: Copy original episodes for second portion
        print(f"\n[2/3] Copying original {keep_count} episodes...")
        # Create a new dataset with only the kept episodes
        ds_keep = LeRobotDataset.create(
            repo_id=repo_id + "_kept",
            fps=int(fps),
            root=keep_tgt_path,
            robot_type=robot_type,
            features=features
        )
        meta_keep = ds_keep.meta
        
        # Add tasks
        for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
            meta_keep.add_task(task)
        
        video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
        
        for src_ep_idx_str, ep in tqdm(keep_episodes.items(), desc="Copying episodes"):
            src_ep_idx = int(src_ep_idx_str)
            ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])
            
            # Find source parquet
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize
            
            try:
                src_parquet_rel = src_info["data_path"].format(
                    episode_chunk=src_chunk_idx,
                    episode_index=src_ep_idx
                )
                src_parquet_path = (src_root / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    alt = find_parquet_by_episode(src_root, src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(src_root, src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    continue
            
            # New episode index
            new_ep_idx = int(meta_keep.info["total_episodes"])
            tgt_chunk_idx = meta_keep.get_episode_chunk(new_ep_idx)
            tgt_parquet_rel = meta_keep.data_path.format(
                episode_chunk=tgt_chunk_idx,
                episode_index=new_ep_idx
            )
            tgt_parquet_path = meta_keep.root / tgt_parquet_rel
            
            start_global_index = int(meta_keep.info.get("total_frames", 0))
            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy and patch parquet
            try:
                df = pd.read_parquet(str(src_parquet_path))
                n_rows = len(df)
                
                def make_episode_index_col(series, scalar_val):
                    if series.dtype == object and n_rows > 0 and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return series.apply(lambda _: [int(scalar_val)])
                    else:
                        return pd.Series([int(scalar_val)] * n_rows, index=series.index)
                
                if "episode_index" in df.columns:
                    df["episode_index"] = make_episode_index_col(df["episode_index"], new_ep_idx)
                else:
                    df["episode_index"] = [int(new_ep_idx)] * n_rows
                
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df.columns:
                    if df["index"].dtype == object and n_rows > 0 and isinstance(df["index"].iloc[0], (list, tuple, np.ndarray)):
                        df["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df["index"] = new_global_indices
                else:
                    df["index"] = new_global_indices
                
                df.to_parquet(str(tgt_parquet_path), index=False)
            except Exception as e:
                shutil.copy2(str(src_parquet_path), str(tgt_parquet_path))
            
            # Copy videos
            for vid_key in video_keys:
                try:
                    src_vpath_rel = src_info["video_path"].format(
                        episode_chunk=src_chunk_idx,
                        video_key=vid_key,
                        episode_index=src_ep_idx
                    )
                    src_vpath = (src_root / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(src_root, src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        continue
                
                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_keep.video_path.format(
                    episode_chunk=tgt_vchunk_idx,
                    video_key=vid_key,
                    episode_index=new_ep_idx
                )
                tgt_vpath = meta_keep.root / tgt_vpath_rel
                tgt_vpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_vpath), str(tgt_vpath))
            
            # Get episode stats
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
            
            meta_keep.save_episode(
                episode_index=new_ep_idx,
                episode_length=ep_length,
                episode_tasks=ep_tasks,
                episode_stats=ep_stats
            )
        
        # Step 3: Merge extracted and kept portions
        print(f"\n[3/3] Merging extracted and kept portions...")
        if not MERGE_AVAILABLE:
            raise RuntimeError("merge_lerobot module not available. Cannot perform merge operation.")
        
        final_repo_id = repo_id + "_time_scaling"
        merge_repos(
            src_paths=[extract_tgt_path, keep_tgt_path],
            tgt_path=tgt_path,
            repo_id=final_repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            force=force
        )
        
        print(f"\n[+] Time scaling complete!")
        print(f"    Final dataset: {tgt_path}")
        print(f"    Final repo_id: {final_repo_id}")
        
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract (downsample) frames from a LeRobot dataset"
    )
    parser.add_argument(
        "--src_path",
        required=True,
        help="Path to source LeRobot dataset"
    )
    parser.add_argument(
        "--tgt_path",
        required=True,
        help="Path to target (extracted) dataset"
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Repository ID for the new dataset"
    )
    parser.add_argument(
        "--extraction_factor",
        type=int,
        default=2,
        help="Extract every Nth frame (default: 2, meaning keep frames 0, 2, 4, ...)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force extraction even if target directory exists"
    )
    parser.add_argument(
        "--merge_src_paths",
        nargs="+",
        default=None,
        help="Optional: Additional source dataset paths to merge with extracted dataset"
    )
    parser.add_argument(
        "--merge_tgt_path",
        type=str,
        default=None,
        help="Optional: Path for merged dataset (default: <tgt_path>_merged)"
    )
    parser.add_argument(
        "--merge_repo_id",
        type=str,
        default=None,
        help="Optional: Repository ID for merged dataset (default: <repo_id>_merged)"
    )
    parser.add_argument(
        "--merge_force",
        action="store_true",
        help="Force merge even if conflicts exist"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=None,
        help="Enable split mode: ratio of data to extract (0.0-1.0). If set, splits dataset, extracts one portion, and merges both."
    )
    
    args = parser.parse_args()
    
    # Check if split mode is enabled
    if args.split_ratio is not None:
        print("=" * 60)
        print("LeRobot Dataset Time Scaling (Split & Extract)")
        print("=" * 60)
        
        time_scaling_with_split(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            split_ratio=args.split_ratio,
            extraction_factor=args.extraction_factor,
            force=args.force
        )
    else:
        print("=" * 60)
        print("LeRobot Dataset Frame Extraction")
        print("=" * 60)
        
        extract_dataset(
            src_path=args.src_path,
            tgt_path=args.tgt_path,
            repo_id=args.repo_id,
            extraction_factor=args.extraction_factor,
            force=args.force,
            merge_src_paths=args.merge_src_paths,
            merge_tgt_path=args.merge_tgt_path,
            merge_repo_id=args.merge_repo_id,
            merge_force=args.merge_force
        )


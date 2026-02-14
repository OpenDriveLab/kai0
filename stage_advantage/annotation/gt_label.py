#!/usr/bin/env python3
"""
# python label.py <dataset_path> --threshold 30 --chunk-size 50 --discretion-type binary --advantage-source absolute_advantage --stage-nums 2 --dry-run
Script to modify task_index in parquet files based on progress rewards.

This script:
1. Reads all parquet files from path/data/chunk-*/*.parquet
2. Calculates reward as: progress[i+50] - progress[i] for each frame
3. Computes reward distribution statistics across all parquets
4. Labels frames with task_index based on reward percentile threshold
   Binary mode:
   - task_index=0 for rewards in bottom (1-threshold)% 
   - task_index=1 for rewards in top threshold%
   n_slices mode:
   - task_index=0 to (n-1) based on reward percentiles (higher reward -> higher task_index)
   - Each slice contains ~(100/n)% of frames

Stage-based mode (--stage-nums > 1):
   - Each frame is assigned to a stage based on its stage_progress_gt value
   - Frames with stage_progress_gt in [i/stage_nums, (i+1)/stage_nums) belong to stage i
   - Each stage has its own reward statistics and percentile boundaries
   - task_index is assigned based on stage-specific percentiles
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def calculate_rewards(data: pd.DataFrame, chunk_size: int = 50, advantage_source: str = "progress") -> np.ndarray:
    """
    Calculate rewards based on progress differences.
    
    Args:
        data: DataFrame containing 'progress' column
        chunk_size: Number of frames to look ahead for progress calculation
        
    Returns:
        Array of rewards for each frame
    """
    n_frames = len(data)
    rewards = np.zeros(n_frames, dtype=np.float32)
    if advantage_source == "absolute_advantage":
        absolute_advantage = data['absolute_advantage'].values
        for i in range(n_frames):
            rewards[i] = absolute_advantage[i]
    elif advantage_source == "relative_advantage":
        relative_advantage = data['relative_advantage'].values
        for i in range(n_frames):
            rewards[i] = relative_advantage[i]
    elif advantage_source == "progress":
        progress = data['progress'].values
        for i in range(n_frames):
            if i + chunk_size < n_frames:
                rewards[i] = progress[i + chunk_size] - progress[i]
            else:
                # For frames near the end, use the last available frame
                rewards[i] = (progress[-1] - progress[i]) / (len(progress) - i) * chunk_size
    else:
        raise ValueError(f"Unknown advantage source: {advantage_source}")
    return rewards


def get_stage_index(stage_progress_gt: float, stage_nums: int) -> int:
    """
    Get the stage index based on stage_progress_gt value.
    
    Args:
        stage_progress_gt: The stage progress value (0-1) for a single frame
        stage_nums: Number of stages to divide into
        
    Returns:
        Stage index (0 to stage_nums-1)
    """
    if stage_nums == 1:
        return 0
    
    step = 1.0 / stage_nums
    stage_idx = int(stage_progress_gt / step)
    # Handle edge case where stage_progress_gt == 1.0
    if stage_idx >= stage_nums:
        stage_idx = stage_nums - 1
    return stage_idx


def collect_all_rewards(base_path: str, chunk_size: int = 50, advantage_source: str = "progress",
                        stage_nums: int = 1) -> Tuple[Dict[int, List[float]], List[str]]:
    """
    Collect all rewards from all parquet files to compute statistics.
    
    Args:
        base_path: Base directory path containing data/chunk-*/*.parquet files
        chunk_size: Number of frames to look ahead for progress calculation
        advantage_source: Source of advantage values
        stage_nums: Number of stages to divide data into based on stage_progress_gt
        
    Returns:
        Tuple of (rewards_by_stage, parquet_files)
        rewards_by_stage: Dict mapping stage_index to list of rewards
    """
    # Find all parquet files
    pattern = os.path.join(base_path, "data", "chunk-*", "*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found matching pattern: {pattern}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Initialize rewards by stage
    rewards_by_stage = {i: [] for i in range(stage_nums)}
    
    # Collect rewards from all files
    print("Collecting rewards from all files...")
    for parquet_file in tqdm(parquet_files):
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            # Calculate rewards for all frames
            rewards = calculate_rewards(df, chunk_size, advantage_source)
            
            if stage_nums == 1:
                # No stage division, all rewards go to stage 0
                rewards_by_stage[0].extend(rewards.tolist())
            else:
                # Divide rewards by stage based on each frame's stage_progress_gt
                if 'stage_progress_gt' not in df.columns:
                    raise ValueError(f"Column 'stage_progress_gt' not found in {parquet_file}. "
                                   f"Required when stage_nums > 1.")
                
                stage_progress_gt_values = df['stage_progress_gt'].values
                # Each frame has its own stage_progress_gt, assign it to the corresponding stage
                for frame_idx in range(len(rewards)):
                    spg = stage_progress_gt_values[frame_idx]
                    stage_idx = get_stage_index(spg, stage_nums)
                    rewards_by_stage[stage_idx].append(rewards[frame_idx])
            
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
    
    return rewards_by_stage, parquet_files


def compute_reward_statistics(rewards: List[float]) -> dict:
    """
    Compute reward distribution statistics.
    
    Args:
        rewards: List of all rewards
        
    Returns:
        Dictionary containing percentile information
    """
    if len(rewards) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'percentiles': {p: 0.0 for p in range(0, 101, 10)}
        }
    
    rewards_array = np.array(rewards)
    
    # Compute percentiles in 10% increments
    percentiles = list(range(0, 101, 10))
    percentile_values = np.percentile(rewards_array, percentiles)
    
    stats = {
        'mean': np.mean(rewards_array),
        'std': np.std(rewards_array),
        'min': np.min(rewards_array),
        'max': np.max(rewards_array),
        'percentiles': dict(zip(percentiles, percentile_values))
    }
    
    return stats


def update_tasks_jsonl(base_path: str, discretion_type: str, n_slices: int = 10) -> None:
    """
    Update the tasks.jsonl file based on discretization type.
    
    Args:
        base_path: Base directory path containing meta/tasks.jsonl
        discretion_type: Type of discretization ("binary" or "n_slices")
        n_slices: Number of slices for n_slices mode
    """
    tasks_file = os.path.join(base_path, "meta", "tasks.jsonl")
    
    # Ensure meta directory exists
    meta_dir = os.path.join(base_path, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    tasks = []
    if discretion_type == "binary":
        tasks = [
            {"task_index": 0, "task": "fold the cloth, Advantage: negative"},
            {"task_index": 1, "task": "fold the cloth, Advantage: positive"},
        ]
    elif discretion_type == "n_slices":
        for i in range(n_slices):
            tasks.append({"task_index": i, "task": f"fold the cloth, Advantage: {i}"})
    
    # Write tasks to jsonl file
    with open(tasks_file, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"\n✓ Updated {tasks_file} with {len(tasks)} task(s)")


def assign_task_index(parquet_file: str, threshold_percentile: float, 
                      chunk_size: int = 50, discretion_type: str = "binary",
                      percentile_boundaries: List[float] = None, n_slices: int = 10,
                      advantage_source: str = "progress") -> None:
    """
    Assign task_index to frames in a parquet file based on reward threshold.
    (Used when stage_nums=1)
    
    Args:
        parquet_file: Path to the parquet file
        threshold_percentile: Percentile value for threshold (used in binary mode)
        chunk_size: Number of frames to look ahead for progress calculation
        discretion_type: Type of discretization ("binary" or "n_slices")
        percentile_boundaries: List of percentile boundary values (used in n_slices mode)
        n_slices: Number of slices for n_slices mode
    """
    # Read parquet file
    df = pd.read_parquet(parquet_file)
    
    # Calculate rewards
    rewards = calculate_rewards(df, chunk_size, advantage_source)
    
    if discretion_type == "binary":
        # Binary mode: task_index = 0 for rewards below threshold, 1 for >= threshold
        task_index = (rewards >= threshold_percentile).astype(np.int32)
    elif discretion_type == "n_slices":
        # n-slices mode: task_index from 0 to (n_slices-1) based on percentile boundaries
        task_index = np.zeros(len(rewards), dtype=np.int32)
        for i in range(len(percentile_boundaries) - 1):
            mask = (rewards >= percentile_boundaries[i]) & (rewards < percentile_boundaries[i + 1])
            task_index[mask] = i
        # Handle the top slice
        task_index[rewards >= percentile_boundaries[-1]] = n_slices - 1
    else:
        raise ValueError(f"Unknown discretion_type: {discretion_type}")
    
    # Add or update task_index column
    df['task_index'] = task_index
    
    # Save back to parquet file
    df.to_parquet(parquet_file, index=False)


def assign_task_index_staged(parquet_file: str, 
                             threshold_percentiles_by_stage: Dict[int, float],
                             percentile_boundaries_by_stage: Dict[int, List[float]],
                             chunk_size: int = 50, 
                             discretion_type: str = "binary",
                             n_slices: int = 10,
                             advantage_source: str = "progress",
                             stage_nums: int = 1) -> None:
    """
    Assign task_index to frames in a parquet file based on stage-specific thresholds.
    Each frame's stage is determined by its own stage_progress_gt value.
    
    Args:
        parquet_file: Path to the parquet file
        threshold_percentiles_by_stage: Dict mapping stage_idx to threshold value (binary mode)
        percentile_boundaries_by_stage: Dict mapping stage_idx to percentile boundaries (n_slices mode)
        chunk_size: Number of frames to look ahead for progress calculation
        discretion_type: Type of discretization ("binary" or "n_slices")
        n_slices: Number of slices for n_slices mode
        advantage_source: Source of advantage values
        stage_nums: Number of stages
    """
    # Read parquet file
    df = pd.read_parquet(parquet_file)
    
    # Calculate rewards for all frames
    rewards = calculate_rewards(df, chunk_size, advantage_source)
    
    # Get stage_progress_gt values for each frame
    if 'stage_progress_gt' not in df.columns:
        raise ValueError(f"Column 'stage_progress_gt' not found in {parquet_file}")
    stage_progress_gt_values = df['stage_progress_gt'].values
    
    # Initialize task_index array
    task_index = np.zeros(len(rewards), dtype=np.int32)
    
    # Assign task_index based on each frame's stage and stage-specific thresholds
    for frame_idx in range(len(rewards)):
        reward = rewards[frame_idx]
        spg = stage_progress_gt_values[frame_idx]
        stage_idx = get_stage_index(spg, stage_nums)
        
        if discretion_type == "binary":
            threshold = threshold_percentiles_by_stage[stage_idx]
            task_index[frame_idx] = 1 if reward >= threshold else 0
        elif discretion_type == "n_slices":
            boundaries = percentile_boundaries_by_stage[stage_idx]
            # Find the slice this reward belongs to
            slice_idx = 0
            for j in range(len(boundaries) - 1):
                if reward >= boundaries[j] and reward < boundaries[j + 1]:
                    slice_idx = j
                    break
            # Handle the top slice
            if reward >= boundaries[-1]:
                slice_idx = n_slices - 1
            task_index[frame_idx] = slice_idx
    
    # Add or update task_index column
    df['task_index'] = task_index
    
    # Save back to parquet file
    df.to_parquet(parquet_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Modify task_index in parquet files based on progress rewards"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Base path containing data/chunk-*/*.parquet files"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Threshold percentile for task_index labeling (default: 70, meaning top 70%% get task_index=1)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of frames to look ahead for progress calculation (default: 50)"
    )
    parser.add_argument(
        "--discretion-type",
        type=str,
        default="binary",
        choices=["binary", "n_slices"],
        help="Discretization type: 'binary' splits into 0/1, 'n_slices' splits into 0 to (n-1) (default: binary)"
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=10,
        help="Number of slices for n_slices mode (default: 10)"
    )
    parser.add_argument(
        "--advantage-source",
        type=str,
        default="progress",
        choices=["progress", "absolute_advantage", "relative_advantage"]
    )
    parser.add_argument(
        "--stage-nums",
        type=int,
        default=1,
        help="Number of stages to divide data based on each frame's stage_progress_gt. "
             "1 means no division (original behavior). "
             "2 means divide by 0.5 (frames with stage_progress_gt < 0.5 and >= 0.5). "
             "3 means divide by 1/3 and 2/3, etc. "
             "Each stage calculates its own reward percentiles independently. (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute statistics without modifying files"
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Path does not exist: {args.data_path}")
    
    print(f"Processing data from: {args.data_path}")
    print(f"Discretization type: {args.discretion_type}")
    if args.discretion_type == "binary":
        print(f"Threshold: {args.threshold}% (top {args.threshold}% will be task_index=1)")
    elif args.discretion_type == "n_slices":
        print(f"Number of slices: {args.n_slices}")
    print(f"Progress offset: {args.chunk_size} frames")
    print(f"Stage nums: {args.stage_nums}")
    if args.stage_nums > 1:
        step = 1.0 / args.stage_nums
        boundaries = [step * i for i in range(args.stage_nums + 1)]
        print(f"Stage boundaries (based on frame's stage_progress_gt): {boundaries}")
    print("-" * 80)
    
    # Step 1: Collect all rewards (by stage if stage_nums > 1)
    rewards_by_stage, parquet_files = collect_all_rewards(
        args.data_path, args.chunk_size, args.advantage_source, args.stage_nums
    )
    
    total_frames = sum(len(rewards) for rewards in rewards_by_stage.values())
    print(f"\nTotal frames processed: {total_frames}")
    
    # Print frames per stage
    if args.stage_nums > 1:
        print("\nFrames per stage:")
        for stage_idx in range(args.stage_nums):
            stage_frames = len(rewards_by_stage[stage_idx])
            percentage = 100.0 * stage_frames / total_frames if total_frames > 0 else 0
            step = 1.0 / args.stage_nums
            lower = step * stage_idx
            upper = step * (stage_idx + 1)
            print(f"  Stage {stage_idx} (stage_progress_gt in [{lower:.3f}, {upper:.3f})): "
                  f"{stage_frames} frames ({percentage:.1f}%)")
    
    # Step 2: Compute statistics for each stage
    stats_by_stage = {}
    threshold_percentiles_by_stage = {}
    percentile_boundaries_by_stage = {}
    
    for stage_idx in range(args.stage_nums):
        print(f"\n{'=' * 80}")
        if args.stage_nums > 1:
            step = 1.0 / args.stage_nums
            lower = step * stage_idx
            upper = step * (stage_idx + 1)
            print(f"STAGE {stage_idx} REWARD STATISTICS (stage_progress_gt in [{lower:.3f}, {upper:.3f}))")
        else:
            print("REWARD STATISTICS")
        print("=" * 80)
        
        stage_rewards = rewards_by_stage[stage_idx]
        stats = compute_reward_statistics(stage_rewards)
        stats_by_stage[stage_idx] = stats
        
        print(f"Frames count: {len(stage_rewards)}")
        print(f"Mean:   {stats['mean']:.6f}")
        print(f"Std:    {stats['std']:.6f}")
        print(f"Min:    {stats['min']:.6f}")
        print(f"Max:    {stats['max']:.6f}")
        print("\nPercentiles:")
        for p, v in stats['percentiles'].items():
            print(f"  {p:3d}%: {v:.6f}")
        
        # Calculate threshold/boundaries for this stage
        if len(stage_rewards) > 0:
            if args.discretion_type == "binary":
                threshold_value = np.percentile(stage_rewards, (100 - args.threshold))
                threshold_percentiles_by_stage[stage_idx] = threshold_value
                print(f"\nThreshold value (top {args.threshold}%): {threshold_value:.6f}")
            elif args.discretion_type == "n_slices":
                n_slices = args.n_slices
                step_pct = 100 / n_slices
                percentile_points = [step_pct * i for i in range(n_slices)]
                boundaries = [np.percentile(stage_rewards, p) for p in percentile_points]
                percentile_boundaries_by_stage[stage_idx] = boundaries
                
                print(f"\n{n_slices}-Slices Boundaries (higher reward -> higher task_index):")
                for i in range(len(boundaries)):
                    if i < len(boundaries) - 1:
                        print(f"  task_index={i}: reward in [{boundaries[i]:.6f}, {boundaries[i+1]:.6f})")
                    else:
                        print(f"  task_index={i}: reward >= {boundaries[i]:.6f}")
        else:
            # Empty stage - use default values
            if args.discretion_type == "binary":
                threshold_percentiles_by_stage[stage_idx] = 0.0
            elif args.discretion_type == "n_slices":
                percentile_boundaries_by_stage[stage_idx] = [0.0] * args.n_slices
            print(f"\nWarning: Stage {stage_idx} has no data!")
    
    print("=" * 80)
    
    if args.dry_run:
        print("\nDry run mode - no files will be modified")
        return
    
    # Step 3: Update tasks.jsonl
    update_tasks_jsonl(args.data_path, args.discretion_type, args.n_slices)
    
    # Step 4: Assign task_index to all parquet files
    print(f"\nAssigning task_index to {len(parquet_files)} files...")
    for parquet_file in tqdm(parquet_files):
        try:
            if args.stage_nums == 1:
                # Use original function for backward compatibility
                assign_task_index(
                    parquet_file, 
                    threshold_percentiles_by_stage.get(0, 0.0), 
                    args.chunk_size,
                    args.discretion_type,
                    percentile_boundaries_by_stage.get(0, None),
                    args.n_slices,
                    args.advantage_source
                )
            else:
                # Use staged function - each frame's stage determined by its stage_progress_gt
                assign_task_index_staged(
                    parquet_file,
                    threshold_percentiles_by_stage,
                    percentile_boundaries_by_stage,
                    args.chunk_size,
                    args.discretion_type,
                    args.n_slices,
                    args.advantage_source,
                    args.stage_nums
                )
        except Exception as e:
            print(f"\nError processing {parquet_file}: {e}")
            continue
    
    print("\n✓ Task completed successfully!")


if __name__ == "__main__":
    main()

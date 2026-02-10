"""
Split a LeRobot-format dataset into multiple disjoint subsets by episode.

Each subset is written as a full LeRobot dataset under dst_path/split_0, split_1, ...
Episodes are shuffled (with a fixed seed) then divided so you can train separate
models on each subset and later mix them with model_arithmetic/arithmetic.py.
"""
from pathlib import Path
import json
import argparse
import shutil
import random
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm

# Video keys expected in LeRobot dataset layout
VIDEO_KEYS = [
    "observation.images.hand_left",
    "observation.images.hand_right",
    "observation.images.top_head",
]


def split_lerobot_data(
    source_path: Path,
    dst_path: Path,
    episode_index: list[int],
    num: int,
) -> None:
    """
    Write one subset of the source LeRobot dataset to dst_path.

    episode_index: list of source episode indices to include (will be renumbered 0..n-1).
    num: subset index (used only for progress messages).
    """
    # Remap episode indices to 0, 1, 2, ...
    old_episode_index = sorted(episode_index)
    new_episode_index = list(range(len(old_episode_index)))
    old2new_episode_index = dict(zip(old_episode_index, new_episode_index))
    new2old_episode_index = dict(zip(new_episode_index, old_episode_index))

    dst_path.mkdir(parents=True, exist_ok=True)

    # Load episode stats for the selected episode indices
    with open(source_path / "meta" / "episodes_stats.jsonl", "r") as f:
        episodes_stats = [json.loads(line) for line in f if line.strip()]
    episodes_stats = [
        ep for ep in episodes_stats if ep["episode_index"] in episode_index
    ]
    assert len(episodes_stats) == len(episode_index), (
        f"episode_index count mismatch: {len(episodes_stats)} vs {len(episode_index)}"
    )
    episodes_stats.sort(key=lambda x: x["episode_index"])

    with open(source_path / "meta" / "info.json", "r") as f:
        info = json.load(f)
    chunks_size = info["chunks_size"]

    # Build new episodes_stats with renumbered episode and frame indices
    new_episodes_stats = []
    new_frame_index = 0
    for episode_stat in tqdm(
        episodes_stats,
        desc=f"Split {num}: episodes_stats",
        total=len(episodes_stats),
    ):
        index_count = episode_stat["stats"]["index"]["count"][0]
        new_idx = old2new_episode_index[episode_stat["episode_index"]]
        episode_stat["episode_index"] = new_idx
        episode_stat["stats"]["index"]["min"] = [new_frame_index]
        episode_stat["stats"]["index"]["max"] = [new_frame_index + index_count - 1]
        episode_stat["stats"]["index"]["mean"] = [(new_frame_index + new_frame_index + index_count - 1) / 2]
        episode_stat["stats"]["index"]["std"] = [np.std(range(new_frame_index, new_frame_index + index_count))]
        episode_stat["stats"]["index"]["count"] = [index_count]
        new_frame_index += index_count
        new_episodes_stats.append(episode_stat)

    (dst_path / "meta").mkdir(parents=True, exist_ok=True)
    with open(dst_path / "meta" / "episodes_stats.jsonl", "w") as f:
        for episode_stat in new_episodes_stats:
            f.write(json.dumps(episode_stat) + "\n")

    # Write info.json with updated totals
    with open(dst_path / "meta" / "info.json", "w") as f:
        info["total_episodes"] = len(old_episode_index)
        info["total_frames"] = new_frame_index
        info["total_videos"] = len(old_episode_index) * 3
        info["total_chunks"] = len(old_episode_index) // chunks_size + 1
        info["splits"] = {"train": f"0:{len(old_episode_index)}"}
        json.dump(info, f, indent=4)
    # Copy and reindex parquet files
    for new_stat in new_episodes_stats:
        new_index = new_stat["episode_index"]
        old_index = new2old_episode_index[new_index]
        old_episode_path = source_path / "data" / f"chunk-{old_index // chunks_size:03d}" / f"episode_{old_index:06d}.parquet"
        new_episode_path = dst_path / "data" / f"chunk-{new_index // chunks_size:03d}" / f"episode_{new_index:06d}.parquet"
        if not new_episode_path.parent.exists():
            new_episode_path.parent.mkdir(parents=True, exist_ok=True)
        parquet = pd.read_parquet(old_episode_path)
        parquet["index"] = parquet["index"] - parquet["index"].min() + new_stat["stats"]["index"]["min"][0]
        parquet["episode_index"] = new_index
        parquet.to_parquet(new_episode_path, index=False)

    # Update record CSV files (e.g. v1/record.csv)
    record_files = [
        p for p in os.listdir(source_path / "meta")
        if "record.csv" in p and p.lower().startswith("v")
    ]
    for record_file in record_files:
        record_data = pd.read_csv(source_path / "meta" / record_file)
        record_data["episode_index"] = record_data["video"].str.split("_").str[-1].str.split(".").str[0].astype(int)
        record_data = record_data.loc[record_data["episode_index"].isin(episode_index)]
        record_data["episode_index"] = record_data["episode_index"].map(old2new_episode_index)
        record_data["video"] = record_data["video"].map(
            lambda x: f"episode_{old2new_episode_index[int(x.split('.')[0].split('_')[-1])]:06d}.mp4"
        )
        record_data.drop(columns=["episode_index"], inplace=True)
        record_data.to_csv(dst_path / "meta" / record_file, index=False)

    # Write episodes.jsonl with renumbered episode_index
    with open(source_path / "meta" / "episodes.jsonl", "r") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    episodes = [ep for ep in episodes if ep["episode_index"] in old_episode_index]
    assert len(episodes) == len(episode_index), "episode count mismatch in episodes.jsonl"
    episodes.sort(key=lambda x: x["episode_index"])
    for episode in episodes:
        episode["episode_index"] = old2new_episode_index[episode["episode_index"]]
    with open(dst_path / "meta" / "episodes.jsonl", "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")

    shutil.copy(source_path / "meta" / "tasks.jsonl", dst_path / "meta" / "tasks.jsonl")

    # Copy video files
    for new_stat in tqdm(
        new_episodes_stats,
        desc=f"Split {num}: videos",
        total=len(new_episodes_stats),
    ):
        new_index = new_stat["episode_index"]
        old_index = new2old_episode_index[new_index]
        for video_key in VIDEO_KEYS:
            old_episode_path = source_path / "videos" / f"chunk-{old_index // chunks_size:03d}" / video_key / f"episode_{old_index:06d}.mp4"
            new_episode_path = dst_path / "videos" / f"chunk-{new_index // chunks_size:03d}" / video_key / f"episode_{new_index:06d}.mp4"
            if not new_episode_path.parent.exists():
                new_episode_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(old_episode_path, new_episode_path)

    

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a LeRobot dataset into disjoint subsets by episode for training separate models."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the source LeRobot dataset (must contain meta/, data/, videos/).",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        required=True,
        help="Output directory; subsets will be written as dst_path/split_0, split_1, ...",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=4,
        help="Number of disjoint subsets to create (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling episodes before splitting (default: 42).",
    )
    args = parser.parse_args()
    source_path = Path(args.source_path)
    dst_path = Path(args.dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "meta" / "info.json", "r") as f:
        info = json.load(f)
    total_episodes = info["total_episodes"]
    episode_indices = list(range(total_episodes))
    random.seed(args.seed)
    random.shuffle(episode_indices)
    splits = np.array_split(episode_indices, args.split_num)
    max_workers = min(args.split_num, os.cpu_count() or 1)
    with mp.Pool(processes=max_workers) as pool:
        pool.starmap(
            split_lerobot_data,
            [(source_path, dst_path / f"split_{i}", splits[i].tolist(), i) for i in range(args.split_num)],
        )


if __name__ == "__main__":
    main()
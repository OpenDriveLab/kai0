#!/usr/bin/env python3
"""
merge_lerobot.py

Merge several LeRobot-format dataset repos into one, using LeRobotDataset.create()
and LeRobotDatasetMetadata.save_episode to ensure produced meta matches the library's expectations.

Features:
 - supports --src_paths DIR1 DIR2 ...
 - supports --src_list paths.txt  (one path per line; blank lines and lines starting with # are ignored)
 - both options may be used together; duplicates will be removed.
 - retains previous behavior: copying parquet & videos, merging tasks, using --features_json and --force.
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


def ensure_features_compatible(tgt_info: Dict[str, Any], src_info: Dict[str, Any], src_name: str, force: bool, warnings: List[str]):
    # Check fps and robot_type compatibility
    for k in ("fps", "robot_type"):
        if k in tgt_info and k in src_info and tgt_info[k] != src_info[k]:
            msg = f"Conflict {k}: target={tgt_info[k]} vs source({src_name})={src_info[k]}"
            if force:
                warnings.append("FORCED: " + msg)
            else:
                raise RuntimeError(msg)

    # Check features exact equality (or forced)
    if "features" in tgt_info and "features" in src_info and tgt_info["features"] != src_info["features"]:
        msg = f"Conflict features dict between target and source {src_name}"
        if force:
            warnings.append("FORCED: " + msg)
        else:
            raise RuntimeError(msg)


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


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def read_src_list_file(txt_path: Path) -> List[str]:
    """Read a newline-separated file containing dataset paths. Ignore empty lines and lines starting with #."""
    out: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def merge_repos(
    src_paths: List[str],
    tgt_path: str,
    repo_id: str,
    fps: int | None = None,
    robot_type: str | None = None,
    features: Dict[str, Any] | None = None,
    force: bool = False,
):
    # Normalize and deduplicate source roots
    src_roots = []
    seen = set()
    for p in src_paths:
        try:
            pp = Path(p).expanduser().resolve()
        except Exception:
            pp = Path(p).expanduser()
        if not pp.exists():
            print(f"Warning: source path does not exist, skipping: {p}")
            continue
        if str(pp) in seen:
            continue
        seen.add(str(pp))
        src_roots.append(pp)

    if not src_roots:
        raise RuntimeError("No valid source repos provided after parsing --src_paths / --src_list.")

    tgt_root = Path(tgt_path).expanduser().resolve()

    if tgt_root.exists() and any(tgt_root.iterdir()):
        raise RuntimeError(f"Target {tgt_root} exists and is not empty. Move/remove it first.")

    # If features/fps/robot_type not provided, try to infer from first source
    if features is None:
        try:
            first_info = load_info(src_roots[0])
            features = first_info.get("features", None)
            if fps is None:
                fps = first_info.get("fps", 30)
            if robot_type is None:
                robot_type = first_info.get("robot_type", "agilex")
        except Exception:
            # fallback defaults
            features = features or {}
            fps = fps or 30
            robot_type = robot_type or "agilex"

    # Create target dataset structure via API
    print("[*] Creating target LeRobot dataset structure...")
    ds_target = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=str(tgt_root), robot_type=robot_type, features=features)
    meta_target = ds_target.meta
    print(f"[*] Target created at {tgt_root}. Initial total_episodes={meta_target.info.get('total_episodes', 0)}")

    merge_warnings: List[str] = []

    # For each source repo
    for src in src_roots:
        print(f"[+] Merging source: {src}")
        # load source info and episodes via lerobot utils
        try:
            src_info = load_info(src)
        except Exception as e:
            raise RuntimeError(f"Cannot load info.json from {src}: {e}")

        # Check compatibility
        ensure_features_compatible(meta_target.info, src_info, src.name, force, merge_warnings)

        # --- MERGE TASKS from source into target meta ---
        try:
            src_tasks_dict, _ = load_tasks(src)  # load_tasks returns (tasks_dict, task_to_index)
            # src_tasks_dict is mapping idx -> task string (keys may be ints or strings)
            for k, task in sorted(src_tasks_dict.items(), key=lambda kv: int(kv[0])):
                # Only add if task not present in target
                if meta_target.get_task_index(task) is None:
                    # add_task will append to meta/tasks.jsonl and update meta.info
                    meta_target.add_task(task)
            print(f"   - merged tasks from {src} (total target tasks now: {meta_target.info.get('total_tasks')})")
        except Exception:
            # If load_tasks fails or no tasks, just continue; tasks may still be added when saving episodes
            pass

        # load episodes dict (keys may be strings)
        try:
            src_episodes = load_episodes(src)  # returns dict: key -> episode_dict
        except Exception as e:
            # fallback: no episodes present
            print(f"   - Warning: cannot load episodes from {src}: {e}. Skipping this source.")
            continue

        # load episodes_stats if present (may return dict keyed by str)
        try:
            src_episodes_stats = load_episodes_stats(src)
        except Exception:
            src_episodes_stats = {}

        # Determine video keys present in source features (dtype == "video")
        src_video_keys = [k for k, v in src_info.get("features", {}).items() if v.get("dtype") == "video"]
        # Intersect with target video keys to avoid copying unexpected video types
        video_keys = [k for k in src_video_keys if k in meta_target.video_keys]

        # iterate episodes in numeric order. src_episodes keys may be strings, use int(key) to sort
        items_sorted = sorted(src_episodes.items(), key=lambda kv: int(kv[0]))
        for src_ep_idx_str, ep in tqdm(items_sorted, desc=f"episodes in {src.name}", leave=False):
            src_ep_idx = int(src_ep_idx_str)  # guaranteed int for formatting and calculations
            ep_length = int(ep.get("length", 0))
            ep_tasks = ep.get("tasks", [])

            # try to construct source parquet path using src_info["data_path"]
            src_chunksize = int(src_info.get("chunks_size", 1000))
            src_chunk_idx = src_ep_idx // src_chunksize
            # Pass episode_chunk as integer (so template with :03d works)
            try:
                src_parquet_rel = src_info["data_path"].format(episode_chunk=src_chunk_idx, episode_index=src_ep_idx)
                src_parquet_path = (Path(src) / src_parquet_rel).resolve()
                if not src_parquet_path.is_file():
                    # fallback search by name
                    alt = find_parquet_by_episode(Path(src), src_ep_idx)
                    if alt:
                        src_parquet_path = alt
                    else:
                        raise FileNotFoundError
            except Exception:
                alt = find_parquet_by_episode(Path(src), src_ep_idx)
                if alt:
                    src_parquet_path = alt
                else:
                    raise FileNotFoundError(f"Parquet for episode {src_ep_idx} not found under {src}")

            # Determine new episode index in target (append at end)
            new_ep_idx = int(meta_target.info["total_episodes"])

            # target chunk idx & target parquet relative path
            tgt_chunk_idx = meta_target.get_episode_chunk(new_ep_idx)
            # Pass episode_chunk as integer (so template with :03d works)
            tgt_parquet_rel = meta_target.data_path.format(episode_chunk=tgt_chunk_idx, episode_index=new_ep_idx)
            tgt_parquet_path = meta_target.root / tgt_parquet_rel

            # --- COPY & PATCH PARQUET: update episode_index and index columns before saving ---
            # compute starting global index for this episode (current total_frames)
            start_global_index = int(meta_target.info.get("total_frames", 0))

            tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = pd.read_parquet(str(src_parquet_path))
                n_rows = len(df)

                # helper: if column stores per-cell list/array (e.g. [123]) keep that format
                def make_episode_index_col(series, scalar_val):
                    if series.dtype == object and n_rows > 0 and isinstance(series.iloc[0], (list, tuple, np.ndarray)):
                        return series.apply(lambda _: [int(scalar_val)])
                    else:
                        return pd.Series([int(scalar_val)] * n_rows, index=series.index)

                # episode_index: replace with new_ep_idx (scalar per-row or list-wrapped)
                if "episode_index" in df.columns:
                    df["episode_index"] = make_episode_index_col(df["episode_index"], new_ep_idx)
                else:
                    df["episode_index"] = [int(new_ep_idx)] * n_rows

                # index (global frame index): produce range(start_global_index, ...)
                new_global_indices = list(range(start_global_index, start_global_index + n_rows))
                if "index" in df.columns:
                    if df["index"].dtype == object and n_rows > 0 and isinstance(df["index"].iloc[0], (list, tuple, np.ndarray)):
                        df["index"] = [[int(x)] for x in new_global_indices]
                    else:
                        df["index"] = new_global_indices
                else:
                    df["index"] = new_global_indices

                # write patched parquet
                df.to_parquet(str(tgt_parquet_path), index=False)
                print(f"   - copied+patched parquet {src_parquet_path.name} -> {tgt_parquet_rel} (ep_idx -> {new_ep_idx}, start_global_index -> {start_global_index})")

            except Exception as e:
                # fallback: if pandas read/write fails, do raw copy but warn
                merge_warnings.append(f"Failed to patch parquet for episode {src_ep_idx} from {src}: {e}. Falling back to raw copy.")
                shutil.copy2(str(src_parquet_path), str(tgt_parquet_path))
                print(f"   - warning: fallback copied parquet {src_parquet_path.name} -> {tgt_parquet_rel}")


            # copy videos for video_keys
            for vid_key in video_keys:
                # construct source video path using src_info["video_path"] template
                try:
                    src_vpath_rel = src_info["video_path"].format(episode_chunk=src_chunk_idx, video_key=vid_key, episode_index=src_ep_idx)
                    src_vpath = (Path(src) / src_vpath_rel).resolve()
                    if not src_vpath.is_file():
                        alt_vid = find_video_by_episode_and_key(Path(src), src_ep_idx, vid_key)
                        if alt_vid:
                            src_vpath = alt_vid
                        else:
                            merge_warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found in {src}; skipping that video.")
                            continue
                except Exception:
                    alt_vid = find_video_by_episode_and_key(Path(src), src_ep_idx, vid_key)
                    if alt_vid:
                        src_vpath = alt_vid
                    else:
                        merge_warnings.append(f"Video {vid_key} for episode {src_ep_idx} not found in {src}; skipping that video.")
                        continue

                tgt_vchunk_idx = tgt_chunk_idx
                tgt_vpath_rel = meta_target.video_path.format(episode_chunk=tgt_vchunk_idx, video_key=vid_key, episode_index=new_ep_idx)
                tgt_vpath = meta_target.root / tgt_vpath_rel
                tgt_vpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_vpath), str(tgt_vpath))
                print(f"     copied video {vid_key}: {src_vpath.name} -> {tgt_vpath_rel}")

            # episode_stats retrieval (source stats may use string keys)
            ep_stats = {}
            if isinstance(src_episodes_stats, dict):
                ep_stats = src_episodes_stats.get(str(src_ep_idx), src_episodes_stats.get(src_ep_idx, {}))
            else:
                ep_stats = {}

            # Register episode in target meta (this updates target info/episodes/stats)
            # save_episode will also add tasks that appear in ep_tasks if they aren't present yet
            meta_target.save_episode(episode_index=new_ep_idx, episode_length=ep_length, episode_tasks=ep_tasks, episode_stats=ep_stats)
            print(f"   - registered episode {new_ep_idx} (len={ep_length}) in target meta")


    # Summary
    print("[+] Merge finished.")
    print(f"    Target total_episodes: {meta_target.info.get('total_episodes')}")
    print(f"    Target total_frames  : {meta_target.info.get('total_frames')}")
    if merge_warnings:
        print("Merge warnings:")
        for w in merge_warnings:
            print("  -", w)

    # Smoke test: try to instantiate LeRobotDataset on target
    try:
        ds_check = LeRobotDataset(repo_id=repo_id, root=meta_target.root)
        print("[*] Smoke test load succeeded: LeRobotDataset loaded.")
        print("    num_episodes:", ds_check.num_episodes, "num_frames:", ds_check.num_frames)
    except Exception as e:
        print("Warning: failed to load target with LeRobotDataset:", e)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_paths", nargs="*", help="paths to source lerobot repos (space separated)")
    parser.add_argument("--src_list", type=str, help="path to newline-separated text file listing source repo paths (comments with # allowed)")
    parser.add_argument("--tgt_path", required=True, help="target merged repo local path")
    parser.add_argument("--repo_id", required=True, help="target repo_id to create")
    parser.add_argument("--fps", type=int, default=None, help="fps for target (if unspecified, infer from first source)")
    parser.add_argument("--robot_type", type=str, default=None, help="robot_type for target (infer from first source if omitted)")
    parser.add_argument("--features_json", type=str, default=None, help="optional: path to json file with features dict to force")
    parser.add_argument("--force", action="store_true", help="force merge on minor conflicts")
    args = parser.parse_args()

    # example usage: python merge_lerobot.py --src_list ./merge_list.txt --tgt_path /cpfs01/shared/data/flat_cloth/flat_cloth_v8merge --fps 30 --robot_type agilex --repo_id test --features_json ./features.json --force

    # Build combined src_paths (from CLI list and/or file) and deduplicate
    combined_srcs: List[str] = []
    if args.src_paths:
        combined_srcs.extend(args.src_paths)
    if args.src_list:
        p = Path(args.src_list)
        if not p.exists():
            print(f"ERROR: --src_list file not found: {p}")
            sys.exit(1)
        lines = read_src_list_file(p)
        combined_srcs.extend(lines)

    if not combined_srcs:
        print("ERROR: no source paths provided (use --src_paths or --src_list).")
        sys.exit(1)

    feat = None
    if args.features_json:
        feat = json.load(open(args.features_json, "r", encoding="utf-8"))

    merge_repos(combined_srcs, args.tgt_path, args.repo_id, fps=args.fps, robot_type=args.robot_type, features=feat, force=args.force)

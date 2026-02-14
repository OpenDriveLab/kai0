from functools import partial
import os
from pathlib import Path
from typing import Callable, Hashable, Iterable, Protocol, TypeVar
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

from mini_lerobot.constant import HF_LEROBOT_HOME
from mini_lerobot.metadata import LeRobotDatasetMetadata, LeRobotDatasetInfo, DEFAULT_FEATURES, _EpisodeSummary, LeRobotDatasetFeatureStat
from mini_lerobot.video import validate_and_compute_stats, transcode_video_simple

T = TypeVar("T", bound=Hashable)

class EpisodeProducer(Protocol):
    def __call__(self, video_map: dict[str, Path], *args) -> tuple[dict[str, np.ndarray], list[str]]:
        """
        The producer shall produce data for a whole episode based on *args, doing 3 things:
        1. Put videos to paths in video_map
        2. Return other features as numpy arrays
        3. Return a list of tasks
        """

def unique(seq: Iterable[T]) -> tuple[list[T], list[int]]:
    seen = {}  # For orderedness
    indices = []
    for x in seq:
        index = seen.setdefault(x, len(seen))
        indices.append(index)
    return list(seen.keys()), indices

def process_episode(producer: EpisodeProducer, info: LeRobotDatasetInfo, root: Path, episode_index: int, *args) -> _EpisodeSummary:
    episode_chunk = episode_index // info.chunks_size
    video_map = {
        key: root / info.video_path.format(episode_chunk=episode_chunk, episode_index=episode_index, video_key=key)
        for key, feature in info.features.items()
        if feature.dtype == "video"
    }
    custom_features = { key: feature for key, feature in info.features.items() if feature.dtype != "video" and key not in DEFAULT_FEATURES }
    feature_data, tasks = producer(video_map, *args)
    # Validate & Compute stats
    ep_stats = {}
    episode_length = len(tasks)
    assert custom_features.keys() == feature_data.keys()
    for key, feature in custom_features.items():
        d = feature_data[key]
        assert feature.shape == d.shape[1:], f"Feature {key} has shape {d.shape}, expected {feature.shape}"
        assert d.shape[0] == episode_length, f"Feature {key} has length {d.shape[0]}, expected {episode_length}"
        assert d.dtype == np.dtype(feature.dtype), f"Feature {key} has dtype {d.dtype}, expected {feature.dtype}"
    for key, video_path in video_map.items():
        video_feature = info.features[key]
        height, width, _ = video_feature.shape
        ep_stats[key] = validate_and_compute_stats(video_path, key, info.fps, episode_length, height, width)
    # Fill in default features
    frame_indices = np.arange(episode_length, dtype=np.int64)
    feature_data["timestamp"] = frame_indices / info.fps
    feature_data["frame_index"] = frame_indices
    feature_data["episode_index"] = np.full((episode_length,), episode_index, dtype=np.int64)
    feature_data["index"] = frame_indices
    tasks, task_indices = unique(tasks)
    feature_data["task_index"] = np.array(task_indices, dtype=np.int64)
    for key, d in feature_data.items():
        ep_stats[key] = get_feature_stats(d, axis=0, keepdims=d.ndim==1)
    ep_stats = {k: ep_stats[k] for k in info.features.keys()}  # reorder
    feature_data["timestamp"] = feature_data["timestamp"].astype(np.float32)  # Postpone type cast to make stat prettier
    # NOTE: task_index and index need fix

    data_path = root / info.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
    fields = []
    field_names = []
    for key, feature in info.features.items():
        if feature.dtype == "video":
            continue
        array = feature_data[key]
        if array.ndim == 1:
            pa_array = pa.array(array)
        elif array.ndim == 2:
            _, width = array.shape
            pa_array = pa.FixedSizeListArray.from_arrays(array.ravel(), width)
        else:
            pa_array = pa.FixedShapeTensorArray.from_numpy_ndarray(array)
        fields.append(pa_array)
        field_names.append(key)
    table = pa.Table.from_arrays(fields, field_names)
    pq.write_table(table, data_path)

    return _EpisodeSummary(episode_index=episode_index, episode_chunk=episode_chunk, tasks=tasks, length=episode_length, stats=ep_stats)

def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool):
    return LeRobotDatasetFeatureStat(
        min=np.min(array, axis=axis, keepdims=keepdims),
        max=np.max(array, axis=axis, keepdims=keepdims),
        mean=np.mean(array, axis=axis, keepdims=keepdims, dtype=np.float64),
        std=np.std(array, axis=axis, keepdims=keepdims, dtype=np.float64),
        count=len(array),
    )

def _init_worker():
    os.environ["FFMPEG_SINGLE_THREAD"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

class LeRobotDatasetBuilder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: Path | None = None,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        if root.exists():
            raise FileExistsError(f"Dataset already exists at {root}")
        root.mkdir(parents=True, exist_ok=False)
        self.metadata = LeRobotDatasetMetadata.create(repo_id, fps, features, robot_type, root)
        (root / "data").mkdir(parents=False, exist_ok=False)
        if self.metadata.info.video_path:
            (root / "videos").mkdir(parents=False, exist_ok=False)

    def _prepare_new_chunk(self):
        if self.metadata.total_episodes % self.metadata.chunks_size != 0:
            return
        chunk = self.metadata.total_episodes // self.metadata.chunks_size
        assert chunk == self.metadata.total_chunks
        (self.root / "data" / f"chunk-{chunk:03d}").mkdir(parents=False, exist_ok=False)
        if self.metadata.video_path:
            video_chunk = self.root / "videos" / f"chunk-{chunk:03d}"
            video_chunk.mkdir(parents=False, exist_ok=False)
            for key in self.metadata.video_keys:
                (video_chunk / key).mkdir(parents=False, exist_ok=False)

    def add_episode(self, producer: EpisodeProducer, *args):
        self._prepare_new_chunk()
        summary = process_episode(producer, self.metadata.info, self.root, self.metadata.total_episodes, *args)
        self._process_summary(summary)

    def add_episodes(self, producer: EpisodeProducer, *iterables, max_workers: int = 0):
        packed_args = tuple(zip(*iterables, strict=True))
        if len(packed_args) == 0:
            return
        if max_workers <= 0:
            for args in tqdm(packed_args):
                self.add_episode(producer, *args)
        else:
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
                process_episode_fn = partial(process_episode, producer, self.metadata.info, self.root)
                while len(packed_args) > 0:
                    packed_args = self._add_episodes_by_chunk(executor, process_episode_fn, packed_args)

    def _add_episodes_by_chunk(self, executor: ProcessPoolExecutor, process_episode_fn: Callable, packed_args: tuple):
        self._prepare_new_chunk()
        chunk_size = self.metadata.chunks_size - self.metadata.total_episodes % self.metadata.chunks_size
        args_chunk, remaining_args = packed_args[:chunk_size], packed_args[chunk_size:]
        jobs = [executor.submit(process_episode_fn, i + self.metadata.total_episodes, *args) for i, args in enumerate(args_chunk)]
        results = {}
        pbar = tqdm(as_completed(jobs), total=len(jobs), dynamic_ncols=True, disable=None, desc=f"Chunk {self.metadata.total_chunks}", postfix={"pending": 0})
        for job in pbar:
            summary: _EpisodeSummary = job.result()
            results[summary.episode_index] = summary
            while summary := results.pop(self.metadata.total_episodes, None):
                self._process_summary(summary)
            pbar.set_postfix({"pending": len(results)})
        return remaining_args

    def _process_summary(self, summary: _EpisodeSummary):
        assert summary.episode_index == self.metadata.total_episodes
        # Fix up globally-dependent data
        frame_offset = self.metadata.total_frames
        global_task_indices = np.array([self.metadata.upsert_task(task) for task in summary.tasks], dtype=np.int64)
        data_path = self.root / self.metadata.data_path.format(episode_chunk=summary.episode_chunk, episode_index=summary.episode_index)
        stats_fix = self._fix_data(data_path, frame_offset, global_task_indices)
        summary.stats.update(stats_fix)
        self.metadata.add_episode(summary)

    def _fix_data(self, data_path: Path, frame_offset: int, global_task_indices: np.ndarray):
        # Ugly workaround
        table = pq.read_table(data_path)
        def edit_column(table: pa.Table, column_name: str, fn: Callable[[np.ndarray], np.ndarray]):
            column = table[column_name]
            column_data = column.to_numpy()
            new_column_data = fn(column_data)
            new_column = pa.array(new_column_data, type=column.type)
            return table.set_column(table.schema.get_field_index(column_name), column_name, new_column)
        table = edit_column(table, 'index', lambda x: x + frame_offset)
        table = edit_column(table, 'task_index', lambda x: global_task_indices[x])
        pq.write_table(table, data_path)
        return {
            "index": get_feature_stats(table["index"].to_numpy(), 0, True),
            "task_index": get_feature_stats(table["task_index"].to_numpy(), 0, True),
        }

    def flush(self):
        self.metadata.dump()

from dataclasses import dataclass
import operator
from pathlib import Path
from typing import Sequence, SupportsIndex

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

from mini_lerobot.constant import HF_LEROBOT_HOME
from mini_lerobot.metadata import DEFAULT_FEATURES, LeRobotDatasetFeature, LeRobotDatasetMetadata
from mini_lerobot.video import InMemoryVideo

@dataclass
class LeRobotDatasetConfig:
    features: dict[str, LeRobotDatasetFeature]
    delta_indices: dict[str, np.ndarray]
    fps: int
    dataset_index: int | None = None

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.features.items() if v.dtype == "video")

    @property
    def table_keys(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.features.items() if v.dtype != "video")

def read_table(data_path: Path, columns: tuple[str, ...]):
    table = pq.read_table(str(data_path), columns=list(columns))
    def _read_column(column_name: str) -> np.ndarray:
        column = table[column_name]
        assert isinstance(column, pa.ChunkedArray)
        column = column.combine_chunks()
        if isinstance(column, pa.FixedSizeListArray):
            # Special treatment
            flat_column = column.flatten()
            flat_values = flat_column.to_numpy()
            return flat_values.reshape(len(column), column.type.list_size)
        elif isinstance(column, pa.ListArray):
            # Slow generic path for >= 2D
            type = column.type
            while isinstance(type, pa.ListType):
                type = type.value_type
            return np.array(column.to_pylist(), dtype=type.to_pandas_dtype())
        else:
            return column.to_numpy()

    return {k: _read_column(k) for k in columns}

class LeRobotDatasetEpisode:
    def __init__(self, data_path: Path, video_paths: dict[str, Path], config: LeRobotDatasetConfig):
        self._data_path = data_path
        self._video_paths = video_paths
        self._config = config
        self._table = read_table(data_path, config.table_keys)
        sizes = [len(v) for v in self._table.values()]
        assert all(s == sizes[0] for s in sizes), f"Inconsistent column sizes: {sizes}"
        self._size = sizes[0]
        self._videos = {k: InMemoryVideo(p, cfr=config.fps) for k, p in video_paths.items() if k in config.video_keys}

    def read(self, indices: int | np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for key, feature in self._config.features.items():
            this_indices, mask = self._format_indices(indices, key)
            if feature.dtype == "video":
                result[key] = self._videos[key].read(this_indices)
            else:
                result[key] = self._table[key][this_indices]
            if mask is not None:
                result[f"{key}_is_pad"] = mask
        if self._config.dataset_index is not None:
            if isinstance(indices, int):
                result["dataset_index"] = self._config.dataset_index
            else:
                result["dataset_index"] = np.full_like(indices, self._config.dataset_index, dtype=np.int64)
        return result

    def _format_indices(self, indices: int | np.ndarray, key: str):
        delta_indices = self._config.delta_indices.get(key)
        if delta_indices is not None:
            if isinstance(indices, int):
                indices_wd = delta_indices + indices
            else:
                indices_wd = indices[:, None] + delta_indices[None, :]
            indices_wd_clip = np.clip(indices_wd, 0, self._size - 1)
            mask = indices_wd != indices_wd_clip
            return indices_wd_clip, mask
        else:
            return indices, None

class LeRobotBuffer:
    def __init__(self, size: int, config: LeRobotDatasetConfig):
        buffers = {}
        for key, feature in config.features.items():
            base_shape = feature.shape
            if base_shape == (1,) and key in DEFAULT_FEATURES:
                base_shape = ()
            delta_idx = config.delta_indices.get(key)
            if delta_idx is not None:
                shape = (size, delta_idx.size, *base_shape)
            else:
                shape = (size, *base_shape)
            dtype = feature.dtype
            if dtype in {"video", "image"}:
                dtype = np.uint8
            else:
                dtype = np.dtype(dtype)
            buffers[key] = np.empty(shape, dtype=dtype)
            if delta_idx is not None:
                buffers[f"{key}_is_pad"] = np.empty((size, delta_idx.size), dtype=bool)
        if config.dataset_index is not None:
            buffers["dataset_index"] = np.empty((size,), dtype=np.int64)
        self._size = size
        self._buffers = buffers

    def __len__(self):
        return self._size
    
    def assign(self, indices: np.ndarray, data: dict[str, np.ndarray]):
        assert self._buffers.keys() == data.keys(), f"Buffer keys {self._buffers.keys()} do not match data keys {data.keys()}"
        for key, value in data.items():
            self._buffers[key][indices] = value

    def get(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        size = np.prod(shape, dtype=np.int64)
        assert size <= self._size, f"Requested size {size} exceeds buffer size {self._size}"
        def _format(a: np.ndarray) -> np.ndarray:
            a = a[:size]
            return a.reshape(shape + a.shape[1:])
        return {key: _format(self._buffers[key]) for key in self._buffers}

class LeRobotDatasetEpisodeList:
    def __init__(self, episodes: list[LeRobotDatasetEpisode], config: LeRobotDatasetConfig):
        self._episodes = episodes
        self._config = config
        sizes = np.array([ep._size for ep in episodes], dtype=np.int64)
        splits = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=splits[1:])
        self._splits = splits
        self._size = splits[-1]
        self._indexer = np.arange(self._size, dtype=np.int64)
        self._episode_indexer = np.repeat(np.arange(len(episodes)), sizes)
        self._buffer = None
    
    def _get_buffer(self, size: int):
        if self._buffer is None or len(self._buffer) < size:
            self._buffer = LeRobotBuffer(size, self._config)
        return self._buffer

    def __len__(self):
        return self._size

    def __getitem__(self, idx: SupportsIndex | slice | Sequence[int]) -> dict[str, np.ndarray]:
        try:
            # Fast path for the common case of scalar index
            global_index = operator.index(idx)
            episode_index = self._episode_indexer[global_index].item()
            local_index = global_index - self._splits[episode_index].item()
            return self._episodes[episode_index].read(local_index)
        except TypeError:
            pass
        global_indices = self._indexer[idx]
        raw_shape = global_indices.shape
        global_indices = global_indices.ravel()
        episode_indices = self._episode_indexer[global_indices]
        buffer = self._get_buffer(len(global_indices))

        episode_map = group_indices(episode_indices)

        for episode_index, group in episode_map.items():
            episode = self._episodes[episode_index]
            local_indices = global_indices[group] - self._splits[episode_index]
            buffer.assign(group, episode.read(local_indices))

        return buffer.get(raw_shape)

def group_indices(arr: np.ndarray) -> dict[int, np.ndarray]:
    assert arr.ndim == 1 and arr.size > 0
    sorted_indices = np.argsort(arr)
    sorted_values = arr[sorted_indices]
    change_points = np.where(sorted_values[1:] != sorted_values[:-1])[0] + 1
    groups = np.split(sorted_indices, change_points)
    unique_values = [sorted_values[0]] + sorted_values[change_points].tolist()
    return dict(zip(unique_values, groups))

class LeRobotDataset:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        *,
        keys: tuple[str, ...] | None = None,
        _dataset_index: int | None = None,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s

        self.meta = LeRobotDatasetMetadata(repo_id, self.root)
        # self.episodes = episodes if episodes is not None else list(range(self.meta.total_episodes))
        if episodes is None:
            episodes = list(range(self.meta.total_episodes))
        def _normalize_and_validate_episode(i: int):
            if i < 0:
                i += self.meta.total_episodes
            assert 0 <= i < self.meta.total_episodes, f"Episode index {i} out of range [0, {self.meta.total_episodes})"
            return i
        self.episodes = [_normalize_and_validate_episode(i) for i in episodes]

        fps = self.meta.fps
        delta_indices = {}
        if delta_timestamps is not None:
            for key, delta_ts in delta_timestamps.items():
                delta_ts = np.array(delta_ts, dtype=np.float64)
                delta_idxf = delta_ts * fps
                delta_idx = np.rint(delta_idxf).astype(np.int64)
                assert np.all(np.abs(delta_idxf - delta_idx) < tolerance_s * fps), \
                    f"Delta timestamps for {key} are not multiples of frame interval {1/fps}: {delta_ts}"
                delta_indices[key] = delta_idx

        if keys is None:
            keys = self.meta.features.keys()
        self._config = LeRobotDatasetConfig(
            features={ k: self.meta.features[k] for k in keys },
            delta_indices=delta_indices,
            fps=fps,
            dataset_index=_dataset_index,
        )

        def _make_episode(index: int):
            data_path = self.meta.get_data_file_path(index)
            video_paths = {k: self.meta.get_video_file_path(index, k) for k in self._config.video_keys}
            return LeRobotDatasetEpisode(data_path, video_paths, self._config)
        self._episode_list = LeRobotDatasetEpisodeList([_make_episode(i) for i in self.episodes], self._config)

    def __len__(self):
        return len(self._episode_list)
    
    def __getitem__(self, index: int | slice | Sequence[int]) -> dict[str, np.ndarray]:
        data =  self._episode_list[index]
        if "task_index" in data:
            task_index = data["task_index"]
            if task_index.shape == ():
                data["task"] = self.meta.tasks[task_index].task
            else:
                assert task_index.ndim == 1
                data["task"] = [self.meta.tasks[i].task for i in task_index]
        return data

class MultiLeRobotDataset:
    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict[str, list[int]] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict[str, float] | None = None,
        *,
        keys: tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        if keys is None:
            metas = [LeRobotDatasetMetadata(repo_id, self.root / repo_id) for repo_id in repo_ids]
            all_feature_keys = [set(meta.features.keys()) for meta in metas]
            intersection_keys = set.intersection(*all_feature_keys)
            if len(intersection_keys) == 0:
                raise RuntimeError(
                    "Multiple datasets were provided but they had no keys common to all of them. "
                    "The multi-dataset functionality currently only keeps common keys."
                )
            for repo_id, feature_keys in zip(self.repo_ids, all_feature_keys, strict=True):
                extra_keys = feature_keys.difference(intersection_keys)
                if extra_keys:
                    print(
                        f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                        "other datasets."
                    )
            keys = [key for key in metas[0].features.keys() if key in intersection_keys]  # Restore order

        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                keys=keys,
                _dataset_index=i
            )
            for i, repo_id in enumerate(self.repo_ids)
        ]
        self._episode_list = LeRobotDatasetEpisodeList(
            [episode for ds in self._datasets for episode in ds._episode_list._episodes],
            self._datasets[0]._config
        )

    def __len__(self):
        return len(self._episode_list)

    def __getitem__(self, index: int | slice | Sequence[int]) -> dict[str, np.ndarray]:
        data =  self._episode_list[index]
        if "task_index" in data:
            task_index = data["task_index"]
            dataset_index = data["dataset_index"]
            if task_index.shape == ():
                data["task"] = self._datasets[dataset_index].meta.tasks[task_index].task
            else:
                assert task_index.ndim == 1
                data["task"] = [self._datasets[i].meta.tasks[j].task for i, j in zip(dataset_index, task_index, strict=True)]
        return data

import json
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from dataclasses_json import config, dataclass_json, Undefined, DataClassJsonMixin
import numpy as np

from mini_lerobot.constant import CODEBASE_VERSION, DEFAULT_CHUNK_SIZE, DEFAULT_PARQUET_PATH, DEFAULT_VIDEO_PATH, HF_LEROBOT_HOME

T = TypeVar("T")

def compatible_dict_access(cls: type[T]) -> type[T]:
    assert is_dataclass(cls), "compatible_dict_access can only be applied to dataclasses"
    field_names = frozenset(f.name for f in fields(cls))
    def __getitem__(self, key):
        if key in field_names:
            return getattr(self, key)
        raise KeyError(f"{key} is not a valid field name")
    cls.__getitem__ = __getitem__
    return cls

def drop_none_field():
    return field(metadata=config(exclude=lambda x: x is None), default=None)

def array_field():
    return field(metadata=config(encoder=lambda x: x.tolist(), decoder=lambda x: np.array(x)))

def scalar_array_field():
    # represent scalar x as [x] in JSON to match LeRobot behavior
    return field(metadata=config(encoder=lambda x: [x], decoder=lambda x: x[0]))

def load_json(cls: type[T], path: Path) -> T:
    assert issubclass(cls, DataClassJsonMixin)
    with open(path, "r") as f:
        data = json.load(f)
    return cls.from_dict(data)

def dump_json(obj: T, path: Path):
    assert isinstance(obj, DataClassJsonMixin)
    with open(path, "w") as f:
        json.dump(obj.to_dict(), f, ensure_ascii=False, indent=4)

def load_json_lines(cls: type[T], path: Path) -> list[T]:
    assert issubclass(cls, DataClassJsonMixin)
    with open(path, "r") as f:
        return [cls.from_dict(json.loads(line)) for line in f]

def dump_json_lines(obj: Iterable[T], path: Path):
    with open(path, "w") as f:
        for item in obj:
            assert isinstance(item, DataClassJsonMixin)
            json.dump(item.to_dict(), f, ensure_ascii=False)
            f.write("\n")

@compatible_dict_access
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class LeRobotDatasetFeature:
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None = field(default=None)
    info: dict[str, str | int | float | bool] | None = drop_none_field()

    def fill_video_info(self, fps: int):
        assert self.dtype == "video", "fill_video_info can only be called on video features"
        height, width, channels = self.shape
        self.info = {
            "video.height": height,
            "video.width": width,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": False
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LeRobotDatasetFeature": ...

DEFAULT_FEATURES = {
    "timestamp": LeRobotDatasetFeature(dtype="float32", shape=(1,), names=None),
    "frame_index": LeRobotDatasetFeature(dtype="int64", shape=(1,), names=None),
    "episode_index": LeRobotDatasetFeature(dtype="int64", shape=(1,), names=None),
    "index": LeRobotDatasetFeature(dtype="int64", shape=(1,), names=None),
    "task_index": LeRobotDatasetFeature(dtype="int64", shape=(1,), names=None),
}

@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class LeRobotDatasetInfo:
    """Correspond to meta/info.json"""
    codebase_version: str
    robot_type: str | None
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    splits: dict[str, str]
    data_path: str
    video_path: str | None
    features: dict[str, LeRobotDatasetFeature]

@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class LeRobotDatasetEpisodeInfo:
    episode_index: int
    tasks: list[str]
    length: int

@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class LeRobotDatasetFeatureStat:
    min: np.ndarray = array_field()
    max: np.ndarray = array_field()
    mean: np.ndarray = array_field()
    std: np.ndarray = array_field()
    count: int = scalar_array_field()

    @staticmethod
    def aggregate(stats: Sequence["LeRobotDatasetFeatureStat"]) -> "LeRobotDatasetFeatureStat":
        assert len(stats) > 0, "Cannot aggregate empty stats"
        all_min = np.stack([s.min for s in stats], axis=0)
        all_max = np.stack([s.max for s in stats], axis=0)
        all_mean = np.stack([s.mean for s in stats], axis=0)
        all_std = np.stack([s.std for s in stats], axis=0)
        all_count = np.array([s.count for s in stats], dtype=np.int64).reshape((-1,) + (1,) * (all_mean.ndim - 1))
        min = all_min.min(axis=0)
        max = all_max.max(axis=0)
        total_count = all_count.sum().item()
        mean = (all_mean * all_count).sum(axis=0) / total_count
        weighted_vars = all_count * (all_std**2 + all_mean**2)
        variance = weighted_vars.sum(axis=0) / total_count - mean**2
        std = np.sqrt(np.maximum(variance, 0))
        return LeRobotDatasetFeatureStat(min=min, max=max, mean=mean, std=std, count=total_count)

@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class LeRobotDatasetEpisodeStats:
    episode_index: int
    stats: dict[str, LeRobotDatasetFeatureStat]

@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class LeRobotDatasetTask:
    task_index: int
    task: str

@dataclass
class _EpisodeSummary:
    episode_index: int
    episode_chunk: int
    tasks: list[str]
    length: int
    stats: dict[str, LeRobotDatasetFeatureStat]

class LeRobotDatasetMetadata:
    def __init__(self, repo_id: str, root: Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        self.meta_root = LeRobotDatasetMetadata._ensure_meta_root(self.root)
        self.load_metadata()

    def load_metadata(self):
        # Load
        info = load_json(LeRobotDatasetInfo, self.meta_root / "info.json")
        episodes = load_json_lines(LeRobotDatasetEpisodeInfo, self.meta_root / "episodes.jsonl")
        episodes_stats = load_json_lines(LeRobotDatasetEpisodeStats, self.meta_root / "episodes_stats.jsonl")
        tasks = load_json_lines(LeRobotDatasetTask, self.meta_root / "tasks.jsonl")
        # Validate
        assert info.total_episodes == len(episodes) == len(episodes_stats), "Mismatch in episode counts"
        assert info.total_frames == sum(e.length for e in episodes), "Mismatch in frame counts"
        assert info.total_tasks == len(tasks), "Mismatch in task counts"
        assert all(e.episode_index == i for i, e in enumerate(episodes))
        assert all(s.episode_index == i for i, s in enumerate(episodes_stats))
        assert all(t.task_index == i for i, t in enumerate(tasks))
        task_to_task_index = {t.task: t.task_index for t in tasks}
        assert len(task_to_task_index) == len(tasks), "Duplicate tasks found"
        # Store
        self.info = info
        self.episodes = episodes
        self.episodes_stats = episodes_stats
        self.tasks = tasks
        self._task_to_task_index = task_to_task_index

    @property
    def data_path(self):
        return self.info.data_path

    @property
    def video_path(self):
        return self.info.video_path

    @property
    def robot_type(self):
        return self.info.robot_type

    @property
    def fps(self):
        return self.info.fps

    @property
    def features(self):
        return self.info.features

    @property
    def image_keys(self):
        return [key for key, ft in self.features.items() if ft.dtype == "image"]

    @property
    def video_keys(self):
        return [key for key, ft in self.features.items() if ft.dtype == "video"]

    @property
    def camera_keys(self):
        return [key for key, ft in self.features.items() if ft.dtype in ("video", "image")]

    @property
    def names(self):
        return {key: ft.names for key, ft in self.features.items()}

    @property
    def shapes(self):
        return {key: ft.shape for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return self.info.total_episodes

    @property
    def total_frames(self) -> int:
        return self.info.total_frames

    @property
    def total_tasks(self) -> int:
        return self.info.total_tasks

    @property
    def total_chunks(self) -> int:
        return self.info.total_chunks

    @property
    def chunks_size(self) -> int:
        return self.info.chunks_size

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        return self.root / self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        return self.root / self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    def upsert_task(self, task: str) -> int:
        if task not in self._task_to_task_index:
            self.info.total_tasks += 1
            self.tasks.append(LeRobotDatasetTask(task_index=len(self.tasks), task=task))
            self._task_to_task_index[task] = len(self.tasks) - 1
        return self._task_to_task_index[task]

    def add_episode(self, summary: _EpisodeSummary):
        self.info.total_episodes += 1
        self.info.total_chunks = (self.info.total_episodes + self.info.chunks_size - 1) // self.info.chunks_size  # Ceil divide
        self.info.splits = {"train": f"0:{self.info.total_episodes}"}
        self.info.total_frames += summary.length
        self.info.total_videos += len(self.video_keys)
        assert summary.episode_index == len(self.episodes) and summary.episode_chunk == self.info.total_chunks - 1
        self.episodes.append(LeRobotDatasetEpisodeInfo(episode_index=summary.episode_index, tasks=summary.tasks, length=summary.length))
        self.episodes_stats.append(LeRobotDatasetEpisodeStats(episode_index=summary.episode_index, stats=summary.stats))
    
    @property
    def stats(self):
        return {
            k: LeRobotDatasetFeatureStat.aggregate([episode.stats[k] for episode in self.episodes_stats])
            for k in self.features.keys()
        }

    @staticmethod
    def _ensure_meta_root(root: Path) -> Path:
        meta_root = root / "meta"
        if not meta_root.exists():
            raise FileNotFoundError(f"Metadata directory not found: {meta_root}")
        if not meta_root.is_dir():
            raise NotADirectoryError(f"Metadata root is not a directory: {meta_root}")
        return meta_root
    
    @staticmethod
    def create(
        repo_id: str,
        fps: int,
        features: dict[str, LeRobotDatasetFeature | dict],
        robot_type: str | None = None,
        root: Path | None = None,
    ):
        root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        meta_root = root / "meta"
        meta_root.mkdir(parents=False, exist_ok=False)  # Disallow overwriting
        features = {k: LeRobotDatasetFeature.from_dict(v) if isinstance(v, dict) else v for k, v in features.items()}
        assert features.keys().isdisjoint(DEFAULT_FEATURES.keys())
        features.update(DEFAULT_FEATURES)
        use_videos = any(ft.dtype == "video" for ft in features.values())
        for feature in features.values():
            if feature.dtype == "video":
                feature.fill_video_info(fps)
        info = LeRobotDatasetInfo(
            codebase_version=CODEBASE_VERSION,
            robot_type=robot_type,
            total_episodes=0,
            total_frames=0,
            total_tasks=0,
            total_videos=0,
            total_chunks=0,
            chunks_size=DEFAULT_CHUNK_SIZE,
            fps=fps,
            splits={},
            data_path=DEFAULT_PARQUET_PATH,
            video_path=DEFAULT_VIDEO_PATH if use_videos else None,
            features=features,
        )
        dump_json(info, meta_root / "info.json")
        for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
            (meta_root / name).touch(exist_ok=False)
        return LeRobotDatasetMetadata(repo_id, root)

    def dump(self):
        dump_json(self.info, self.meta_root / "info.json")
        dump_json_lines(self.episodes, self.meta_root / "episodes.jsonl")
        dump_json_lines(self.episodes_stats, self.meta_root / "episodes_stats.jsonl")
        dump_json_lines(self.tasks, self.meta_root / "tasks.jsonl")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: {self.total_episodes},\n"
            f"    Total frames: {self.total_frames},\n"
            f"    Features: {list(self.features.keys())},\n"
            "})"
        )

if __name__ == "__main__":
    from pathlib import Path
    root = Path("autobio-bench/insert-blender-2")
    # meta = LeRobotDatasetMetadata.load(root)
    # from pprint import pprint
    # print(meta.info)
    
    features ={
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": None,
        },
        "actions": {
            "dtype": "float32",
            "shape": (7, 8),
            "names": ["actions"],
        },
    }
    for camera_key in ['a']:
        features[camera_key] = {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        }
    for camera_key in ['b']:
        features[camera_key] = {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        }

    metadata = LeRobotDatasetMetadata.create(
        root,
        fps=50,
        features=features,
        robot_type=None
    )
    breakpoint()

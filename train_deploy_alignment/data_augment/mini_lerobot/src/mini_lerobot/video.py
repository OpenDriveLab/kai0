import os
from pathlib import Path
import subprocess
import threading

import av
from av import VideoCodecContext, VideoFrame
from av.container import InputContainer
from av.codec.context import ThreadType
import numpy as np

from mini_lerobot.metadata import LeRobotDatasetFeatureStat

def use_single_thread():
    return os.getenv("FFMPEG_SINGLE_THREAD", "0") == "1"

def transcode_video_simple(src: Path, dst: Path):
    av1_params = []
    extra_params = []
    if use_single_thread():
        av1_params.append("lp=1")
        extra_params += ("-threads", "1")
    cmd = [
        "ffmpeg",
        "-nostdin", "-n",  # interactive tty is problematic in multiprocessing
        "-i", str(src),
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-crf", "30",
        "-g", "2",
        "-svtav1-params", ":".join(av1_params),
        "-hide_banner", "-loglevel", "error",
        *extra_params,
        str(dst)
    ]
    env = {
        "SVT_LOG": "1"
    }
    subprocess.run(cmd, env=env, check=True)

def validate_and_compute_stats(video_path: Path, key: str, fps: int, num_frames: int, height: int, width: int):
    with av.open(video_path) as container:
        assert len(container.streams.video) == 1
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        if use_single_thread():
            codec_context.thread_type = ThreadType.NONE
            codec_context.thread_count = 1
        assert stream.guessed_rate == fps, f"Video {key} has fps {stream.guessed_rate}, expected {fps}"
        assert stream.frames == num_frames, f"Video {video_path} has {stream.frames} frames, expected {num_frames}"
        assert stream.height == height and stream.width == width, f"Video {key} has resolution {(stream.height, stream.width)}, expected {(height, width)}"
        return get_video_feature_stats(container, codec_context, height, width, num_frames)

def get_video_feature_stats(container: InputContainer, codec_context: VideoCodecContext, height: int, width: int, num_frames: int):
    min_val = np.full((3,), 255, dtype=np.uint8)
    max_val = np.full((3,), 0, dtype=np.uint8)
    sum_val = np.zeros((3,), dtype=np.uint64)
    sos_val = np.zeros((3,), dtype=np.uint64)
    count = 0
    axis = (0, 1)
    for packet in container.demux(video=0):
        for frame in codec_context.decode(packet):
            rgb_array = frame.to_rgb().to_ndarray()
            assert rgb_array.shape == (height, width, 3)
            min_val = np.minimum(min_val, rgb_array.min(axis=axis))
            max_val = np.maximum(max_val, rgb_array.max(axis=axis))
            sum_val += rgb_array.sum(axis=axis, dtype=np.uint64)
            sos_val += (rgb_array.astype(np.uint16) ** 2).sum(axis=axis, dtype=np.uint64)
            count += 1
    assert count == num_frames

    num_pixels = count * height * width
    mean = sum_val / num_pixels
    var = sos_val / num_pixels - mean ** 2 * (num_pixels / (num_pixels - 1))
    std = np.sqrt(var)

    return LeRobotDatasetFeatureStat(
        # To match LeRobot format
        min=min_val[:, None, None] / 255,
        max=max_val[:, None, None] / 255,
        mean=mean[:, None, None] / 255,
        std=std[:, None, None] / 255,
        count=count,
    )

class InMemoryVideo:
    def __init__(self, path: Path, cfr: int):
        with av.open(path) as container:
            assert len(container.streams.video) == 1, f"Expected one video stream, got {len(container.streams.video)}"
            self._stream = container.streams.video[0]
            self._stream.thread_type = "NONE"
            self._stream.thread_count = 1
            self._cc = self._stream.codec_context

            duration = 1 / self._stream.time_base / cfr
            assert duration.denominator == 1, f"Non-integer duration: {duration}"
            duration = duration.numerator

            packets = list(container.demux(video=0))
            assert packets[-1].pts is None
            assert packets[0].pts == 0 and packets[0].is_keyframe
            packets = packets[:-1]
            pts = np.array([p.pts for p in packets], dtype=np.int64)
            dts = np.array([p.dts for p in packets], dtype=np.int64)
            assert np.all(np.diff(pts) == duration), f"Unexpected PTS: {pts}"
            assert np.all(np.diff(dts) == duration), f"Unexpected DTS: {dts}"
            self._duration = duration
            is_keyframe = np.array([p.is_keyframe for p in packets], dtype=bool)
            keyframes = np.flatnonzero(is_keyframe)
            keyframe_map = np.zeros(len(packets), dtype=np.int64)
            keyframe_map[keyframes] = keyframes
            np.maximum.accumulate(keyframe_map, out=keyframe_map)
            self._keyframe_map = keyframe_map
            self._packets = packets

        height = self._stream.height
        width = self._stream.width
        self._shape = (height, width, 3)

        self._current_keyframe_index: int = None
        self._current_frame_index: int = None
        self._current_frame: VideoFrame = None
        self._lock = threading.Lock()

    def read(self, indices: int | np.ndarray):
        if isinstance(indices, int):
            return self._read_video_frame(indices)
        buffer = np.empty(indices.shape + self._shape, dtype=np.uint8)
        flat_indices = indices.ravel()
        flat_buffer = buffer.reshape((-1,) + self._shape)
        flat_indices_argsort = np.argsort(flat_indices)
        flat_indices_ordered = flat_indices[flat_indices_argsort]

        for dest_index, frame_index in zip(flat_indices_argsort, flat_indices_ordered):
            flat_buffer[dest_index] = self._read_video_frame(frame_index)

        return buffer

    def _read_video_frame(self, this_frame_index: int):
        with self._lock:
            return self._unsafe_read_video_frame(this_frame_index)

    def _unsafe_read_video_frame(self, this_frame_index: int):
        this_keyframe_index = self._keyframe_map[this_frame_index]
        if this_keyframe_index != self._current_keyframe_index or this_frame_index < self._current_frame_index:
            # Seek
            self._cc.flush_buffers()
            self._current_keyframe_index = self._current_frame_index = this_keyframe_index
            self._current_frame = self._decode_frame(self._current_frame_index)
        while self._current_frame_index < this_frame_index:
            self._current_frame_index += 1
            self._current_frame = self._decode_frame(self._current_frame_index)
        assert self._current_frame.pts == this_frame_index * self._duration, f"Expected PTS {this_frame_index * self._duration}, got {self._current_frame.pts}"
        self._current_frame = self._current_frame.to_rgb()
        return np.asarray(self._current_frame.planes[0]).reshape(self._shape)

    def _decode_frame(self, frame_index: int) -> VideoFrame:
        frames = self._cc.decode(self._packets[frame_index])
        assert len(frames) == 1, f"Expected one frame, got {len(frames)}"
        return frames[0]

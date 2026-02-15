#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 dual-arm DAgger collection: policy runs on a remote server (WebSocket),
local IPC runs ROS2 + arms and cameras. Supports intervention (d) and episode save (s).
"""

import argparse
import time
import threading
import queue
import json
import numpy as np
import cv2
import os
import signal
import sys
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from sensor_msgs.msg import JointState
import termios
import tty
import select
from pathlib import Path
import logging
import h5py
import dm_env
import av

try:
    import pyrealsense2 as rs
except ImportError:
    print("Warning: pyrealsense2 not installed; camera features disabled.")

# ===== ROS2 =====
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
# import 
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

try:
    from arx5_arm_msg.msg import RobotStatus, RobotCmd
    print("[INFO] arx5_arm_msg loaded")
except ImportError:
    print("[ERROR] arx5_arm_msg not found; install ARX5 message package.")
    from sensor_msgs.msg import JointState
    RobotStatus = JointState
    RobotCmd = JointState
    print("[WARN] Using JointState as fallback for RobotCmd")


# ===== openpi_client =====
from openpi_client import image_tools, websocket_client_policy

# Globals
CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
stream_buffer = None
observation_window = deque(maxlen=2)
lang_embeddings = "hang the cloth"
dagger_mode_active = False
dagger_mode_lock = threading.Lock()
save_data_requested = False
save_data_lock = threading.Lock()
collection_active = False
collection_lock = threading.Lock()

# Debug / history
published_actions_history = []  # list[np.ndarray(shape=(14,))]
observed_qpos_history = []      # list[np.ndarray(shape=(14,))]
inferred_chunks = []            # list[dict(start_step:int, chunk:np.ndarray[chunk,14])]
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()


def encode_video_frames(
    images: np.ndarray,
    dst: Path,
    fps: int,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 23,
    fast_decode: int = 0,
    log_level: int = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    if vcodec not in {"h264", "hevc", "libx264", "libx265", "libsvtav1"}:
        raise ValueError(f"Unsupported codec {vcodec}")
    video_path = Path(dst)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)
    if (vcodec in {"libsvtav1", "hevc", "libx265"}) and pix_fmt == "yuv444p":
        pix_fmt = "yuv420p"
    h, w, _ = images[0].shape
    options = {}
    for k, v in {"g": g, "crf": crf}.items():
        if v is not None:
            options[k] = str(v)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        options[key] = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)
    with av.open(str(video_path), "w") as out:
        stream = out.add_stream(vcodec, fps, options=options)
        stream.pix_fmt, stream.width, stream.height = pix_fmt, w, h
        for i, img in enumerate(images):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"Encoding frame {i+1}")
        for pkt in stream.encode():
            out.mux(pkt)
    if log_level is not None:
        av.logging.restore_default_callback()
    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path}")


def create_video_from_images(images, output_path, fps=30, codec="libx264", quality=23):
    if not images:
        raise ValueError("No image data")
    print(f"Encoding video, codec: {codec} CRF: {quality}")
    encode_video_frames(np.asarray(images), Path(output_path), fps=fps, vcodec=codec, crf=quality, overwrite=True)
    print(f"Video saved to: {output_path}")


def _save_intervention_json(dataset_path: str, interventions: List[int]):
    """Save intervention labels to a JSON file
    
    Args:
        dataset_path: path without suffix
        interventions: list of labels (0=policy, 1=human)
    """
    intervention_np = np.array(interventions, dtype=np.uint8)
    total_frames = len(interventions)
    intervention_frames = int(np.sum(intervention_np == 1))
    auto_frames = int(np.sum(intervention_np == 0))
    
    # Compute segment info
    changes = np.where(np.diff(intervention_np) != 0)[0] + 1
    segments = []
    prev_idx = 0
    
    for change_idx in changes:
        segments.append({
            "start_frame": int(prev_idx),
            "end_frame": int(change_idx - 1),
            "num_frames": int(change_idx - prev_idx),
            "type": "intervention" if intervention_np[prev_idx] == 1 else "auto"
        })
        prev_idx = int(change_idx)
    
    # Last segment
    segments.append({
        "start_frame": int(prev_idx),
        "end_frame": int(total_frames - 1),
        "num_frames": int(total_frames - prev_idx),
        "type": "intervention" if intervention_np[prev_idx] == 1 else "auto"
    })
    
    # Extract episode index from path
    episode_num = -1
    try:
        basename = os.path.basename(dataset_path)
        if basename.startswith("episode_"):
            episode_num = int(basename.replace("episode_", ""))
    except:
        pass
    
    # Build JSON
    json_data = {
        "episode": episode_num,
        "total_frames": total_frames,
        "intervention_frames": intervention_frames,
        "auto_frames": auto_frames,
        "intervention_ratio": round(intervention_frames / total_frames, 4) if total_frames > 0 else 0,
        "num_segments": len(segments),
        "segments": segments,
        "intervention": [int(x) for x in interventions]
    }
    
    json_path = dataset_path + "_intervention.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f'\033[36m  Intervention JSON: {json_path}\033[0m')


def save_data(timesteps, actions, dataset_path, interventions=None):
    """Save HDF5 in joint-only format (no images)
    
    Args:
        timesteps: list of timesteps
        actions: list of actions
        dataset_path: path without suffix
        interventions: optional labels (0=policy, 1=human)
    """
    data_size = len(actions)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': []
    }

    for ts, action in zip(timesteps, actions):
        obs = ts.observation
        data_dict['/observations/qpos'].append(obs['qpos'])
        data_dict['/observations/qvel'].append(obs['qvel'])
        data_dict['/observations/effort'].append(obs['effort'])
        data_dict['/action'].append(action)

    t0 = time.time()
    print('\033[33m>>> Saving HDF5...\033[0m')

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        obs_grp = root.create_group('observations')
        obs_grp.create_dataset('qpos', (data_size, 14), dtype='float32')
        obs_grp.create_dataset('qvel', (data_size, 14), dtype='float32')
        obs_grp.create_dataset('effort', (data_size, 14), dtype='float32')
        root.create_dataset('action', (data_size, 14), dtype='float32')

        for name, arr in data_dict.items():
            root[name][...] = np.array(arr)
        
        # Save intervention labels if provided
        if interventions is not None and len(interventions) == data_size:
            root.create_dataset('intervention', (data_size,), dtype='uint8')
            root['intervention'][...] = np.array(interventions, dtype=np.uint8)
            intervention_count = sum(interventions)
            print(f'\033[36m  Intervention frames: {intervention_count}/{data_size} ({100*intervention_count/data_size:.1f}%)\033[0m')

    print(f'[INFO] HDF5 saved in {time.time() - t0:.1f}s')
    print(f'\033[32m  Path: {dataset_path}.hdf5\033[0m')
    print(f'\033[32m  Frames: {data_size}\033[0m')
    # Export intervention JSON
    if interventions is not None and len(interventions) == data_size:
        _save_intervention_json(dataset_path, interventions)


def save_videos(timesteps_copy, dataset_path, camera_names, fps=30):
    """Save three camera streams as videos."""
    if len(timesteps_copy) == 0:
        print("\033[31mNo image data for video.\033[0m")
        return

    dataset_dir = os.path.dirname(dataset_path)
    episode_name = os.path.basename(dataset_path)
    print("\033[33m>>> Exporting video...\033[0m")
    t0 = time.time()

    for cam_name in camera_names:
        video_dir = os.path.join(dataset_dir, 'video', cam_name)
        os.makedirs(video_dir, exist_ok=True)
        images = [ts.observation['images'][cam_name] for ts in timesteps_copy]
        video_path = os.path.join(video_dir, f'{episode_name}.mp4')
        create_video_from_images(images, video_path, fps)
        print(f"\033[36m  Video: {cam_name} -> {video_path}\033[0m")

    sample_img = timesteps_copy[0].observation['images'][camera_names[0]]
    h, w = sample_img.shape[:2]
    print(f'[INFO] Video saved in {time.time() - t0:.1f}s')
    print(f'\033[32m  Resolution: {w}x{h}, frames: {len(timesteps_copy)}, fps: {fps}\033[0m')

# def apply_gripper_binary(act: np.ndarray, open_val: float, close_val: float, thresh: float) -> np.ndarray:
#     out = act.copy()
#     out[6]  = close_val if act[6] >= thresh else open_val   # 左爪
#     out[13] = close_val if act[13] >= thresh else open_val  # 右爪
#     return out


class StreamActionBuffer:
    """
    Action chunk queue (same design as agilex inference).
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()                 
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            
            # Build old sequence
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def clear(self):
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0


class SimpleDAggerCollector:
    """DAgger collector: async queue + background thread, non-blocking."""

    def __init__(self, camera_names, dataset_dir="./data", dataset_name="dagger_arx", video_fps: int = 30):
        self.camera_names = camera_names
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.video_fps = video_fps
        self.full_dataset_dir = os.path.join(dataset_dir, dataset_name)
        os.makedirs(self.full_dataset_dir, exist_ok=True)
        
        # Thread-safe data queue
        self._frame_queue = queue.Queue(maxsize=10000)
        self._lock = threading.Lock()
        
        # Background write thread
        self._writer_thread = None
        self._writer_running = False
        
        # Current episode (managed by background thread)
        self._current_timesteps: List[dm_env.TimeStep] = []
        self._current_actions: List[np.ndarray] = []
        self._current_interventions: List[int] = []
        
        self.is_collecting = False
        self.frame_count = 0
        self.episode_idx = self._find_next_episode_idx()
        
        # Save state
        self._save_requested = False
        self._save_config = {}
        
        # Pause during master alignment
        self._paused = False

    def _find_next_episode_idx(self):
        if not os.path.exists(self.full_dataset_dir):
            return 0
        episodes = [f for f in os.listdir(self.full_dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
        if not episodes:
            return 0
        idxs = []
        for f in episodes:
            try:
                idxs.append(int(f.split("_")[1].split(".")[0]))
            except Exception:
                continue
        return max(idxs) + 1 if idxs else 0

    def _writer_loop(self):
        """Background thread: drain queue and handle save requests."""
        while self._writer_running:
            # Process queued frames
            try:
                frame_data = self._frame_queue.get(timeout=0.1)
                obs, action, intervention = frame_data
                self._current_timesteps.append(dm_env.TimeStep(dm_env.StepType.MID, None, None, obs))
                self._current_actions.append(np.asarray(action, dtype=float).copy())
                self._current_interventions.append(int(intervention))
            except queue.Empty:
                pass
            
            # Check save request
            if self._save_requested:
                self._do_save()
                self._save_requested = False

    def _do_save(self):
        """Perform save (called from background thread)."""
        if len(self._current_actions) == 0:
            print("[ERROR] No data to save")
            return
        
        dataset_path = os.path.join(self.full_dataset_dir, f"episode_{self.episode_idx}")
        export_video = self._save_config.get("export_video", True)
        video_fps = self._save_config.get("video_fps", self.video_fps)
        
        try:
            timesteps_copy = list(self._current_timesteps)
            actions_copy = list(self._current_actions)
            interventions_copy = list(self._current_interventions)
            
            save_data(timesteps_copy, actions_copy, dataset_path, interventions=interventions_copy)
            
            has_images = timesteps_copy and "images" in timesteps_copy[0].observation
            if export_video and has_images:
                save_videos(timesteps_copy, dataset_path, self.camera_names, fps=video_fps)
            
            print(f"\n[INFO] Saved: {dataset_path}.hdf5 ({len(actions_copy)} frames)")
            self.episode_idx += 1
            
                # Clear for new episode
            self._current_timesteps.clear()
            self._current_actions.clear()
            self._current_interventions.clear()
            self.frame_count = 0
            
            _print_dagger_status(
                f"Episode {self.episode_idx - 1} saved; new episode started.",
                "Press [d] for DAgger mode | [s] to save"
            )
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")
            import traceback
            traceback.print_exc()

    def start_collection(self):
        """Start collection (start background thread)."""
        with self._lock:
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._writer_running = True
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()
            
            # Clear queue and data
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._current_timesteps.clear()
            self._current_actions.clear()
            self._current_interventions.clear()
            self.frame_count = 0
            self.is_collecting = True
        
        print("\n" + "=" * 60)
        print(f"[INFO] DAgger collection started | Episode: {self.episode_idx} | Dir: {self.full_dataset_dir}")
        print("=" * 60 + "\n")

    def stop_collection(self):
        """Stop collection."""
        with self._lock:
            self.is_collecting = False
        if self.has_data():
            print(f"[WARN] Collection stopped; {self.frame_count} frames buffered. Press s to save.")
    
    def pause(self):
        """Pause collection (during alignment)."""
        self._paused = True
    
    def resume(self):
        """Resume collection."""
        self._paused = False

    def has_data(self):
        return self.frame_count > 0 or len(self._current_actions) > 0

    def add_frame(self, observation, action, intervention: int = 1):
        """Add one frame (non-blocking, queued).
        
        Args:
            observation: observation (deep-copied)
            action: action
            intervention: 0=policy, 1=human (default 1)
        """
        if not self.is_collecting or self._paused:
            return
        # Deep-copy observation
        obs_copy = {
            "qpos": np.asarray(observation["qpos"], dtype=float).copy(),
            "qvel": np.asarray(observation["qvel"], dtype=float).copy(),
            "effort": np.asarray(observation["effort"], dtype=float).copy(),
        }
        if "images" in observation:
            obs_copy["images"] = {k: v.copy() for k, v in observation["images"].items()}
        
        action_copy = np.asarray(action, dtype=float).copy()
        
        try:
            self._frame_queue.put_nowait((obs_copy, action_copy, intervention))
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"\r[DAGGER] Frames: {self.frame_count}", end="", flush=True)
        except queue.Full:
            print("[WARN] Queue full, dropping frame")

    def save_current_episode(self, export_video=True, video_fps=30):
        """Request save of current episode (non-blocking)."""
        if not self.has_data():
            print("[ERROR] No data to save")
            return False
        
        self._save_config = {"export_video": export_video, "video_fps": video_fps}
        self._save_requested = True
        print("[INFO] Save requested, processing in background...")
        return True
    
    def shutdown(self):
        """Stop background thread."""
        self._writer_running = False
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)


def _print_dagger_status(stage: str, hint: str = ""):
    """Print DAgger status and key hints."""
    print("\n" + "=" * 50)
    print(f"[DAGGER] {stage}")
    if hint:
        print(f"[HINT] {hint}")
    print("=" * 50)


def keyboard_monitor_thread():
    """Keyboard: d/Ctrl+Q enter DAgger, r resume, Space start collect, s save."""
    global dagger_mode_active, save_data_requested, collection_active
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not shutdown_event.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ord(ch) == 17 or ch.lower() == "d":
                    with dagger_mode_lock:
                        if not dagger_mode_active:
                            dagger_mode_active = True
                            _print_dagger_status(
                                "DAgger mode; inference paused.",
                                "Press [Space] to start collection | [r] to resume"
                            )
                        else:
                            print("\n[WARN] Already in DAgger mode")
                elif ch.lower() == "r":
                    with dagger_mode_lock:
                        if dagger_mode_active:
                            dagger_mode_active = False
                            _print_dagger_status(
                                "Exited DAgger; resuming inference.",
                                "Press [d] to enter DAgger again"
                            )
                        else:
                            print("\n[WARN] Already in inference mode")
                    with collection_lock:
                        collection_active = False
                elif ch == " ":
                    with dagger_mode_lock:
                        if dagger_mode_active:
                            with collection_lock:
                                if not collection_active:
                                    collection_active = True
                                    _print_dagger_status(
                                        "Collecting... Demonstrate with master arms.",
                                        "Press [s] to save | [r] to abort and resume"
                                    )
                                else:
                                    print("\n[WARN] Already collecting")
                        else:
                            print("\n[WARN] Press [d] to enter DAgger mode first")
                elif ch.lower() == "s":
                    # Save anytime (full-session collection)
                    with save_data_lock:
                        save_data_requested = True
                    print("\n[INFO] Save requested, processing in background...")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """
    Non-blocking inference thread (no rate limit).
    """
    global stream_buffer, observation_window, lang_embeddings

    # rate = rclpy.create_rate(args.inference_rate)
    rate = ros_operator.create_rate(args.inference_rate)
    consecutive_failures = 0
    max_consecutive_failures = 5

    while rclpy.ok() and not shutdown_event.is_set():
        try:
            # Pause inference in DAgger mode
            if dagger_mode_active:
                rate.sleep()
                continue
            time1 = time.time()
            
            # 1) Get latest observation
            update_observation_window(args, config, ros_operator)
            if getattr(args, "debug_timing", False):
                print(f"[timing] obs: {time.time() - time1:.4f}s")
            
            if len(observation_window) == 0:
                continue
                
            latest_obs = observation_window[-1]
            imgs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            
            # BGR->RGB & pad/resize
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            proprio = latest_obs["qpos"]

            # 2) Build payload
            payload = {
                "state": proprio,
                "images": {
                    "top_head":  imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left":  imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }

            # 3) Infer
            time1 = time.time()
            actions = policy.infer(payload)["actions"]
            if getattr(args, "debug_timing", False):
                print(f"[timing] infer: {time.time() - time1:.4f}s")

            # 4) Push to buffer
            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
                # print(f"[infer] 推入 chunk，长度={len(actions)}")   # ← 加这行
                
                # Record chunk for debug
                try:
                    step_now = max(len(published_actions_history), len(observed_qpos_history))
                    with inferred_chunks_lock:
                        inferred_chunks.append({
                            "start_step": int(step_now),
                            "chunk": np.asarray(actions, dtype=float).copy()
                        })
                except Exception:
                    pass
                
                consecutive_failures = 0
            elif actions is None:
                print("actions is None")
            elif len(actions) == 0:
                print("len(actions) == 0")

            rate.sleep()

        except Exception as e:
            print(f"[inference_fn_non_blocking_fast] {e}")
            consecutive_failures += 1
            
            # Error recovery
            if consecutive_failures >= max_consecutive_failures:
                print(f"[Infer] Consecutive failures: {consecutive_failures}, clearing buffer")
                stream_buffer.clear()
                consecutive_failures = 0
            
            try:
                rate.sleep()
            except:
                time.sleep(0.001)


def start_inference_thread(args, config, policy, ros_operator):
    inference_thread = threading.Thread(
        target=inference_fn_non_blocking_fast, 
        args=(args, config, policy, ros_operator)
    )
    inference_thread.daemon = True
    inference_thread.start()
    return inference_thread


def _on_sigint(signum, frame):
    """SIGINT handler."""
    try:
        shutdown_event.set()
    except Exception:
        pass


def update_observation_window(args, config, ros_operator):
    """Update observation window."""
    global observation_window
    
    if len(observation_window) == 0:
        # Init observation window
        observation_window.append({
            "qpos": None,
            "images": {
                config["camera_names"][0]: None,
                config["camera_names"][1]: None,
                config["camera_names"][2]: None,
            },
        })

    # Get sensor data
    frame = ros_operator.get_frame()
    if frame is None:
        return
        
    imgs, j_left, j_right = frame
    qpos = ros_operator.get_joint_positions(j_left, j_right)
    
    # JPEG compress (match training)
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    img_front = jpeg_mapping(imgs['cam_high'])
    img_left = jpeg_mapping(imgs['cam_left_wrist'])
    img_right = jpeg_mapping(imgs['cam_right_wrist'])

    observation_window.append({
        "qpos": qpos,
        "images": {
            config["camera_names"][0]: img_front,
            config["camera_names"][1]: img_right,
            config["camera_names"][2]: img_left,
        },
    })
    
    # Record qpos for debug
    try:
        observed_qpos_history.append(np.asarray(qpos, dtype=float).copy())
    except Exception:
        pass


class ARX5ROSController(Node):
    # ARX5 messages
    try:
        from arx5_arm_msg.msg import RobotStatus, RobotCmd
        print("[INFO] arx5_arm_msg loaded")
    except ImportError:
        print("[ERROR] arx5_arm_msg not found; install ARX5 message package.")
        from sensor_msgs.msg import JointState
        RobotStatus = JointState
        RobotCmd = JointState

    def __init__(self, args):
        super().__init__("arx5_controller")
        self.args = args
        self.bridge = CvBridge()

        self.last_qpos = None
        self.qpos_lock = threading.Lock()

        # Data cache
        self.joint_left_deque = deque(maxlen=2000)
        self.joint_right_deque = deque(maxlen=2000)

        # Publishers
        self.pub_left = self.create_publisher(RobotStatus, args.joint_cmd_topic_left, 10)
        self.pub_right = self.create_publisher(RobotStatus, args.joint_cmd_topic_right, 10)

        self.RobotStatus = RobotStatus
        # Subscribe to control commands
        self.create_subscription(
            RobotStatus,
            '/arm_master_l_cmd',
            self.left_arm_command_callback,
            10
        )

        self.create_subscription(
            RobotStatus,
            '/arm_master_r_cmd',
            self.right_arm_command_callback,
            10
        )

        self.get_logger().info(f"Sub left: {args.joint_state_topic_left}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_left,
            self.joint_left_callback,
            10
        )

        self.get_logger().info(f"Sub right: {args.joint_state_topic_right}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_right,
            self.joint_right_callback,
            10
        )

        # Data ready flags
        self.data_ready = {
            'joint_left': False,
            'joint_right': False,
            'cameras': False
        }
        self.data_ready_lock = threading.Lock()
        self.data_collector = SimpleDAggerCollector(
            camera_names=CAMERA_NAMES,
            dataset_dir=getattr(args, "dataset_dir", "./data"),
            dataset_name=getattr(args, "dataset_name", "dagger_arx"),
            video_fps=getattr(args, "video_fps", 30),
        )

        # Master alignment publishers
        self.master_ctrl_left_pub = self.create_publisher(RobotStatus, args.master_ctrl_cmd_topic_left, 10)
        self.master_ctrl_right_pub = self.create_publisher(RobotStatus, args.master_ctrl_cmd_topic_right, 10)
        self.master_cmd_cache = np.zeros(14, dtype=float)

        # Master state subs (for smooth start)
        self.master_left_deque = deque(maxlen=2000)
        self.master_right_deque = deque(maxlen=2000)
        # Subscribe to master state
        self.create_subscription(RobotStatus, args.master_status_topic_left, self.master_status_left_callback, 10)
        self.create_subscription(RobotStatus, args.master_status_topic_right, self.master_status_right_callback, 10)
        self.create_subscription(RobotStatus, getattr(args, "master_status_topic_left_ctrl", args.master_status_topic_left), self.master_status_left_callback, 10)
        self.create_subscription(RobotStatus, getattr(args, "master_status_topic_right_ctrl", args.master_status_topic_right), self.master_status_right_callback, 10)

        # Init cameras
        self.init_cameras()
        self.last_images: dict[str, np.ndarray] = {}

    def joint_left_callback(self, msg):
        """Left arm RobotStatus callback."""
        # print(f"[CB] 收到左臂 RobotStatus 消息")
        # print(f"  - joint_pos 长度: {len(msg.joint_pos)}")
        # print(f"  - 关节数据: {msg.joint_pos}")

        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_left'] = True

    def joint_right_callback(self, msg):
        """Right arm RobotStatus callback."""
        # print(f"[CB] 收到右臂 RobotStatus 消息")
        # print(f"  - joint_pos 长度: {len(msg.joint_pos)}")
        # print(f"  - 关节数据: {msg.joint_pos}")


        self.joint_right_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_right'] = True

    def left_arm_command_callback(self, msg):
        """Left arm command callback."""
        # print(f"[CB] 收到左臂控制命令")
        # print(f"  - joint_pos 长度: {len(msg.joint_pos)}")
        # print(f"  - 控制命令数据: {msg.joint_pos}")

        # Merge left cmd with right current pose and publish
        self.set_joint_positions(np.array(msg.joint_pos + self.joint_right_deque[-1].joint_pos))

    def right_arm_command_callback(self, msg):
        """Right arm command callback."""
        # print(f"[CB] 收到右臂控制命令")
        # print(f"  - joint_pos 长度: {len(msg.joint_pos)}")
        # print(f"  - 控制命令数据: {msg.joint_pos}")

        # Merge right cmd with left current pose and publish
        self.set_joint_positions(np.array(self.joint_left_deque[-1].joint_pos + msg.joint_pos))

    def get_joint_positions(self, j_left: RobotStatus, j_right: RobotStatus) -> np.ndarray:
        """Get 14-D joint position from RobotStatus."""
        # print(f"[DEBUG] 左臂 joint_pos: {j_left.joint_pos}")
        # print(f"[DEBUG] 右臂 joint_pos: {j_right.joint_pos}")

        left = list(j_left.joint_pos)
        right = list(j_right.joint_pos)

        # print(f"[DEBUG] 合并后维度: 左 {len(left)} + 右 {len(right)} = {len(left + right)}")

        q = np.array(left + right, dtype=float)
        with self.qpos_lock:
            self.last_qpos = q.copy()
        return q

        # return np.array(left + right, dtype=float)

    def get_slave_positions(self) -> Optional[np.ndarray]:
        """Get current slave joint angles (None if not ready)."""
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return None
        left = np.array(self.joint_left_deque[-1].joint_pos, dtype=float)
        right = np.array(self.joint_right_deque[-1].joint_pos, dtype=float)
        return np.concatenate([left, right])

    def master_status_left_callback(self, msg: RobotStatus):
        if len(self.master_left_deque) >= 2000:
            self.master_left_deque.popleft()
        self.master_left_deque.append(msg)

    def master_status_right_callback(self, msg: RobotStatus):
        if len(self.master_right_deque) >= 2000:
            self.master_right_deque.popleft()
        self.master_right_deque.append(msg)

    def get_master_positions(self) -> Optional[np.ndarray]:
        if len(self.master_left_deque) == 0 or len(self.master_right_deque) == 0:
            return None
        left = np.array(self.master_left_deque[-1].joint_pos, dtype=float)
        right = np.array(self.master_right_deque[-1].joint_pos, dtype=float)
        return np.concatenate([left, right])

    def reset_master_ctrl_target(self, repeats: int = 5, hz: float = 50.0):
        """Reset master_ctrl target to current slave pose, clear old commands."""
        slave = self.get_slave_positions()
        if slave is None:
            print("[WARN] No slave state; skip master_ctrl target reset")
            return False
        target = slave.copy()
        self.master_cmd_cache = target.copy()
        for _ in range(max(1, repeats)):
            if self.master_ctrl_left_pub is not None:
                msg_l = RobotStatus()
                msg_l.joint_pos = target[:7].tolist()
                self.master_ctrl_left_pub.publish(msg_l)
            if self.master_ctrl_right_pub is not None:
                msg_r = RobotStatus()
                msg_r.joint_pos = target[7:].tolist()
                self.master_ctrl_right_pub.publish(msg_r)
            time.sleep(1.0 / hz)
        print("[INFO] Reset master_ctrl target to current slave pose")
        return True

    def get_frame(self):
        """Get synced sensor data."""
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            # print(f"[DEBUG] 队列状态: 左臂{len(self.joint_left_deque)}, 右臂{len(self.joint_right_deque)}")
            return None
        
        # Latest joint data
        j_left = self.joint_left_deque[-1]
        j_right = self.joint_right_deque[-1]
        
        # print(f"[DEBUG] 获取到关节数据: 左臂{len(j_left.joint_pos)}, 右臂{len(j_right.joint_pos)}")
        
        # Camera images
        imgs = self.get_camera_images()
        if len(imgs) != 3:
            print(f"[DEBUG] 摄像头图像数量: {len(imgs)}")
            return None

        return imgs, j_left, j_right

    def init_cameras(self):
        """初始化RealSense摄像头"""
        try:
            import pyrealsense2 as rs
            self.pipelines = {}
            camera_serials = {
                'cam_high': self.args.camera_front_serial,
                'cam_left_wrist': self.args.camera_left_serial, 
                'cam_right_wrist': self.args.camera_right_serial
            }
            
            print("初始化RealSense相机...")
            for cam_name, serial in camera_serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                pipeline.start(config)
                self.pipelines[cam_name] = pipeline
                print(f"[INFO] {cam_name} 相机已启动")
                
            # 预热相机
            for i in range(30):
                for pipeline in self.pipelines.values():
                    pipeline.wait_for_frames(timeout_ms=5000)
            print("[INFO] 相机预热完成")
            
            with self.data_ready_lock:
                self.data_ready['cameras'] = True
                
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            self.pipelines = {}

    def get_camera_images(self):
        """获取摄像头图像"""
        images = {}
        if not hasattr(self, 'pipelines') or not self.pipelines:
            return images
        # 返回上一次成功的图像，避免短暂超时导致采集中断
        fallback = getattr(self, "last_images", {})

        for cam_name, pipeline in self.pipelines.items():
            try:
                # 非阻塞读取，避免硬等待导致日志刷屏
                frames = pipeline.poll_for_frames()
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        image = np.asanyarray(color_frame.get_data())
                        images[cam_name] = image
                        continue
            except Exception as e:
                # 如果本次失败且有旧帧可用，静默回退
                pass

            if cam_name in fallback:
                images[cam_name] = fallback[cam_name]
            else:
                if not hasattr(self, "_last_cam_warn") or (time.time() - getattr(self, "_last_cam_warn", 0) > 2.0):
                    print(f"获取 {cam_name} 图像失败: 无新帧且无缓存")
                    self._last_cam_warn = time.time()

        if images:
            self.last_images = images
        return images

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        """等待所有传感器数据就绪"""
        print("等待传感器数据就绪...")
        start_time = time.time()
        
        while time.time() - start_time < timeout and rclpy.ok():
            with self.data_ready_lock:
                joints_ready = self.data_ready['joint_left'] and self.data_ready['joint_right']
                cameras_ready = self.data_ready['cameras']
            
            if joints_ready and cameras_ready:
                print("[INFO] 所有传感器数据就绪")
                return True
            
            time.sleep(0.5)
        
        print("[ERROR] 等待传感器数据超时")
        return False

    def set_joint_positions(self, pos: np.ndarray):
        """发布 RobotStatus 控制命令"""
        if not rclpy.ok():
            print("[WARN] ROS 已关闭，跳过发布")
            return

        # print("[DEBUG] set_joint_positions 被调用!")
        # print(f"[DEBUG] 发布的控制命令: {pos}")

        if len(pos) != 14:
            self.get_logger().warn(f"期望14维, 实际 {len(pos)}")
            return

        msg_left = RobotStatus()
        msg_right = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos[:7]]
        msg_right.joint_pos = [float(x) for x in pos[7:]]
        
        # 发布控制命令到左臂和右臂的命令话题
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)

        # print(f"[CMD] 发布 RobotStatus 左臂: {msg_left.joint_pos} 右臂: {msg_right.joint_pos}")

    def set_master_mode(self, node_name: str, mode: str, timeout: float = 3.0):
        """切换主臂节点模式，例如 remote_master / remote_master_ctrl"""
        client = self.create_client(SetParameters, f"{node_name}/set_parameters")
        if not client.wait_for_service(timeout_sec=timeout):
            print(f"[WARN] 服务 {node_name}/set_parameters 不可用，跳过切换 {mode}")
            return False
        req = SetParameters.Request()
        pval = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
        req.parameters = [Parameter(name="arm_control_type", value=pval)]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() and future.result().results:
            res = future.result().results[0]
            if res.successful:
                print(f"[INFO] {node_name} 已切到 {mode}")
                return True
            print(f"[WARN] {node_name} 切换 {mode} 失败: {res.reason}")
            return False
        print(f"[WARN] {node_name} 切换 {mode} 超时/无返回")
        return False

    def publish_master_ctrl_positions(self, pos: np.ndarray):
        """通过 master_ctrl 话题发布 14 维目标关节并缓存"""
        if len(pos) != 14:
            self.get_logger().warn(f"[master_ctrl] 期望14维, 实际 {len(pos)}")
            return
        self.master_cmd_cache = np.asarray(pos, dtype=float).copy()
        if self.master_ctrl_left_pub is not None:
            msg_l = RobotStatus()
            msg_l.joint_pos = self.master_cmd_cache[:7].tolist()
            self.master_ctrl_left_pub.publish(msg_l)
        if self.master_ctrl_right_pub is not None:
            msg_r = RobotStatus()
            msg_r.joint_pos = self.master_cmd_cache[7:].tolist()
            self.master_ctrl_right_pub.publish(msg_r)

    def _smooth_interpolate_positions(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        hz: float,
        max_joint_step: Optional[float],
        publish_fn: Callable[[np.ndarray], None],
        progress_label: str = "进度",
    ) -> bool:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if start.shape[0] != 14 or target.shape[0] != 14:
            self.get_logger().warn(f"[smooth] 期望14维, start={start.shape}, target={target.shape}")
            return False

        max_delta = float(np.max(np.abs(target - start)))
        num_steps = max(1, int(duration * hz))
        if max_joint_step is not None and max_joint_step > 0:
            min_steps = int(np.ceil(max_delta / max_joint_step)) if max_delta > 0 else 1
            num_steps = max(num_steps, min_steps)

        prev = start.copy()
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            cmd = start * (1 - smooth_alpha) + target * smooth_alpha

            if max_joint_step is not None and max_joint_step > 0:
                delta = cmd - prev
                # 不限制两侧夹爪（索引6、13），防止张开→闭合幅值被截断
                if np.isscalar(max_joint_step):
                    step_limits = np.full(14, max_joint_step, dtype=float)
                else:
                    step_limits = np.asarray(max_joint_step, dtype=float)
                    if step_limits.shape[0] != 14:
                        self.get_logger().warn(f"[smooth] step_limits 维度异常: {step_limits.shape}")
                        step_limits = np.full(14, float(max_joint_step), dtype=float)
                step_limits[6] = np.inf
                step_limits[13] = np.inf
                delta = np.clip(delta, -step_limits, step_limits)
                cmd = prev + delta

            publish_fn(cmd)
            prev = cmd.copy()

            if step % 50 == 0 or step == num_steps:
                progress = int(alpha * 100)
                print(f"\r{progress_label}: {progress}%", end="", flush=True)
            time.sleep(1.0 / hz)

        print(f"\n[INFO] {progress_label}完成")
        return True

    def smooth_publish_master_ctrl(self, target_pos: np.ndarray, duration: float = 3.0, hz: float = 50.0, max_joint_step: Optional[float] = None) -> bool:
        """
        平滑插值发布到 master_ctrl 话题，避免主臂瞬移（沿用 smooth_goto_position 的余弦插值）
        """
        target = np.asarray(target_pos, dtype=float)
        start = self.get_master_positions()
        if start is None:
            print("[ERROR] 未获取到主臂状态，终止主臂平滑发布")
            return False

        return self._smooth_interpolate_positions(
            start=start,
            target=target,
            duration=duration,
            hz=hz,
            max_joint_step=max_joint_step,
            publish_fn=self.publish_master_ctrl_positions,
            progress_label="主臂对齐进度",
        )

    def move_master_to_slave_pose(self, duration: float = 3.0, hz: float = 50.0, max_joint_step: float = 0.05):
        """将主臂姿态平滑对齐到当前从臂姿态，内部直接用平滑插值"""
        frame = self.get_frame()
        if frame is None:
            print("[WARN] 无从臂数据，跳过主臂对齐")
            return False
        _, j_left, j_right = frame
        target = np.concatenate([j_left.joint_pos, j_right.joint_pos]).astype(float)
        # 直接用余弦插值 + 单步限幅，发布到 master_ctrl 话题
        start = self.get_master_positions()
        if start is None:
            raise RuntimeError("主臂状态不可用，无法对齐到从臂姿态")
        return self._smooth_interpolate_positions(
            start=start,
            target=target,
            duration=duration,
            hz=hz,
            max_joint_step=max_joint_step,
            publish_fn=self.publish_master_ctrl_positions,
            progress_label="主臂对齐进度",
        )

    def sync_master_hold_to_slave(self, repeats: int = 10, hz: float = 50.0):
        """在 remote_master_ctrl 下，推送当前从臂角度到主臂，被控模式保持当前位，不沿用历史指令"""
        frame = self.get_frame()
        if frame is None:
            print("[WARN] 无从臂数据，跳过主臂保持同步")
            return False
        _, j_left, j_right = frame
        target = np.concatenate([j_left.joint_pos, j_right.joint_pos])
        self.master_cmd_cache = target.copy()
        for _ in range(max(1, repeats)):
            if self.master_ctrl_left_pub is not None:
                msg_l = RobotStatus()
                msg_l.joint_pos = target[:7].tolist()
                self.master_ctrl_left_pub.publish(msg_l)
            if self.master_ctrl_right_pub is not None:
                msg_r = RobotStatus()
                msg_r.joint_pos = target[7:].tolist()
                self.master_ctrl_right_pub.publish(msg_r)
            time.sleep(1.0 / hz)
        print("[INFO] 主臂保持指令已刷新为当前从臂姿态")
        return True

    def hold_master_current_pose(self, repeats: int = 20, hz: float = 50.0):
        """用当前主臂状态刷新被控指令，防止沿用旧指令"""
        cur = self.get_master_positions()
        if cur is None:
            print("[WARN] 未收到主臂状态，跳过保持当前姿态")
            return False
        self.master_cmd_cache = cur.copy()
        for _ in range(max(1, repeats)):
            if self.master_ctrl_left_pub is not None:
                msg_l = RobotStatus()
                msg_l.joint_pos = cur[:7].tolist()
                self.master_ctrl_left_pub.publish(msg_l)
            if self.master_ctrl_right_pub is not None:
                msg_r = RobotStatus()
                msg_r.joint_pos = cur[7:].tolist()
                self.master_ctrl_right_pub.publish(msg_r)
            time.sleep(1.0 / hz)
        print("[INFO] 主臂保持指令已刷新为当前主臂姿态")
        return True


    def smooth_return_to_zero(self, duration: float = 3.0):
        """平滑回零功能"""
        print("正在平滑回零...")
        
        frame = self.get_frame()
        if frame is None:
            print("无法获取当前关节位置")
            return False
            
        _, j_left, j_right = frame
        current_pos = self.get_joint_positions(j_left, j_right)
        target_pos = np.zeros(14)
        target_pos[6] = 3.0
        target_pos[13] = 3.0

        
        control_hz = 50.0
        num_steps = int(duration * control_hz)
        
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            
            interpolated_pos = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
            self.set_joint_positions(interpolated_pos)
            
            progress = int(alpha * 100)
            print(f"\r回零进度: {progress}%", end='', flush=True)
            time.sleep(1.0/control_hz)
        
        print("\n[INFO] 回零完成")
        
        # === 自动张开夹爪 ===
        open_pos = np.zeros(14)
        open_pos[6] = 5.0
        open_pos[13] = 5.0
        self.set_joint_positions(open_pos)
        # print("已自动张开夹爪")
        
        return True


    def exit_return_to_zero(self, duration: float = 3.0):
        """平滑回零（不依赖相机，仅用缓存）"""
        with self.qpos_lock:
            if self.last_qpos is None:
                print("[ERROR] 无缓存关节角，跳过回零")
                return False
            current_pos = self.last_qpos.copy()

        target_pos = np.zeros(14)
        control_hz = 50.0
        num_steps = int(duration * control_hz)

        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interp = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
            self.set_joint_positions(interp)

            progress = int(alpha * 100)
            print(f"\r回零进度: {progress}%", end='', flush=True)
            time.sleep(1.0 / control_hz)

        print("\n[INFO] 回零完成")

        # === 自动张开夹爪 ===
        open_pos = np.zeros(14)
        open_pos[6] = 5.0
        open_pos[13] = 5.0
        self.set_joint_positions(open_pos)
        # print("🟢 已自动张开夹爪")

        return True



    # def smooth_return_to_zero(self, duration: float = 3.0):
    #     """平滑回零功能"""
    #     print("正在平滑回零...")
        
    #     frame = self.get_frame()
    #     if frame is None:
    #         print("无法获取当前关节位置")
    #         return False
            
    #     _, j_left, j_right = frame
    #     current_pos = self.get_joint_positions(j_left, j_right)
    #     target_pos = np.zeros(14)
        
    #     control_hz = 50.0
    #     num_steps = int(duration * control_hz)
        
    #     for step in range(num_steps + 1):
    #         alpha = step / num_steps
    #         smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            
    #         interpolated_pos = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
    #         self.set_joint_positions(interpolated_pos)
            
    #         progress = int(alpha * 100)
    #         print(f"\r回零进度: {progress}%", end='', flush=True)
    #         time.sleep(1.0/control_hz)
        
    #     print("\n✓ 回零完成!")
    #     return True
    
    # def exit_return_to_zero(self, duration: float = 3.0):
    #     """平滑回零（不依赖相机，仅用缓存）"""
    #     with self.qpos_lock:
    #         if self.last_qpos is None:
    #             print("❌ 无缓存关节角，跳过回零")
    #             return False
    #         current_pos = self.last_qpos.copy()

    #     target_pos = np.zeros(14)
    #     control_hz = 50.0
    #     num_steps = int(duration * control_hz)

    #     for step in range(num_steps + 1):
    #         alpha = step / num_steps
    #         smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
    #         interp = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
    #         self.set_joint_positions(interp)

    #         progress = int(alpha * 100)
    #         print(f"\r回零进度: {progress}%", end='', flush=True)
    #         time.sleep(1.0 / control_hz)

    #     print("\n✓ 回零完成!")
    #     return True
    
    def smooth_goto_position(self, target_pos: np.ndarray, duration: float = 3.0, hz: float = 50.0) -> bool:
        """
        平滑插值到任意 14 维目标位（实时读角，无缓存滞后）
        """
        frame = self.get_frame()
        if frame is None:
            print("[ERROR] 无法获取当前关节角，跳过平滑移动")
            return False

        _, j_left, j_right = frame
        current_pos = self.get_joint_positions(j_left, j_right)   # 实时角
        max_delta = np.max(np.abs(target_pos - current_pos))

        num_steps = int(duration * hz)
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interp = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
            self.set_joint_positions(interp)

            if step % 50 == 0 or step == num_steps:
                progress = int(alpha * 100)
                print(f"\r平滑移动进度: {progress}%", end='', flush=True)
            time.sleep(1.0 / hz)

        print("\n[INFO] 平滑移动完成")
        return True

    def cleanup_cameras(self):
        """释放相机资源"""
        if hasattr(self, 'pipelines'):
            print("释放RealSense相机资源...")
            for cam_name, pipeline in self.pipelines.items():
                try:
                    pipeline.stop()
                except Exception:
                    pass



def get_config(args):
    """获取配置"""
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


def model_inference(args, config, ros_operator):
    """主推理循环"""
    global stream_buffer, lang_embeddings, dagger_mode_active, save_data_requested, collection_active

    # 加载WebSocket客户端
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    max_publish_step = config["episode_len"]


    left0  = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    right0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    
    frame = ros_operator.get_frame()
    if frame is None:
        print("[ERROR] 无法获取当前关节角，跳过初始插值")
    else:
        _, j_left, j_right = frame
        current_q = ros_operator.get_joint_positions(j_left, j_right)
        target_q = np.array(left0 + right0)


        # 用通用平滑函数去初始位
        ros_operator.smooth_goto_position(
            target_pos=np.array(left0 + right0),
            duration=3.0,   # 可命令行调
            hz=50.0        # 已 200 Hz
        )
        print("[INFO] 初始姿态平滑完成")


    # 预热推理
    try:
        print("[INFO] 预热推理...")
        update_observation_window(args, config, ros_operator)
        print("[INFO] 观测窗口已更新")
        if len(observation_window) == 0:
            print("[ERROR] 观测窗口仍为空，跳过预热")
        else:
            print("[INFO] Warming up policy.infer()...")
            latest_obs = observation_window[-1]
            image_arrs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
            image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)
            proprio = latest_obs["qpos"]
            payload = {
                "state": proprio,
                "images": {
                    "top_head":  image_arrs[0].transpose(2, 0, 1),
                    "hand_right": image_arrs[1].transpose(2, 0, 1),
                    "hand_left":  image_arrs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }
            _ = policy.infer(payload)["actions"]
            print("[INFO] Warmup done")
    except Exception as e:
        print(f"[ERROR] Warmup failed: {e}")
        import traceback
        traceback.print_exc()

    stream_buffer = StreamActionBuffer(
        max_chunks=args.buffer_max_chunks,
        decay_alpha=args.exp_decay_alpha,
        state_dim=config["state_dim"],
        smooth_method="temporal",
    )
    
    # 启动推理线程
    inference_thread = start_inference_thread(args, config, policy, ros_operator)

    # 主控制循环
    # rate = rclpy.create_rate(args.control_frequency)
    rate = ros_operator.create_rate(args.control_frequency)
    step = 0
    consecutive_empty_actions = 0
    max_empty_actions = 100
    dagger_mode_entered = False
    keyboard_thread_started = False
    
    print("Starting inference control loop...")
    
    # 全程采集：推理开始时自动启动数据采集
    ros_operator.data_collector.start_collection()
    
    try:
        if not keyboard_thread_started:
            threading.Thread(target=keyboard_monitor_thread, daemon=True).start()
            keyboard_thread_started = True
            print("\n" + "=" * 60)
            print("[INFO] 推理运行中，数据采集已启动")
            print("[HINT] Press [d] for DAgger mode | [s] to save")
            print("=" * 60 + "\n")
        while rclpy.ok() and step < max_publish_step and not shutdown_event.is_set():
            # 切换提示
            with dagger_mode_lock:
                if dagger_mode_active and not dagger_mode_entered:
                    print("\n[DAGGER] 正在对齐主臂到从臂姿态...")
                    # 暂停数据采集，避免对齐期间的脏数据
                    ros_operator.data_collector.pause()
                    if stream_buffer:
                        stream_buffer.clear()
                    # 切换前读取主臂姿态并推送缓存，防止瞬移
                    master_hold = ros_operator.get_master_positions()
                    ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
                    ros_operator.set_master_mode(args.master_right_node, "remote_master_ctrl")
                    if master_hold is not None:
                        try:
                            ros_operator.publish_master_ctrl_positions(master_hold)
                            ros_operator.hold_master_current_pose(repeats=50, hz=100.0)
                        except Exception:
                            pass
                    ros_operator.move_master_to_slave_pose(
                        duration=3.0,
                        hz=100.0,
                        max_joint_step=getattr(args, "master_align_max_step", 0.01),
                    )
                    ros_operator.sync_master_hold_to_slave(repeats=50, hz=100.0)
                    ros_operator.set_master_mode(args.master_left_node, "remote_master")
                    ros_operator.set_master_mode(args.master_right_node, "remote_master")
                    # Alignment done; resume collection
                    ros_operator.data_collector.resume()
                    dagger_mode_entered = True
                    _print_dagger_status(
                        "Alignment done; in DAgger mode (human control).",
                        "Press [r] to resume inference | [s] to save episode"
                    )
                elif (not dagger_mode_active) and dagger_mode_entered:
                    # 全程采集模式：不停止采集，只是 intervention 从 1 变成 0
                    # 切回 ctrl 前先抓取姿态，优先用主臂自身状态，防止拉回到从臂姿态
                    master_hold = ros_operator.get_master_positions()
                    if master_hold is None:
                        frame = ros_operator.get_frame()
                        if frame is not None:
                            _, j_left, j_right = frame
                            master_hold = np.concatenate([j_left.joint_pos, j_right.joint_pos]).astype(float)
                    ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
                    ros_operator.set_master_mode(args.master_right_node, "remote_master_ctrl")
                    # 清空推理缓冲，避免复用旧动作导致跳变
                    if stream_buffer:
                        stream_buffer.clear()
                    
                    # 主臂归位放到后台线程，不阻塞推理
                    def _master_return_home():
                        try:
                            if master_hold is not None:
                                ros_operator.publish_master_ctrl_positions(master_hold)
                            ros_operator.hold_master_current_pose(repeats=100, hz=100.0)
                            master_init = np.array(left0 + right0, dtype=float)
                            ros_operator.smooth_publish_master_ctrl(
                                master_init,
                                duration=getattr(args, "master_init_duration", 2.0),
                                hz=100.0,
                                max_joint_step=getattr(args, "master_align_max_step", 0.01),
                            )
                        except Exception as e:
                            print(f"[WARN] 主臂归位失败: {e}")
                    threading.Thread(target=_master_return_home, daemon=True).start()
                    
                    dagger_mode_entered = False
                    _print_dagger_status(
                        "已恢复推理（主臂后台归位中）",
                        "Press [d] for DAgger | [s] to save episode"
                    )

            # 保存请求（任何时候都可以保存，异步执行不阻塞）
            with save_data_lock:
                if save_data_requested:
                    ros_operator.data_collector.save_current_episode(
                        export_video=args.save_video,
                        video_fps=getattr(args, "video_fps", 30),
                    )
                    # 保存是异步的，后台线程会自动重置数据并继续采集
                    save_data_requested = False

            # DAgger 模式：采集人工数据（intervention=1），不发推理控制
            if dagger_mode_active:
                frame = ros_operator.get_frame()
                if frame is None:
                    rate.sleep()
                    continue
                imgs, j_left, j_right = frame
                obs = {
                    "qpos": np.concatenate([j_left.joint_pos, j_right.joint_pos]),
                    "qvel": np.concatenate([j_left.joint_vel, j_right.joint_vel]),
                    "effort": np.concatenate([j_left.joint_cur, j_right.joint_cur]),
                    "images": imgs,
                }
                action = obs["qpos"].copy()
                if ros_operator.data_collector.is_collecting:
                    ros_operator.data_collector.add_frame(obs, action, intervention=1)
                rate.sleep()
                continue

            frame = ros_operator.get_frame()
            if frame is None:
                print("[DEBUG] 无法获取传感器数据，跳过此轮控制")
                rate.sleep()
                continue

            imgs, j_left, j_right = frame
            qpos = ros_operator.get_joint_positions(j_left, j_right)
            observed_qpos_history.append(qpos.copy())

            act = stream_buffer.pop_next_action()
            # import ipdb; ipdb.set_trace()
            if act is not None:
                consecutive_empty_actions = 0
                # print(f"[DEBUG] 获取到的动作: {act}")
                
                if args.use_eef_correction:
                    act = apply_eef_correction(act, qpos, args)

                act = apply_gripper_binary(act)

                ros_operator.set_joint_positions(act)
                published_actions_history.append(act.copy())
                
                # 推理模式下继续采集数据（intervention=0，模型执行）
                if ros_operator.data_collector.is_collecting:
                    obs = {
                        "qpos": np.concatenate([j_left.joint_pos, j_right.joint_pos]),
                        "qvel": np.concatenate([j_left.joint_vel, j_right.joint_vel]),
                        "effort": np.concatenate([j_left.joint_cur, j_right.joint_cur]),
                        "images": imgs,
                    }
                    ros_operator.data_collector.add_frame(obs, act, intervention=0)

                step += 1
            else:
                consecutive_empty_actions += 1
                if consecutive_empty_actions >= max_empty_actions:
                    print(f"[main] 连续 {consecutive_empty_actions} 次无动作, 执行安全回零")
                    ros_operator.smooth_return_to_zero(duration=3.0)
                    consecutive_empty_actions = 0

            rate.sleep()
                
    except Exception as e:
        print(f"[main] 主循环异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_event.set()
        if inference_thread.is_alive():
            inference_thread.join(timeout=2.0)
        # print("执行安全回零...")
        # ros_operator.smooth_return_to_zero(duration=3.0)
        ros_operator.data_collector.shutdown()  # 关闭后台采集线程
        ros_operator.cleanup_cameras()
        print("[INFO] ARX5双臂推理控制器已安全关闭")

        return inference_thread




def apply_eef_correction(act: np.ndarray, qpos: np.ndarray, args) -> np.ndarray:
    """应用末端微调"""
    left0, right0 = qpos[:6], qpos[7:13]
    dl = np.array(args.eef_corr_left)
    dr = np.array(args.eef_corr_right)
    
    left6 = apply_micro_correction(left0, dl, "base",
                                   args.eef_lambda,
                                   args.eef_step_limit_m,
                                   args.eef_joint_step_limit)
    right6 = apply_micro_correction(right0, dr, "base",
                                    args.eef_lambda,
                                    args.eef_step_limit_m,
                                    args.eef_joint_step_limit)
    
    act2 = act.copy()
    act2[:6], act2[7:13] = left6, right6
    return act2

def apply_gripper_binary(act: np.ndarray, close_val: float = 0.0, open_val: float = 5.0, thresh: float = 2.5) -> np.ndarray:
    """应用夹爪二分阈值"""
    act2 = act.copy()
    act2[6] = open_val if act[6] >= thresh else close_val
    act2[13] = open_val if act[13] >= thresh else close_val
    return act2


def main():

    # 处理 Ctrl+C 信号
    def _on_sigint(sig, frame):
        print("接收到终止信号，准备关闭程序...")
        shutdown_event.set()          # 通知所有线程
        sys.exit(0)                   # 强制结束进程

    parser = argparse.ArgumentParser(description="ARX-X5 双臂推理控制器（重构版）")
    
    # =================== 基础话题 ===================
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--joint_cmd_topic_right", default="/arm_master_r_status")
    parser.add_argument("--joint_state_topic_right", default="/arm_slave_r_status")
    parser.add_argument("--master_status_topic_left", default="/arm_master_l_status", help="左主臂状态话题（remote_master 模式）")
    parser.add_argument("--master_status_topic_right", default="/arm_master_r_status", help="右主臂状态话题（remote_master 模式）")
    parser.add_argument("--master_status_topic_left_ctrl", default="/arm_master_ctrl_status_left", help="左主臂状态话题（remote_master_ctrl 模式）")
    parser.add_argument("--master_status_topic_right_ctrl", default="/arm_master_ctrl_status_right", help="右主臂状态话题（remote_master_ctrl 模式）")

    # =================== 摄像头序列号参数 ===================
    parser.add_argument("--camera_front_serial", type=str, default='213722070209')
    parser.add_argument("--camera_left_serial", type=str, default='213722070377')
    parser.add_argument("--camera_right_serial", type=str, default='213522071788')

    # =================== 推理参数 ===================
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host IP")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)

    # =================== 平滑参数 ===================
    parser.add_argument("--use_temporal_smoothing", action="store_true", default=True)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=10)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)

    # =================== 二分阈值参数 ===================
    parser.add_argument("--gripper_open",  type=float, default=0.0,   help="夹爪打开位置 (rad)")
    parser.add_argument("--gripper_close", type=float, default=-0.8,  help="夹爪闭合位置 (rad)")
    parser.add_argument("--gripper_thresh",type=float, default=0.5,   help="推理值≥该阈值时视为闭合")

    # =================== 末端微调 ===================
    parser.add_argument("--use_eef_correction", action="store_true")
    parser.add_argument("--eef_corr_left", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_corr_right", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_lambda", type=float, default=0.001)
    parser.add_argument("--eef_step_limit_m", type=float, default=0.01)
    parser.add_argument("--eef_joint_step_limit", nargs=6, type=float, default=[0.1]*6)

    parser.add_argument("--auto_homing", action="store_true", default=True, help="Return to initial pose on startup")
    parser.add_argument("--exit_homing", action="store_true", help="Return to initial pose on exit")
    parser.add_argument("--dataset_dir", type=str, default=os.path.expanduser("~/data/dagger"), help="DAgger dataset root")
    parser.add_argument("--dataset_name", type=str, default="aloha_mobile_dummy", help="DAgger dataset name")
    parser.add_argument("--save_video", action="store_true", default=True, help="Save camera videos")
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--master_align_max_step", type=float, default=0.05, help="Max joint step for master alignment (rad)")
    parser.add_argument("--debug_timing", action="store_true", help="Print inference/obs timing")
    parser.add_argument("--master_left_node", type=str, default="/arm_master_l", help="Left master node name")
    parser.add_argument("--master_right_node", type=str, default="/arm_master_r", help="Right master node name")
    parser.add_argument(
        "--master_ctrl_cmd_topic_left",
        type=str,
        default="/arm_master_ctrl_cmd_left",
        help="Left master control command topic",
    )
    parser.add_argument(
        "--master_ctrl_cmd_topic_right",
        type=str,
        default="/arm_master_ctrl_cmd_right",
        help="Right master control command topic",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        rclpy.init()
        ros_operator = ARX5ROSController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
        spin_thread.start()
        print("[INFO] ROS spin thread started")
        # Switch masters to remote_master_ctrl
        ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
        ros_operator.set_master_mode(args.master_right_node, "remote_master_ctrl")

        if not ros_operator.wait_for_data_ready(timeout=15.0):
            print("[ERROR] Sensor data not ready; exiting.")
            return
        try:
            frame = ros_operator.get_frame()
            if frame is not None:
                _, j_left, j_right = frame
                ros_operator.master_cmd_cache = np.concatenate([j_left.joint_pos, j_right.joint_pos])
                print("[INFO] Master cache initialized from slave pose")
        except Exception:
            pass
        if args.auto_homing:
            print("[INFO] Homing master and slave arms...")
            left0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
            right0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
            init_pos = np.array(left0 + right0)
            
            def _master_homing():
                ros_operator.smooth_publish_master_ctrl(
                    target_pos=init_pos,
                    duration=3.0,
                    hz=50.0,
                    max_joint_step=0.05,
                )
            
            def _slave_homing():
                ros_operator.smooth_goto_position(target_pos=init_pos, duration=3.0, hz=50.0)
            
            master_thread = threading.Thread(target=_master_homing)
            slave_thread = threading.Thread(target=_slave_homing)
            master_thread.start()
            slave_thread.start()
            master_thread.join()
            slave_thread.join()
            print("[INFO] Homing done")
            time.sleep(0.5)
        print("Press Enter when ready...")
        input("Arms ready. Press Enter to start inference...")
        config = get_config(args)
        inference_thread = model_inference(args, config, ros_operator)
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            shutdown_event.set()
            if rclpy.ok():
                rclpy.shutdown()
            print("[INFO] ROS2 shutdown")
        except Exception as e:
            print(f"ROS2 shutdown error: {e}")
        print("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
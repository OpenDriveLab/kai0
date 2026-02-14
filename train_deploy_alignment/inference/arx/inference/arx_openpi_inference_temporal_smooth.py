#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 dual-arm inference with temporal smoothing (no RTC).

Async inference thread pushes chunks to a stream buffer; main loop consumes with
temporal smoothing over chunk boundaries. Same design as Agilex temporal_smoothing.
Set lang_embeddings at top to match training. See train_deploy_alignment/inference/arx/README.md.
"""

import argparse
import time
import threading
import json
import numpy as np
import cv2
import os
import signal
import sys
from collections import deque
from typing import Dict, Any, Optional, List
from sensor_msgs.msg import JointState

try:
    import pyrealsense2 as rs
except ImportError:
    print("Warning: pyrealsense2 not installed; camera features unavailable.")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

try:
    from arx5_arm_msg.msg import RobotStatus, RobotCmd
    print("Loaded arx5_arm_msg (RobotStatus, RobotCmd)")
except ImportError:
    print("arx5_arm_msg not found; ensure ARX5 message package is installed.")
    from sensor_msgs.msg import JointState
    RobotStatus = JointState
    RobotCmd = JointState
    print("Using JointState as fallback for RobotCmd.")

from openpi_client import image_tools, websocket_client_policy

CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
stream_buffer = None
observation_window = deque(maxlen=2)
lang_embeddings = "hang the cloth"
published_actions_history = []
observed_qpos_history = []
inferred_chunks = []
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()



class StreamActionBuffer:
    """Chunk queue for actions with latency trim and temporal smoothing (same design as Agilex)."""
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


def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """Non-blocking inference thread; pushes chunks to stream_buffer with temporal smoothing."""
    global stream_buffer, observation_window, lang_embeddings
    rate = ros_operator.create_rate(args.inference_rate)
    consecutive_failures = 0
    max_consecutive_failures = 5

    while rclpy.ok() and not shutdown_event.is_set():
        try:
            time1 = time.time()
            update_observation_window(args, config, ros_operator)
            print("Get Observation Time", time.time() - time1, "s")
            
            if len(observation_window) == 0:
                continue
                
            latest_obs = observation_window[-1]
            imgs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            proprio = latest_obs["qpos"]
            payload = {
                "state": proprio,
                "images": {
                    "top_head":  imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left":  imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }
            time1 = time.time()
            actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - time1, "s")
            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
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
            if consecutive_failures >= max_consecutive_failures:
                print(f"[inference] Consecutive failures {consecutive_failures}, clearing buffer")
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
    """Update observation window from ROS/sensors."""
    global observation_window
    if len(observation_window) == 0:
        observation_window.append({
            "qpos": None,
            "images": {
                config["camera_names"][0]: None,
                config["camera_names"][1]: None,
                config["camera_names"][2]: None,
            },
        })
    frame = ros_operator.get_frame()
    if frame is None:
        return
        
    imgs, j_left, j_right = frame
    qpos = ros_operator.get_joint_positions(j_left, j_right)
    
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
    try:
        observed_qpos_history.append(np.asarray(qpos, dtype=float).copy())
    except Exception:
        pass


class ARX5ROSController(Node):
    try:
        from arx5_arm_msg.msg import RobotStatus, RobotCmd
        print("Loaded arx5_arm_msg (RobotStatus, RobotCmd)")
    except ImportError:
        print("arx5_arm_msg not found; install ARX5 message package.")
        from sensor_msgs.msg import JointState
        RobotStatus = JointState
        RobotCmd = JointState

    def __init__(self, args):
        super().__init__("arx5_controller")
        self.args = args
        self.bridge = CvBridge()
        self.last_qpos = None
        self.qpos_lock = threading.Lock()
        self.joint_left_deque = deque(maxlen=2000)
        self.joint_right_deque = deque(maxlen=2000)
        self.pub_left = self.create_publisher(RobotStatus, args.joint_cmd_topic_left, 10)
        self.pub_right = self.create_publisher(RobotStatus, args.joint_cmd_topic_right, 10)
        self.RobotStatus = RobotStatus
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
        self.get_logger().info(f"Subscribed to left arm: {args.joint_state_topic_left}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_left,
            self.joint_left_callback,
            10
        )
        self.get_logger().info(f"Subscribed to right arm: {args.joint_state_topic_right}")
        self.create_subscription(
            RobotStatus,
            args.joint_state_topic_right,
            self.joint_right_callback,
            10
        )
        self.data_ready = {
            'joint_left': False,
            'joint_right': False,
            'cameras': False
        }
        self.data_ready_lock = threading.Lock()
        self.init_cameras()

    def joint_left_callback(self, msg):
        """Left arm RobotStatus callback."""
        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_left'] = True

    def joint_right_callback(self, msg):
        """Right arm RobotStatus callback."""
        self.joint_right_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready['joint_right'] = True

    def left_arm_command_callback(self, msg):
        """Left arm command: merge with current right arm position and publish."""
        self.set_joint_positions(np.array(msg.joint_pos + self.joint_right_deque[-1].joint_pos))

    def right_arm_command_callback(self, msg):
        """Right arm command: merge with current left arm position and publish."""
        self.set_joint_positions(np.array(self.joint_left_deque[-1].joint_pos + msg.joint_pos))

    def get_joint_positions(self, j_left: RobotStatus, j_right: RobotStatus) -> np.ndarray:
        """Get 14-D joint positions from RobotStatus messages."""
        left = list(j_left.joint_pos)
        right = list(j_right.joint_pos)
        q = np.array(left + right, dtype=float)
        with self.qpos_lock:
            self.last_qpos = q.copy()
        return q

    def get_frame(self):
        """Get synchronized sensor data (joints + camera images)."""
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return None
        j_left = self.joint_left_deque[-1]
        j_right = self.joint_right_deque[-1]
        imgs = self.get_camera_images()
        if len(imgs) != 3:
            return None
        return imgs, j_left, j_right

    def init_cameras(self):
        """Initialize RealSense cameras."""
        try:
            import pyrealsense2 as rs
            self.pipelines = {}
            camera_serials = {
                'cam_high': self.args.camera_front_serial,
                'cam_left_wrist': self.args.camera_left_serial,
                'cam_right_wrist': self.args.camera_right_serial
            }
            print("Initializing RealSense cameras...")
            for cam_name, serial in camera_serials.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                pipeline.start(config)
                self.pipelines[cam_name] = pipeline
                print(f"  {cam_name} started")
            for i in range(30):
                for pipeline in self.pipelines.values():
                    pipeline.wait_for_frames(timeout_ms=5000)
            print("Cameras warmed up.")
            with self.data_ready_lock:
                self.data_ready['cameras'] = True
        except Exception as e:
            print(f"Camera init failed: {e}")
            self.pipelines = {}

    def get_camera_images(self):
        """Get images from all cameras."""
        images = {}
        if not hasattr(self, 'pipelines') or not self.pipelines:
            return images
        for cam_name, pipeline in self.pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    image = np.asanyarray(color_frame.get_data())
                    images[cam_name] = image
            except Exception as e:
                print(f"Failed to get image from {cam_name}: {e}")
        return images

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        """Wait until all sensor data is ready."""
        print("Waiting for sensor data...")
        start_time = time.time()
        while time.time() - start_time < timeout and rclpy.ok():
            with self.data_ready_lock:
                joints_ready = self.data_ready['joint_left'] and self.data_ready['joint_right']
                cameras_ready = self.data_ready['cameras']
            if joints_ready and cameras_ready:
                print("All sensor data ready.")
                return True
            time.sleep(0.5)
        print("Timeout waiting for sensor data.")
        return False

    def set_joint_positions(self, pos: np.ndarray):
        """Publish RobotStatus control commands."""
        if not rclpy.ok():
            return
        if len(pos) != 14:
            self.get_logger().warn(f"Expected 14-D, got {len(pos)}")
            return
        msg_left = RobotStatus()
        msg_right = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos[:7]]
        msg_right.joint_pos = [float(x) for x in pos[7:]]
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)

    def smooth_return_to_zero(self, duration: float = 3.0):
        """Smooth return to zero pose."""
        print("Returning to zero...")
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current joint position")
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
            print(f"\rReturn to zero: {progress}%", end='', flush=True)
            time.sleep(1.0/control_hz)
        print("\nReturn to zero done.")
        open_pos = np.zeros(14)
        open_pos[6] = 5.0
        open_pos[13] = 5.0
        self.set_joint_positions(open_pos)
        return True

    def exit_return_to_zero(self, duration: float = 3.0):
        """Smooth return to zero using cached qpos (no camera)."""
        with self.qpos_lock:
            if self.last_qpos is None:
                print("No cached qpos; skip return to zero.")
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
            print(f"\rReturn to zero: {progress}%", end='', flush=True)
            time.sleep(1.0 / control_hz)
        print("\nReturn to zero done.")
        open_pos = np.zeros(14)
        open_pos[6] = 5.0
        open_pos[13] = 5.0
        self.set_joint_positions(open_pos)
        return True

    def smooth_goto_position(self, target_pos: np.ndarray, duration: float = 3.0, hz: float = 50.0) -> bool:
        """Smooth interpolate to 14-D target position (read current joints from sensors)."""
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current joints; skip smooth move.")
            return False

        _, j_left, j_right = frame
        current_pos = self.get_joint_positions(j_left, j_right)
        max_delta = np.max(np.abs(target_pos - current_pos))
        print(f"Max joint delta: {max_delta:.4f} rad")

        num_steps = int(duration * hz)
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interp = current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha
            self.set_joint_positions(interp)

            if step % 50 == 0 or step == num_steps:
                progress = int(alpha * 100)
                print(f"\rSmooth move: {progress}%", end='', flush=True)
            time.sleep(1.0 / hz)
        print("\nSmooth move done.")
        return True

    def cleanup_cameras(self):
        """Release camera resources."""
        if hasattr(self, 'pipelines'):
            print("Stopping RealSense cameras...")
            for cam_name, pipeline in self.pipelines.items():
                try:
                    pipeline.stop()
                except Exception:
                    pass



def get_config(args):
    """Build config dict from args."""
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


def model_inference(args, config, ros_operator):
    """Main inference loop: connect policy, run inference thread, consume stream_buffer."""
    global stream_buffer, lang_embeddings
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]


    left0  = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    right0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
    
    frame = ros_operator.get_frame()
    if frame is None:
        print("Cannot get current joints; skip initial move.")
    else:
        _, j_left, j_right = frame
        current_q = ros_operator.get_joint_positions(j_left, j_right)
        target_q = np.array(left0 + right0)
        ros_operator.smooth_goto_position(
            target_pos=np.array(left0 + right0),
            duration=3.0,
            hz=50.0
        )
        print("Initial pose move done.")

    try:
        print("Warming up inference...")
        update_observation_window(args, config, ros_operator)
        if len(observation_window) == 0:
            print("Observation window empty; skip warmup.")
        else:
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
            print("Warmup done.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()

    stream_buffer = StreamActionBuffer(
        max_chunks=args.buffer_max_chunks,
        decay_alpha=args.exp_decay_alpha,
        state_dim=config["state_dim"],
        smooth_method="temporal",
    )
    
    inference_thread = start_inference_thread(args, config, policy, ros_operator)
    rate = ros_operator.create_rate(args.control_frequency)
    step = 0
    consecutive_empty_actions = 0
    max_empty_actions = 100
    print("Starting control loop...")
    try:
        while rclpy.ok() and step < max_publish_step and not shutdown_event.is_set():
            frame = ros_operator.get_frame()
            if frame is None:
                rate.sleep()
                continue

            imgs, j_left, j_right = frame
            qpos = ros_operator.get_joint_positions(j_left, j_right)
            observed_qpos_history.append(qpos.copy())

            act = stream_buffer.pop_next_action()
            # import ipdb; ipdb.set_trace()
            if act is not None:
                consecutive_empty_actions = 0
                if args.use_eef_correction:
                    act = apply_eef_correction(act, qpos, args)

                act = apply_gripper_binary(act)

                ros_operator.set_joint_positions(act)
                published_actions_history.append(act.copy())

                step += 1
                if step % 50 == 0:
                    print(f"[main] step {step}, buffer size: {len(stream_buffer.cur_chunk)}")
            else:
                consecutive_empty_actions += 1
                if consecutive_empty_actions >= max_empty_actions:
                    print(f"[main] No actions for {consecutive_empty_actions} steps; safe return to zero")
                    ros_operator.smooth_return_to_zero(duration=3.0)
                    consecutive_empty_actions = 0

            rate.sleep()
                
    except Exception as e:
        print(f"[main] Loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_event.set()
        if inference_thread.is_alive():
            inference_thread.join(timeout=2.0)
        ros_operator.cleanup_cameras()
        print("ARX5 inference controller shut down.")

        return inference_threadF




def apply_eef_correction(act: np.ndarray, qpos: np.ndarray, args) -> np.ndarray:
    """Apply end-effector micro correction."""
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
    """Apply gripper binary threshold (open/close)."""
    act2 = act.copy()
    act2[6] = open_val if act[6] >= thresh else close_val
    act2[13] = open_val if act[13] >= thresh else close_val
    return act2


def main():

    def _on_sigint(sig, frame):
        shutdown_event.set()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ARX-X5 dual-arm inference (temporal smoothing).")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--joint_cmd_topic_right", default="/arm_master_r_status")
    parser.add_argument("--joint_state_topic_right", default="/arm_slave_r_status")

    parser.add_argument("--camera_front_serial", type=str, default='213722070209')
    parser.add_argument("--camera_left_serial", type=str, default='213722070377')
    parser.add_argument("--camera_right_serial", type=str, default='213522071788')

    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)

    parser.add_argument("--use_temporal_smoothing", action="store_true", default=True)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--min_smooth_steps", type=int, default=10)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)

    parser.add_argument("--gripper_open",  type=float, default=0.0,   help="Gripper open position (rad).")
    parser.add_argument("--gripper_close", type=float, default=-0.8,  help="Gripper close position (rad).")
    parser.add_argument("--gripper_thresh",type=float, default=0.5,   help="Inference value >= thresh -> close.")
    parser.add_argument("--use_eef_correction", action="store_true")
    parser.add_argument("--eef_corr_left", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_corr_right", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--eef_lambda", type=float, default=0.001)
    parser.add_argument("--eef_step_limit_m", type=float, default=0.01)
    parser.add_argument("--eef_joint_step_limit", nargs=6, type=float, default=[0.1]*6)

    parser.add_argument("--auto_homing", action="store_true", default=True, help="Return to zero on startup.")
    parser.add_argument("--exit_homing", action="store_true", help="Return to zero on exit.")

    args = parser.parse_args()
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        rclpy.init()
        ros_operator = ARX5ROSController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
        spin_thread.start()
        print("ROS spin thread started.")

        if not ros_operator.wait_for_data_ready(timeout=15.0):
            print("Sensor data not ready; exiting.")
            return

        if args.auto_homing:
            print("Auto homing...")
            ros_operator.smooth_return_to_zero(duration=3.0)
            time.sleep(1.0)

        print("Press Enter to start inference...")
        input("Arms ready. Press Enter to start...")

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
            print("ROS2 shutdown.")
        except Exception as e:
            print(f"ROS2 shutdown error: {e}")
        print("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()

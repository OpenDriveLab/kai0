# -- coding: UTF-8
"""
Agilex inference with OpenPi RTC (real-time control) and temporal smoothing.

- RTC: inference thread sends prev_action_chunk and inference_delay to the policy server;
  the server runs a model with Pi0RTCConfig (pi0_rtc) which uses guidance to align the
  new chunk with the previously executed prefix.
- Temporal smoothing: StreamActionBuffer.integrate_new_chunk blends overlapping chunks
  (linear 100% old -> 0% new over the overlap) to avoid discontinuities at chunk boundaries.

Start the policy server with an RTC config (e.g. pi05_rtc_flatten_fold_inference) and this script with
--rtc_mode. See train_deploy_alignment/inference/agilex/README.md for setup.
"""
import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from openpi_client import image_tools, websocket_client_policy
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import os
import signal
import sys
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]

stream_buffer = None   # type: StreamActionBuffer

observation_window = None
rtc_prev_chunk_lock = threading.Lock()
rtc_prev_chunk = None

# helper to launch background threads
def start_inference_thread(target, args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t

# lang_embeddings = "fold the cloth"
lang_embeddings = "fold the sleeve"

RIGHT_OFFSET = 0.003
delay_buffer = deque(maxlen=20)  # RTT sliding window (seconds)
pred_delay_steps = 0  # Predicted delay in steps (at publish rate)

# Optional: observation_lock, action_lock for thread safety
published_actions_history = []
observed_qpos_history = []
publish_step_global = 0
inferred_chunks = []            # list[dict(start_step:int, chunk:np.ndarray[chunk,14])]
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()




def joint_actions_clip(action: np.ndarray):
    pass

def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """
    Non-blocking inference: pack latest observation, call infer, push chunk to stream_buffer;
    """
    global stream_buffer
    # assert stream_buffer is not None

    global observation_window

    global lang_embeddings

    # global action_lock
    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    while not rospy.is_shutdown():
        try:
            time1 = time.time()
            # 1) Get latest observation (non-blocking)
            update_observation_window(args, config, ros_operator)

            print("Get Observation Time", time.time() - time1, "s")
            time1 = time.time()

            latest_obs = observation_window[-1]
            imgs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            # BGR->RGB and pad/resize to model input size
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            proprio = latest_obs["qpos"]

            # 2) Build payload
            payload = {
                "state": proprio,
                "images": {
                    "top_head":  imgs[0].transpose(2, 0, 1),   # CHW
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left":  imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }


            # 3) Infer (block until chunk received)
            # Expected: {"actions": np.ndarray [chunk, state_dim]}
            # out = policy.infer(payload)
            # actions = out.get("actions", None)
            actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - time1, "s")
            time1 = time.time()

            # 4) Push to stream buffer (consumed by main loop / temporal smoothing)
            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
                # Record chunk for debug plotting
                try:
                    step_now = max(len(published_actions_history), len(observed_qpos_history))
                    with inferred_chunks_lock:
                        inferred_chunks.append({
                            "start_step": int(step_now),
                            "chunk": np.asarray(actions, dtype=float).copy()
                        })
                except Exception:
                    pass
            elif actions is None:
                print("actions is None")
            elif len(actions) == 0:
                print("len(actions) == 0")

            # 5) Throttle inference rate
            print("Append Buffer Time", time.time() - time1, "s")
            time1 = time.time()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        except Exception as e:
            rospy.logwarn(f"[inference_fn_non_blocking_fast] {e}")
            try:
                rate.sleep()
            except Exception:
                try:
                    time.sleep(0.001)
                except Exception:
                    pass
            continue


class StreamActionBuffer:
    """
    Chunk queue for actions. New chunks appended; each publish step pops one action;
    supports latency trim and temporal smoothing over overlapping chunks.
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

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (unused)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate new inference chunk: (1) trim front by max_k for latency; (2) if cur_chunk
        exists, smooth overlap (100% old -> 0% old); (3) reset k=0.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if str(self.smooth_method).lower() == "raw":
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return

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

    

    def pop_left_step_from_all(self):
        """Legacy (unused)."""
        with self.lock:
            if len(self.cur_chunk) > 0:
                self.cur_chunk.popleft()

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return next action to publish; increment k."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def temporal_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        """Legacy (unused)."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            return np.asarray(self.cur_chunk[0], dtype=float)


    


    def delta_eef_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        # TODO: implement this
        pass


def inference_fn_non_blocking_rtc(args, config, policy, ros_operator):
    """Async RTC inference thread: use prev_chunk and delay; push inferred chunk to stream_buffer."""
    global stream_buffer, observation_window, rtc_prev_chunk
    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    chunk_size = config["chunk_size"]
    exec_h = chunk_size if getattr(args, "rtc_execute_horizon", None) is None else args.rtc_execute_horizon
    exec_h = max(1, min(exec_h, chunk_size))
    while not rospy.is_shutdown() and not shutdown_event.is_set():
        try:
            update_observation_window(args, config, ros_operator)
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
                    "top_head": image_arrs[0].transpose(2, 0, 1),
                    "hand_right": image_arrs[1].transpose(2, 0, 1),
                    "hand_left": image_arrs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
                "execute_horizon": exec_h,
                "enable_rtc": True,
                "mask_prefix_delay": getattr(args, "rtc_mask_prefix_delay", False),
                "max_guidance_weight": getattr(args, "rtc_max_guidance_weight", 0.5),
            }
            with rtc_prev_chunk_lock:
                pc = np.array(rtc_prev_chunk) if rtc_prev_chunk is not None else None
            if pc is not None:
                payload["prev_action_chunk"] = pc.tolist()
            payload["inference_delay"] = int(max(0, pred_delay_steps))
            out = policy.infer(payload)
            actions = out.get("actions", None) if isinstance(out, dict) else None
            if actions is None or len(actions) == 0:
                rate.sleep()
                continue
            # Update prev_chunk for next call
            with rtc_prev_chunk_lock:
                rtc_prev_chunk = np.array(actions, dtype=float)
            # Push to buffer
            stream_buffer.integrate_new_chunk(
                np.asarray(actions, dtype=float),
                max_k=int(getattr(args, "latency_k", 0)),
                min_m=int(getattr(args, "min_smooth_steps", 8)),
            )
        except Exception as e:
            rospy.logwarn(f"[inference_fn_non_blocking_rtc] {e}")
        try:
            rate.sleep()
        except Exception:
            pass



def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    time3 = time.time()

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if time.time() - time3 > 0.01:
            print("Get Frame Time is too long", time.time() - time3, "s")
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            robot_base,
        ) = result
        # print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
            }
        )

    # time2 = time.time()
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args, ros_operator)
    # print("Get Observation Time", time.time() - time2, "s")
    img_front = jpeg_mapping(img_front)
    img_left = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
        axis=0,
    )

    observation_window.append(
        {
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
        }
    )
    # Record observed qpos for plotting
    try:
        observed_qpos_history.append(np.asarray(qpos, dtype=float).copy())
    except Exception:
        pass


def inference_fn(args, config, policy):
    global observation_window
    global lang_embeddings

    # print(f"Start inference_thread_fn: t={t}")
    while True and not rospy.is_shutdown():
        # time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        # convert bgr ro rgb
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]["qpos"]

        payload = {
            "state": proprio,
            "images": {
                "top_head": image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left": image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }

        time1 = time.time()

        # actions shaped as [64, 14] in format [left, right]
        actions = policy.infer(payload)["actions"]


        print(f"Model inference time: {(time.time() - time1)*1000:.3f} ms")

        return actions


# --- RTC helpers (delay estimation & chunk execution) ---
def _update_delay_buffer(rtt_sec: float, publish_rate: float):
    """Record one inference RTT and update predicted delay steps."""
    global pred_delay_steps
    if rtt_sec is None or not np.isfinite(rtt_sec):
        return
    delay_buffer.append(float(rtt_sec))
    if len(delay_buffer) == 0:
        pred_delay_steps = 0
        return
    median_rtt = float(np.median(np.asarray(delay_buffer, dtype=float)))
    pred_delay_steps = int(max(0, round(median_rtt * float(publish_rate))))


def _rtc_infer(
    policy,
    payload: dict,
    prev_chunk: np.ndarray | None,
    delay_steps: int,
    execute_horizon: int,
    publish_rate: float,
    *,
    enable_rtc: bool,
    mask_prefix_delay: bool,
    max_guidance_weight: float,
):
    """Inference wrapper with delay and prev chunk. Returns actions and RTT (sec)."""
    rtc_payload = dict(payload)
    if prev_chunk is not None:
        pc = np.asarray(prev_chunk, dtype=float)
        # pad/crop to model action_dim=32 to avoid server-side shape surprises
        if pc.shape[-1] < 32:
            pad_dim = 32 - pc.shape[-1]
            pc = np.concatenate([pc, np.zeros((*pc.shape[:-1], pad_dim), dtype=pc.dtype)], axis=-1)
        elif pc.shape[-1] > 32:
            pc = pc[..., :32]
        rtc_payload["prev_action_chunk"] = pc.tolist()
    rtc_payload["inference_delay"] = int(max(0, delay_steps))
    rtc_payload["execute_horizon"] = int(max(1, execute_horizon))
    # Enable RTC with conservative params
    rtc_payload["enable_rtc"] = bool(enable_rtc)
    rtc_payload["mask_prefix_delay"] = bool(mask_prefix_delay)
    rtc_payload["max_guidance_weight"] = float(max_guidance_weight)

    t0 = time.time()
    out = policy.infer(rtc_payload)
    rtt = time.time() - t0
    _update_delay_buffer(rtt, publish_rate)
    return out, rtt


def _rtc_align_chunks(prev_chunk: np.ndarray, next_chunk: np.ndarray, delay_steps: int, execute_horizon: int):
    """
    Align action chunk per paper real-time execution.
    Returns:
      actions_to_execute: [execute_horizon, state_dim]
      shifted_chunk:      [chunk_size, state_dim] (new chunk shifted for next round)
    """
    delay_steps = max(0, delay_steps)
    execute_horizon = max(1, execute_horizon)
    chunk_size = prev_chunk.shape[0]
    assert next_chunk.shape[0] == chunk_size, (prev_chunk.shape, next_chunk.shape)
    d = min(delay_steps, execute_horizon, chunk_size)
    s = min(execute_horizon, chunk_size)
    actions_to_execute = np.concatenate(
        [
            prev_chunk[:d],
            next_chunk[d:s],
        ],
        axis=0,
    )
    # Shift new chunk, reserve execute_horizon slots
    pad = np.zeros((s, next_chunk.shape[1]), dtype=float)
    shifted = np.concatenate([next_chunk[s:], pad], axis=0)
    return actions_to_execute, shifted


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    global lang_embeddings

    global stream_buffer

    # Load client
    policy = websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Initialize position of the puppet arm
    left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    # left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.0]
    right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    # right0 = [0.0042737800000000005, -0.020549032000000002, 0.005773964, 0.020392036000000002, 0.413108808, 0.08352187200000001, 0.0975]
    # right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.0]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    # Startup: one blocking inference for warmup
    try:
        update_observation_window(args, config, ros_operator)
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
                "top_head": image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left": image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }
        try:
            _ = policy.infer(payload)
        except Exception as e:
            rospy.logwarn(f"[startup_warmup_infer] {e}")
    except Exception as e:
        rospy.logwarn(f"[startup_warmup_prep] {e}")
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config["state_dim"])
    action = None
    # Persistent accumulated joint correction offsets (first 6 joints)
    corr_left_q6 = np.zeros(6, dtype=float)
    corr_right_q6 = np.zeros(6, dtype=float)
    corrected_action_buffer = np.zeros([chunk_size, config["state_dim"]])
    # Warm start prev_chunk with current proprio to avoid first RTC step pulling toward zeros
    prev_chunk = np.tile(proprio[None, : config["state_dim"]], (chunk_size, 1))
    with rtc_prev_chunk_lock:
        rtc_prev_chunk = prev_chunk.copy()
    rtc_pending_actions = np.zeros((0, config["state_dim"]), dtype=float)
    rtc_plan_idx = 0
    rtc_exec_h = chunk_size if getattr(args, "rtc_execute_horizon", None) is None else args.rtc_execute_horizon
    execute_horizon = min(rtc_exec_h, chunk_size)
    # Inference loop
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():


                if not args.use_temporal_smoothing and not args.use_delta_eef_smoothing:
                    if args.rtc_mode:
                        # Async RTC: inference thread fills stream_buffer; here only consume
                        if stream_buffer is None:
                            stream_buffer = StreamActionBuffer(
                                max_chunks=args.buffer_max_chunks,
                                decay_alpha=args.exp_decay_alpha,
                                state_dim=config["state_dim"],
                                smooth_method="raw" if getattr(args, "rtc_disable_smoothing", False) else "temporal",
                            )
                            # Start thread
                            threading.Thread(
                                target=inference_fn_non_blocking_rtc,
                                args=(args, config, policy, ros_operator),
                                daemon=True,
                            ).start()
                        raw_action = stream_buffer.pop_next_action()
                        if raw_action is None:
                            rate.sleep()
                            continue
                    else:
                        # Legacy chunk logic
                        update_observation_window(args, config, ros_operator)
                        # When coming to the end of the action chunk
                        if t % chunk_size == 0:
                            # Start inference
                            action_buffer = inference_fn(args, config, policy).copy()
                            # Build corrected actions for the whole chunk (apply deltas on raw actions)
                            corrected_action_buffer = action_buffer.copy()

                            if args.use_ik_fine_tuning:
                                max_n = min(chunk_size, corrected_action_buffer.shape[0]) if hasattr(corrected_action_buffer, "shape") else chunk_size
                                if getattr(args, "eef_corr_all_action_chunk", True):
                                    # Batch correction per raw action
                                    ql_batch = corrected_action_buffer[:max_n, :6]
                                    qr_batch = corrected_action_buffer[:max_n, 7:13]
                                    ql_new_batch = apply_micro_correction_batch(
                                        ql_batch,
                                        args.eef_corr_left,
                                        args.eef_corr_left_frame,
                                        args.eef_corr_lambda,
                                        args.eef_corr_step_limit_m,
                                        args.eef_corr_joint_step_limits,
                                    )
                                    qr_new_batch = apply_micro_correction_batch(
                                        qr_batch,
                                        args.eef_corr_right,
                                        args.eef_corr_right_frame,
                                        args.eef_corr_lambda,
                                        args.eef_corr_step_limit_m,
                                        args.eef_corr_joint_step_limits,
                                    )
                                    corrected_action_buffer[:max_n, :6] = ql_new_batch
                                    corrected_action_buffer[:max_n, 7:13] = qr_new_batch
                                else:
                                    # Single delta computed from the first raw action, apply to all (batch)
                                    ba0 = action_buffer[0]
                                    ql_ref0 = np.array(ba0[:6], dtype=float)
                                    qr_ref0 = np.array(ba0[7:13], dtype=float)
                                    ql_new0 = apply_micro_correction(
                                        ql_ref0,
                                        args.eef_corr_left,
                                        args.eef_corr_left_frame,
                                        args.eef_corr_lambda,
                                        args.eef_corr_step_limit_m,
                                        args.eef_corr_joint_step_limits,
                                    )
                                    qr_new0 = apply_micro_correction(
                                        qr_ref0,
                                        args.eef_corr_right,
                                        args.eef_corr_right_frame,
                                        args.eef_corr_lambda,
                                        args.eef_corr_step_limit_m,
                                        args.eef_corr_joint_step_limits,
                                    )
                                    delta_left_single = (ql_new0 - ql_ref0)
                                    delta_right_single = (qr_new0 - qr_ref0)
                                    corrected_action_buffer[:max_n, :6] = action_buffer[:max_n, :6] + delta_left_single
                                    corrected_action_buffer[:max_n, 7:13] = action_buffer[:max_n, 7:13] + delta_right_single

                        # Use corrected raw action for this step
                        raw_action = corrected_action_buffer[t % chunk_size]

                    # Interpolate action sequence (already corrected)
                    if args.use_actions_interpolation:
                        interp_actions = interpolate_action(args, pre_action, raw_action)
                    else:
                        interp_actions = raw_action[np.newaxis, :]
                    # Execute the interpolated actions one by one (no extra delta here)
                    for act in interp_actions:
                        if args.use_temporal_smoothing:
                            with ros_operator.lock:
                                if ros_operator.communication_flag:
                                    rate.sleep()
                                    continue
                                ros_operator.communication_flag = True
                        else:
                            if args.ctrl_type == "joint":
                                left_action = act[:7].copy()
                                right_action = act[7:14].copy()
                                right_action[6] = max(0.0, right_action[6]-RIGHT_OFFSET)
                                ros_operator.puppet_arm_publish(left_action, right_action)
                                try:
                                    published_actions_history.append(np.concatenate([left_action, right_action], axis=0).astype(float))
                                except Exception:
                                    pass
                            elif args.ctrl_type == "eef":
                                left_action = act[:7]
                                right_action = act[7:14]

                                ros_operator.endpose_publish(left_action, right_action)

                            if args.use_robot_base:
                                vel_action = act[14:16]
                                ros_operator.robot_base_publish(vel_action)

                        rate.sleep()
                    t += 1

                    print("Published Step", t)
                    try:
                        publish_step_global = len(published_actions_history)
                    except Exception:
                        pass
                    try:
                        published_actions_history.append(np.concatenate([left_action, right_action], axis=0).astype(float))
                    except Exception:
                        pass
                    # Track previous action as corrected raw action for next interpolation
                    pre_action = raw_action.copy()

             
                if shutdown_event.is_set():
                    break
                if args.use_temporal_smoothing:

                    if stream_buffer is None:
                        stream_buffer = StreamActionBuffer(
                            max_chunks=args.buffer_max_chunks,
                            decay_alpha=args.exp_decay_alpha,
                            state_dim=config["state_dim"],
                            smooth_method="temporal",
                        )
                        start_inference_thread(inference_fn_non_blocking_fast, (args, config, policy, ros_operator))
                    act = stream_buffer.pop_next_action()
                    if act is not None:
                        if args.ctrl_type == "joint":
                            left_action = act[:7].copy()
                            right_action = act[7:14].copy()
                            left_action[6] = max(0.0, left_action[6]-RIGHT_OFFSET)
                            right_action[6] = max(0.0, right_action[6]-RIGHT_OFFSET)
                            ros_operator.puppet_arm_publish(left_action, right_action)
                            try:
                                published_actions_history.append(np.concatenate([left_action, right_action], axis=0).astype(float))
                            except Exception:
                                pass
                        elif args.ctrl_type == "eef":
                            left_action = act[:7]
                            right_action = act[7:14]
                    else:
                        print("act is None")
                        time.sleep(0.001)
                        continue
                    # if args.use_robot_base:
                    #     vel_action = act[14:16]
                    #     ros_operator.robot_base_publish(vel_action)
                    print("Published Step", t)
                    try:
                        publish_step_global = len(published_actions_history)
                    except Exception:
                        pass

                    rate.sleep()
                    t += 1

                if shutdown_event.is_set():
                    break


# ROS operator class
class RosOperator:
    def __init__(self, args):
        self.communication_thread = None  
        self.communication_flag = False   
        self.lock = threading.Lock()      
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.endpose_left_publisher = None
        self.endpose_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # Set timestep
        joint_state_msg.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def endpose_publish(self, left, right):
        endpose_msg = PosCmd()
        endpose_msg.x, endpose_msg.y, endpose_msg.z = left[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = left[3:6]
        endpose_msg.gripper = left[6]
        self.endpose_left_publisher.publish(endpose_msg)

        endpose_msg.x, endpose_msg.y, endpose_msg.z = right[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = right[3:6]
        endpose_msg.gripper = right[6]
        self.endpose_right_publisher.publish(endpose_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # Set the timestep
            joint_state_msg.name = [
                "joint0",
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = [
                "joint0",
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (
                self.args.use_depth_image
                and (
                    len(self.img_left_depth_deque) == 0
                    or len(self.img_right_depth_deque) == 0
                    or len(self.img_front_depth_deque) == 0
                )
            )
        ):
            return False
        if self.args.use_depth_image:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                    self.img_left_depth_deque[-1].header.stamp.to_sec(),
                    self.img_right_depth_deque[-1].header.stamp.to_sec(),
                    self.img_front_depth_deque[-1].header.stamp.to_sec(),
                ]
            )
        else:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                ]
            )

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (
            len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_depth_image and (
            len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_depth_image and (
            len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False
        if self.args.use_robot_base and (
            len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), "passthrough")

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "passthrough")

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), "passthrough")

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), "passthrough")

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), "passthrough")

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            robot_base,
        )

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node("joint_state_publisher", anonymous=True)
        rospy.Subscriber(
            self.args.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_right_topic,
            Image,
            self.img_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        if self.args.use_depth_image:
            rospy.Subscriber(
                self.args.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.img_right_depth_topic,
                Image,
                self.img_right_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=1000,
                tcp_nodelay=True,
            )
        rospy.Subscriber(
            self.args.puppet_arm_left_topic,
            JointState,
            self.puppet_arm_left_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.puppet_arm_right_topic,
            JointState,
            self.puppet_arm_right_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.robot_base_topic,
            Odometry,
            self.robot_base_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10
        )
        self.endpose_left_publisher = rospy.Publisher(self.args.endpose_left_cmd_topic, PosCmd, queue_size=10)
        self.endpose_right_publisher = rospy.Publisher(self.args.endpose_right_cmd_topic, PosCmd, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Random seed",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_cmd_topic",
        action="store",
        type=str,
        help="endpose_left_cmd_topic",
        default="/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_cmd_topic",
        action="store",
        type=str,
        help="endpose_right_cmd_topic",
        default="/pos_cmd_right",
        required=False,
    )

    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--rtc_mode",
        action="store_true",
        help="Enable real-time chunking alignment (requires server realtime_action support)",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--rtc_mask_prefix_delay",
        action="store_true",
        help="Enable masking prefix with previous chunk during RTC (can increase stability or cause jumps)",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--rtc_max_guidance_weight",
        action="store",
        type=float,
        help="Max guidance weight for RTC (smaller is more stable)",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--rtc_execute_horizon",
        action="store",
        type=int,
        help="Execute horizon s_min for RTC (defaults to chunk_size when None)",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--rtc_disable_smoothing",
        action="store_true",
        help="Disable chunk smoothing in RTC mode (just drop delay prefix and execute raw chunk)",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        help="The maximum change allowed for each joint per timestep",
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_actions_interpolation",
        action="store_true",
        help="Whether to interpolate the actions if the difference is too large",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="localhost",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )

    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )

    parser.add_argument(
        "--use_ik_fine_tuning",
        action="store_true",
        help="Whether to use IK fine-tuning",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--eef_corr_left",
        nargs=3,
        type=float,
        metavar=("DX","DY","DZ"),
        help="Left arm EE micro-correction (meters) in selected frame",
        default=[0.0, 0.0, 0.0],
        required=False,
    )
    parser.add_argument(
        "--eef_corr_right",
        nargs=3,
        type=float,
        metavar=("DX","DY","DZ"),
        help="Right arm EE micro-correction (meters) in selected frame",
        default=[0.0, 0.0, -0.1],
        required=False,
    )
    parser.add_argument(
        "--eef_corr_left_frame",
        type=str,
        choices=["tool", "base"],
        help="Interpret left correction in this frame",
        default="base",
        required=False,
    )
    parser.add_argument(
        "--eef_corr_right_frame",
        type=str,
        choices=["tool", "base"],
        help="Interpret right correction in this frame",
        default="base",
        required=False,
    )
    parser.add_argument(
        "--eef_corr_lambda",
        type=float,
        help="Damping for DLS (position-only IK step)",
        default=0.001,
        required=False,
    )
    parser.add_argument(
        "--eef_corr_step_limit_m",
        type=float,
        help="Max EE correction magnitude per cycle (meters)",
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--eef_corr_joint_step_limits",
        nargs=6,
        type=float,
        metavar=("dq1","dq2","dq3","dq4","dq5","dq6"),
        help="Per-joint max step (rad) applied to DLS increment",
        default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        required=False,
    )
    parser.add_argument(
        "--eef_corr_all_action_chunk",
        action="store_true",
        help="Apply EE correction per action in a chunk before interpolation (default ON)",
        default=True,
        required=False,
    )

    parser.add_argument(
        "--use_temporal_smoothing",
        action="store_true",
        help="Enable non-blocking communication and control execution",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--latency_k",
        type=int,
        help="Max Latency in steps",
        default=8,
        required=False,
    )
    parser.add_argument(
        "--inference_rate",
        type=float,
        help="Inference loop rate (Hz)",
        default=3.0,
        required=False,
    )
    parser.add_argument(
        "--min_smooth_steps",
        type=int,
        help="Minimum smoothing steps m",
        default=8,
        required=False,
    )
    parser.add_argument(
        "--buffer_max_chunks",
        type=int,
        help="Maximum number of chunks in the buffer",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--exp_decay_alpha",
        type=float,
        help="Exponential decay alpha",
        default=0.25,
        required=False,
    )
    
    parser.add_argument(
        "--use_delta_eef_smoothing",
        action="store_true",
        help="Whether to use delta eef smoothing",
        default=False,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    # Register SIGINT to save plots on Ctrl+C
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup: save all plots
        save_debug_plots_on_interrupt()


if __name__ == "__main__":
    main()


# [WARN] [1761292008.952089469]: Hardware Notification:Right MIPI error,1.76129e+12,Error,Hardware Error
# [WARN] [1761292009.914298641]: Hardware Notification:Depth stream start failure,1.76129e+12,Error,Hardware Error
# [WARN] [1761292009.942287487]: Hardware Notification:Depth stream start failure,1.76129e+12,Error,Hardware Error
# [WARN] [1761292009.952186196]: Hardware Notification:Depth stream start failure,1.76129e+12,Error,Hardware Error

# conda activate aloha_pi0_py310 && python agilex_inference_openpi_rtc.py --host 192.168.9.239 --port 8000 --rtc_mode --chunk_size 50 --rtc_execute_horizon 8

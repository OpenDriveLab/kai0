# -- coding: UTF-8
"""
OpenPi inference with multiple smoothing methods (--smooth_method).

1. naive_async:
   - Ignores previous action chunks; switches to the new chunk as soon as it is ready.
   - Simplest and fastest, but may be less smooth at chunk boundaries.

2. temporal_ensembling (temporal_agg):
   - Each inference returns an action chunk [chunk_size, state_dim].
   - Maintains a history of predictions per timestep.
   - For timestep t, aggregates all actions that were ever predicted for t.
   - Uses exponential weights: wi = exp(-m * i), w0 for the oldest prediction.
   - Smaller m means new observations are incorporated faster.
   - Key: aggregates multiple predictions for the *same* timestep, not adjacent timesteps.
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
import signal
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]

action_buffer = None   # type: TemporalEnsemblingBuffer or NaiveAsyncBuffer

observation_window = None

lang_embeddings = "fold the sleeve"

RIGHT_OFFSET = 0.003
published_actions_history = []  # list[np.ndarray(shape=(14,))]
observed_qpos_history = []      # list[np.ndarray(shape=(14,))]
publish_step_global = 0     
shutdown_event = threading.Event()


class TemporalEnsemblingBuffer:
    """
    Temporal Ensembling Buffer - ACT-style temporal aggregation.

    - Stores predictions per (inference_idx, timestep).
    - For timestep t, aggregates all chunks that ever predicted t.
    - Weights: wi = exp(-m * i), i=0 is oldest.

    Args:
        max_timesteps: Maximum number of timesteps.
        chunk_size: Chunk size per inference.
        state_dim: State dimension.
        exp_weight_m: Exponential weight m; smaller = faster incorporation of new obs (typical 0.01--0.1).
    """
    def __init__(self, max_timesteps=10000, chunk_size=50, state_dim=14, exp_weight_m=0.01):
        self.max_timesteps = max_timesteps
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.exp_weight_m = exp_weight_m
        self.lock = threading.Lock()

        # Sparse storage: only (inference_idx, timestep) that have predictions.
        # Format: {timestep: [(inference_idx, action), ...]}
        self.predictions = {}  # dict[int, list[tuple[int, np.ndarray]]]

        self.current_t = 0
        self.inference_count = 0
        self.last_action = None  # fallback when no prediction

    def add_chunk(self, actions_chunk: np.ndarray, start_timestep: int = None):
        """
        Add a new inference chunk.
        actions_chunk: [chunk_size, state_dim]. start_timestep: chunk start; None = use current_t.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            if start_timestep is None:
                start_timestep = self.current_t

            inference_idx = self.inference_count
            self.inference_count += 1

            for i, action in enumerate(actions_chunk):
                timestep = start_timestep + i
                if timestep < 0:
                    continue
                if timestep not in self.predictions:
                    self.predictions[timestep] = []
                self.predictions[timestep].append((inference_idx, action.copy()))

            self._cleanup_old_predictions()

    def _cleanup_old_predictions(self):
        """Remove predictions for timesteps already executed."""
        cleanup_threshold = max(0, self.current_t - 10)
        keys_to_remove = [t for t in self.predictions.keys() if t < cleanup_threshold]
        for t in keys_to_remove:
            del self.predictions[t]
    
    def _get_action_unlocked(self, timestep: int) -> np.ndarray:
        """Get aggregated action for timestep (caller must hold lock). Exponential weighted average."""
        if timestep not in self.predictions or len(self.predictions[timestep]) == 0:
            return self.last_action

        predictions = self.predictions[timestep]
        num_predictions = len(predictions)

        if num_predictions == 1:
            action = predictions[0][1].copy()
            self.last_action = action.copy()
            return action

        predictions_sorted = sorted(predictions, key=lambda x: x[0])
        actions = np.array([p[1] for p in predictions_sorted])
        indices = np.arange(num_predictions)
        exp_weights = np.exp(-self.exp_weight_m * indices)
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights[:, np.newaxis]
        aggregated_action = (actions * exp_weights).sum(axis=0)
        
        self.last_action = aggregated_action.copy()
        return aggregated_action

    def get_action(self, timestep: int = None) -> np.ndarray:
        """Get aggregated action for timestep (default current_t). Returns None if no prediction."""
        with self.lock:
            if timestep is None:
                timestep = self.current_t
            return self._get_action_unlocked(timestep)
    
    def pop_next_action(self) -> np.ndarray:
        """Pop and return the next action to execute; increment current_t."""
        with self.lock:
            action = self._get_action_unlocked(self.current_t)
            self.current_t += 1
            return action
    
    def has_prediction(self, timestep: int = None) -> bool:
        """Whether there is a prediction for the given timestep."""
        with self.lock:
            if timestep is None:
                timestep = self.current_t
            return timestep in self.predictions and len(self.predictions[timestep]) > 0
    
    def get_current_timestep(self) -> int:
        """Current timestep."""
        with self.lock:
            return self.current_t
    
    def reset(self):
        """Reset buffer."""
        with self.lock:
            self.predictions = {}
            self.current_t = 0
            self.inference_count = 0
            self.last_action = None


class NaiveAsyncBuffer:
    """
    Naive async: switch to the new chunk as soon as it is ready; no smoothing.
    Uses global timestep to index into the current chunk (accounts for inference delay).
    """
    def __init__(self, chunk_size=50, state_dim=14):
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.lock = threading.Lock()
        self.current_chunk = None
        self.chunk_start_t = 0
        self.global_t = 0
        self.last_action = None

    def add_chunk(self, actions_chunk: np.ndarray, start_timestep: int = None):
        """
        Replace current chunk. Skip steps = global_t - start_timestep for latency compensation.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return

            self.current_chunk = actions_chunk.copy()

            if start_timestep is not None:
                skip_steps = max(0, self.global_t - start_timestep)
            else:
                skip_steps = 0
            skip_steps = min(skip_steps, len(actions_chunk) - 1)
            
            self.chunk_start_t = self.global_t - skip_steps
            
            print(f"[NaiveAsync] New chunk received, size={len(actions_chunk)}, "
                  f"global_t={self.global_t}, start_timestep={start_timestep}, "
                  f"skip_steps={skip_steps}, chunk_start_t={self.chunk_start_t}")
    
    def pop_next_action(self) -> np.ndarray:
        """Return next action from current chunk, or last_action if chunk exhausted."""
        with self.lock:
            if self.current_chunk is None:
                self.global_t += 1
                return self.last_action

            chunk_index = self.global_t - self.chunk_start_t

            if chunk_index >= len(self.current_chunk):
                self.global_t += 1
                return self.last_action

            if chunk_index < 0:
                chunk_index = 0
            
            action = self.current_chunk[chunk_index].copy()
            self.global_t += 1
            self.last_action = action.copy()
            return action
    
    def has_prediction(self, timestep: int = None) -> bool:
        """Whether an action is available (current chunk or last_action)."""
        with self.lock:
            if self.current_chunk is None:
                return self.last_action is not None
            chunk_index = self.global_t - self.chunk_start_t
            if chunk_index < len(self.current_chunk):
                return True
            return self.last_action is not None

    def get_current_timestep(self) -> int:
        """Current global timestep."""
        with self.lock:
            return self.global_t
    
    def reset(self):
        """Reset buffer."""
        with self.lock:
            self.current_chunk = None
            self.chunk_start_t = 0
            self.global_t = 0
            self.last_action = None


def inference_fn_action_buffer(args, config, policy, ros_operator):
    """
    Inference thread: pack latest observation into payload, call infer, add chunk to action_buffer.
    """
    global action_buffer
    global observation_window
    global lang_embeddings

    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    
    while not rospy.is_shutdown() and not shutdown_event.is_set():
        try:
            time1 = time.time()
            
            # 1) Get latest observation
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
                    "top_head": imgs[0].transpose(2, 0, 1),   # CHW
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left": imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }

            # 3) Record current timestep before inference (for latency compensation)
            inference_start_t = action_buffer.get_current_timestep()
            
            # 4) Infer
            actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - time1, "s")
            time1 = time.time()

            # 5) Add chunk to action buffer (pass start_timestep for latency compensation)
            if actions is not None and len(actions) > 0:
                current_t = inference_start_t
                action_buffer.add_chunk(actions, start_timestep=current_t)
                print(f"[{args.smooth_method}] Added chunk, current_t={current_t}, chunk_size={len(actions)}")

            print("Add Chunk Time", time.time() - time1, "s")

            # 5) Throttle inference rate
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        except Exception as e:
            rospy.logwarn(f"[inference_fn_action_buffer] {e}")
            try:
                rate.sleep()
            except Exception:
                try:
                    time.sleep(0.001)
                except Exception:
                    pass
            continue


def start_inference_thread(args, config, policy, ros_operator):
    """Start inference thread."""
    inference_thread = threading.Thread(
        target=inference_fn_action_buffer, 
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
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)



def get_config(args):
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


def get_ros_observation(args, ros_operator):
    """Get observation from ROS topics."""
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
        return (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right)


def update_observation_window(args, config, ros_operator):
    """Update observation window buffer."""
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
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

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args, ros_operator)
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


def model_inference(args, config, ros_operator):
    """
    Main inference loop. Uses args.smooth_method: naive_async (switch to latest chunk)
    or temporal_ensembling (aggregate multiple predictions per timestep).
    """
    global lang_embeddings
    global action_buffer

    # Load OpenPi client
    policy = websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")
    print(f"Using smooth method: {args.smooth_method}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Initialize arm pose
    left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    # Warmup inference
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

    # Inference loop
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                if shutdown_event.is_set():
                    break

                # Initialize buffer by smooth method
                if action_buffer is None:
                    if args.smooth_method == "naive_async":
                        action_buffer = NaiveAsyncBuffer(
                            chunk_size=chunk_size,
                            state_dim=config["state_dim"],
                        )
                        print("Initialized NaiveAsyncBuffer")
                    elif args.smooth_method == "temporal_ensembling":
                        action_buffer = TemporalEnsemblingBuffer(
                            max_timesteps=max_publish_step + chunk_size,
                            chunk_size=chunk_size,
                            state_dim=config["state_dim"],
                            exp_weight_m=args.exp_weight_m,
                        )
                        print(f"Initialized TemporalEnsemblingBuffer with exp_weight_m={args.exp_weight_m}")
                    else:
                        raise ValueError(f"Unknown smooth_method: {args.smooth_method}")
                    
                    start_inference_thread(args, config, policy, ros_operator)

                # Wait for prediction
                if not action_buffer.has_prediction():
                    print("Waiting for prediction...")
                    time.sleep(0.001)
                    continue

                act = action_buffer.pop_next_action()
                
                if act is not None:
                    if args.ctrl_type == "joint":
                        left_action = act[:7].copy()
                        right_action = act[7:14].copy()
                        left_action[6] = max(0.0, left_action[6] - RIGHT_OFFSET)
                        right_action[6] = max(0.0, right_action[6] - RIGHT_OFFSET)
                        ros_operator.puppet_arm_publish(left_action, right_action)
                        published_actions_history.append(
                            np.concatenate([left_action, right_action], axis=0).astype(float)
                        )
                    else:
                        print("Make sure ctrl_type is joint")
                else:
                    print("act is None")
                    time.sleep(0.001)
                    continue

                if args.smooth_method == "temporal_ensembling":
                    num_preds = len(action_buffer.predictions.get(t, []))
                    print(f"Published Step {t}, num_predictions for this step: {num_preds}")
                else:
                    print(f"Published Step {t}, global_t: {action_buffer.global_t}")
                
                try:
                    publish_step_global = len(published_actions_history)
                except Exception:
                    pass

                rate.sleep()
                t += 1

                if shutdown_event.is_set():
                    break


class RosOperator:
    """ROS communication (subscribe/publish)."""
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
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            "joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
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
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = [
                "joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
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
                "joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
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
        self.puppet_arm_publish_thread = threading.Thread(
            target=self.puppet_arm_publish_continuous, args=(left, right)
        )
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
        self.puppet_arm_left_publisher = rospy.Publisher(
            self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10
        )
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10
        )
        self.endpose_left_publisher = rospy.Publisher(
            self.args.endpose_left_cmd_topic, PosCmd, queue_size=10
        )
        self.endpose_right_publisher = rospy.Publisher(
            self.args.endpose_right_cmd_topic, PosCmd, queue_size=10
        )
        self.robot_base_publisher = rospy.Publisher(
            self.args.robot_base_cmd_topic, Twist, queue_size=10
        )


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
        "--inference_rate",
        type=float,
        help="Inference loop rate (Hz)",
        default=3.0,
        required=False,
    )
    parser.add_argument(
        "--smooth_method",
        type=str,
        choices=["naive_async", "temporal_ensembling"],
        help="Smoothing method for action chunks: "
             "naive_async: directly switch to new chunk when ready (no smoothing); "
             "temporal_ensembling: aggregate predictions for the same timestep with exponential weights",
        default="temporal_ensembling",
        required=False,
    )
    parser.add_argument(
        "--exp_weight_m",
        type=float,
        help="[temporal_ensembling only] Exponential weight parameter m. "
             "wi = exp(-m * i), where i=0 is the oldest prediction. "
             "Smaller m means faster incorporation of new observations. "
             "Typical values: 0.01 ~ 0.3",
        default=0.01,
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
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

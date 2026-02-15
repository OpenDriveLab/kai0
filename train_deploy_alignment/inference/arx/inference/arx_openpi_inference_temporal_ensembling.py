#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 dual-arm inference with temporal ensembling (same design as Agilex).

--smooth_method:
  - naive_async: use latest chunk as soon as ready; latency skip.
  - temporal_ensembling: aggregate multiple predictions per timestep with exp weights (exp_weight_m).

Inference runs in a background thread; main loop waits for has_prediction(), then pop_next_action()
and set_joint_positions. Set lang_embeddings in arx_openpi_inference_rtc or override below.
See train_deploy_alignment/inference/arx/README.md.
"""

import argparse
import threading
import time
import sys
import os
import signal

import cv2
import numpy as np
import rclpy
from openpi_client import image_tools, websocket_client_policy

import arx_openpi_inference_rtc as arx

# Same camera order as rtc
CAMERA_NAMES = getattr(arx, "CAMERA_NAMES", ["cam_high", "cam_right_wrist", "cam_left_wrist"])
action_buffer = None  # TemporalEnsemblingBuffer or NaiveAsyncBuffer


class TemporalEnsemblingBuffer:
    """
    Per-timestep aggregation: for timestep t, aggregate all chunks that predicted t.
    Weights: wi = exp(-m * i), i=0 oldest. Same API as Agilex.
    """
    def __init__(self, max_timesteps=10000, chunk_size=50, state_dim=14, exp_weight_m=0.01):
        self.max_timesteps = max_timesteps
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.exp_weight_m = exp_weight_m
        self.lock = threading.Lock()
        self.predictions = {}
        self.current_t = 0
        self.inference_count = 0
        self.last_action = None

    def add_chunk(self, actions_chunk: np.ndarray, start_timestep: int = None):
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
        cleanup_threshold = max(0, self.current_t - 10)
        for t in list(self.predictions.keys()):
            if t < cleanup_threshold:
                del self.predictions[t]

    def _get_action_unlocked(self, timestep: int) -> np.ndarray:
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

    def pop_next_action(self) -> np.ndarray:
        with self.lock:
            action = self._get_action_unlocked(self.current_t)
            self.current_t += 1
            return action

    def has_prediction(self, timestep: int = None) -> bool:
        with self.lock:
            if timestep is None:
                timestep = self.current_t
            return timestep in self.predictions and len(self.predictions[timestep]) > 0

    def get_current_timestep(self) -> int:
        with self.lock:
            return self.current_t


class NaiveAsyncBuffer:
    """Switch to new chunk when ready; skip steps for latency. Same API as Agilex."""
    def __init__(self, chunk_size=50, state_dim=14):
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.lock = threading.Lock()
        self.current_chunk = None
        self.chunk_start_t = 0
        self.global_t = 0
        self.last_action = None

    def add_chunk(self, actions_chunk: np.ndarray, start_timestep: int = None):
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

    def pop_next_action(self) -> np.ndarray:
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
        with self.lock:
            if self.current_chunk is None:
                return self.last_action is not None
            chunk_index = self.global_t - self.chunk_start_t
            if chunk_index < len(self.current_chunk):
                return True
            return self.last_action is not None

    def get_current_timestep(self) -> int:
        with self.lock:
            return self.global_t


def inference_fn_action_buffer(args, config, policy, ros_operator):
    """Inference thread: update obs, build payload, infer, add_chunk with start_timestep."""
    global action_buffer
    rate = ros_operator.create_rate(getattr(args, "inference_rate", 4))
    while rclpy.ok() and not arx.shutdown_event.is_set():
        try:
            arx.update_observation_window(args, config, ros_operator)
            if len(arx.observation_window) == 0:
                rate.sleep()
                continue
            latest_obs = arx.observation_window[-1]
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
                    "top_head": imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left": imgs[2].transpose(2, 0, 1),
                },
                "prompt": arx.lang_embeddings,
            }
            inference_start_t = action_buffer.get_current_timestep()
            out = policy.infer(payload)
            actions = out.get("actions", None) if isinstance(out, dict) else None
            if actions is not None and len(actions) > 0:
                action_buffer.add_chunk(np.asarray(actions, dtype=float), start_timestep=inference_start_t)
            rate.sleep()
        except Exception as e:
            try:
                rate.sleep()
            except Exception:
                time.sleep(0.001)


def start_inference_thread(args, config, policy, ros_operator):
    th = threading.Thread(target=inference_fn_action_buffer, args=(args, config, policy, ros_operator), daemon=True)
    th.start()
    return th


def get_config(args):
    return {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }


def main():
    global action_buffer
    action_buffer = None
    parser = argparse.ArgumentParser(description="ARX-X5 temporal ensembling (naive_async or temporal_ensembling).")
    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)
    parser.add_argument("--smooth_method", choices=("naive_async", "temporal_ensembling"), default="temporal_ensembling")
    parser.add_argument("--exp_weight_m", type=float, default=0.01, help="Exp weight for temporal_ensembling.")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--joint_cmd_topic_right", default="/arm_master_r_status")
    parser.add_argument("--joint_state_topic_right", default="/arm_slave_r_status")
    parser.add_argument("--camera_front_serial", type=str, default="152122073503")
    parser.add_argument("--camera_left_serial", type=str, default="213622070289")
    parser.add_argument("--camera_right_serial", type=str, default="152122073474")
    parser.add_argument("--auto_homing", action="store_true", default=True)
    args = parser.parse_args()

    def _on_sigint(sig, frame):
        arx.shutdown_event.set()
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_sigint)

    rclpy.init()
    ros_operator = arx.ARX5ROSController(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()

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
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Server metadata:", policy.get_server_metadata())
    print("Using smooth method:", args.smooth_method)

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Warmup
    try:
        arx.update_observation_window(args, config, ros_operator)
        if len(arx.observation_window) > 0:
            latest_obs = arx.observation_window[-1]
            imgs = [latest_obs["images"][config["camera_names"][i]] for i in range(3)]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            payload = {
                "state": latest_obs["qpos"],
                "images": {"top_head": imgs[0].transpose(2, 0, 1), "hand_right": imgs[1].transpose(2, 0, 1), "hand_left": imgs[2].transpose(2, 0, 1)},
                "prompt": arx.lang_embeddings,
            }
            _ = policy.infer(payload)
        print("Warmup done.")
    except Exception as e:
        print("Warmup failed:", e)

    rate = ros_operator.create_rate(args.control_frequency)
    t = 0

    try:
        while rclpy.ok() and t < max_publish_step and not arx.shutdown_event.is_set():
            if action_buffer is None:
                if args.smooth_method == "naive_async":
                    action_buffer = NaiveAsyncBuffer(chunk_size=chunk_size, state_dim=config["state_dim"])
                else:
                    action_buffer = TemporalEnsemblingBuffer(
                        max_timesteps=max_publish_step + chunk_size,
                        chunk_size=chunk_size,
                        state_dim=config["state_dim"],
                        exp_weight_m=args.exp_weight_m,
                    )
                start_inference_thread(args, config, policy, ros_operator)

            if not action_buffer.has_prediction():
                time.sleep(0.001)
                continue

            act = action_buffer.pop_next_action()
            if act is not None:
                act = arx.apply_gripper_binary(act)
                ros_operator.set_joint_positions(act)
            if t % 50 == 0:
                print("Published step", t)
            t += 1
            rate.sleep()
    except Exception as e:
        print("Loop error:", e)
        import traceback
        traceback.print_exc()
    finally:
        ros_operator.cleanup_cameras()
        arx.shutdown_event.set()
        if rclpy.ok():
            rclpy.shutdown()
        print("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()

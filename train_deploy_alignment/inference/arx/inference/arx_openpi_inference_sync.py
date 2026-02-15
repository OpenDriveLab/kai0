#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 dual-arm synchronous inference: blocking infer each chunk, then execute step-by-step.

Same logic as Agilex sync: when t % chunk_size == 0, get latest observation, call policy.infer(),
then execute the chunk one step at a time. Optional interpolation between steps. Set lang_embeddings
at top to match training. See train_deploy_alignment/inference/arx/README.md.
"""

import argparse
import time
import threading
import sys
import os
import signal

import cv2
import numpy as np

import rclpy
from openpi_client import image_tools, websocket_client_policy

import arx_openpi_inference_rtc as arx


def get_config(args):
    return {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": arx.CAMERA_NAMES,
    }


def build_payload(latest_obs, config):
    """Build inference payload from latest observation (same as Agilex)."""
    imgs = [
        latest_obs["images"][config["camera_names"][0]],
        latest_obs["images"][config["camera_names"][1]],
        latest_obs["images"][config["camera_names"][2]],
    ]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
    proprio = latest_obs["qpos"]
    return {
        "state": proprio,
        "images": {
            "top_head": imgs[0].transpose(2, 0, 1),
            "hand_right": imgs[1].transpose(2, 0, 1),
            "hand_left": imgs[2].transpose(2, 0, 1),
        },
        "prompt": arx.lang_embeddings,
    }


def inference_fn_blocking(args, config, policy, ros_operator):
    """Blocking: update observation window, build payload, return actions chunk (same as Agilex inference_fn)."""
    arx.update_observation_window(args, config, ros_operator)
    if len(arx.observation_window) == 0:
        return None
    latest_obs = arx.observation_window[-1]
    payload = build_payload(latest_obs, config)
    out = policy.infer(payload)
    actions = out.get("actions", None) if isinstance(out, dict) else None
    return np.asarray(actions, dtype=float) if actions is not None and len(actions) > 0 else None


def main():
    parser = argparse.ArgumentParser(description="ARX-X5 sync inference (blocking infer per chunk, like Agilex sync).")
    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_publish_step", type=int, default=10000000)
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

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Warmup: one blocking inference (discard result), same as Agilex
    try:
        arx.update_observation_window(args, config, ros_operator)
        if len(arx.observation_window) > 0:
            latest_obs = arx.observation_window[-1]
            payload = build_payload(latest_obs, config)
            _ = policy.infer(payload)
        print("Warmup done.")
    except Exception as e:
        print(f"Warmup failed: {e}")

    pre_action = np.zeros(config["state_dim"])
    rate = ros_operator.create_rate(args.control_frequency)
    action_buffer = np.zeros((chunk_size, config["state_dim"]))
    t = 0

    try:
        while rclpy.ok() and t < max_publish_step and not arx.shutdown_event.is_set():
            arx.update_observation_window(args, config, ros_operator)

            if t % chunk_size == 0:
                action_buffer = inference_fn_blocking(args, config, policy, ros_operator)
                if action_buffer is None or len(action_buffer) == 0:
                    rate.sleep()
                    continue
                action_buffer = np.asarray(action_buffer, dtype=float)
                if action_buffer.shape[0] < chunk_size:
                    pad = np.zeros((chunk_size - action_buffer.shape[0], action_buffer.shape[1]), dtype=action_buffer.dtype)
                    action_buffer = np.concatenate([action_buffer, pad], axis=0)
                corrected_action_buffer = action_buffer.copy()

            raw_action = corrected_action_buffer[t % chunk_size]
            act = arx.apply_gripper_binary(raw_action)
            ros_operator.set_joint_positions(act)
            pre_action = raw_action.copy()
            t += 1
            if t % 50 == 0:
                print(f"Published step {t}")
            rate.sleep()
    except Exception as e:
        print(f"Loop error: {e}")
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

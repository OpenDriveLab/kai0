#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>

#include "arx_x5_src/interfaces/InterfacesThread.hpp"

#include "arx5_arm_msg/msg/robot_cmd.hpp"
#include "arx5_arm_msg/msg/robot_status.hpp"
#include "arm_control/msg/pos_cmd.hpp"
#include "arm_control/msg/joint_control.hpp"
#include <std_msgs/msg/int32_multi_array.hpp>

namespace arx::x5 {
class X5Controller : public rclcpp::Node {
 public:
  X5Controller();
  void cleanup() {
    interfaces_ptr_.reset();
  }

  void CmdCallback(const arx5_arm_msg::msg::RobotCmd::SharedPtr msg);

  void PubState();

  void VrCmdCallback(const arm_control::msg::PosCmd::SharedPtr msg);

  void VrPubState();

  void FollowCmdCallback(const arx5_arm_msg::msg::RobotStatus::SharedPtr msg);
  void JointControlCallback(const arm_control::msg::JointControl::SharedPtr msg);
  void arxJoyCB(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
 private:
  std::shared_ptr<InterfacesThread> interfaces_ptr_;
  enum class CatchControlMode {
    kPosition,
    kTorque
  } catch_control_mode_;

  // 通常 & remote从机模式下
  rclcpp::Publisher<arx5_arm_msg::msg::RobotStatus>::SharedPtr joint_state_publisher_;
  // vr模式下
  rclcpp::Publisher<arm_control::msg::PosCmd>::SharedPtr vr_joint_state_publisher_;

  // 通常模式下
  rclcpp::Subscription<arx5_arm_msg::msg::RobotCmd>::SharedPtr joint_state_subscriber_;
  // vr模式下
  rclcpp::Subscription<arm_control::msg::PosCmd>::SharedPtr vr_joint_state_subscriber_;
  // remote从机模式下
  rclcpp::Subscription<arx5_arm_msg::msg::RobotStatus>::SharedPtr follow_joint_state_subscriber_;
  // joint control
  rclcpp::Subscription<arm_control::msg::JointControl>::SharedPtr joint_control_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr arx_joy_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
  bool new_version_ = false;
  std::vector<double> go_home_positions_ = {0, 0, 0, 0, 0, 0};
};
}

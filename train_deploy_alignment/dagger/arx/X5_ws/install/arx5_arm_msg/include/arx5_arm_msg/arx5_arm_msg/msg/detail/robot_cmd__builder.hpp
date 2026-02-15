// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from arx5_arm_msg:msg/RobotCmd.idl
// generated code does not contain a copyright notice

#ifndef ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__BUILDER_HPP_
#define ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "arx5_arm_msg/msg/detail/robot_cmd__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace arx5_arm_msg
{

namespace msg
{

namespace builder
{

class Init_RobotCmd_mode
{
public:
  explicit Init_RobotCmd_mode(::arx5_arm_msg::msg::RobotCmd & msg)
  : msg_(msg)
  {}
  ::arx5_arm_msg::msg::RobotCmd mode(::arx5_arm_msg::msg::RobotCmd::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotCmd msg_;
};

class Init_RobotCmd_gripper
{
public:
  explicit Init_RobotCmd_gripper(::arx5_arm_msg::msg::RobotCmd & msg)
  : msg_(msg)
  {}
  Init_RobotCmd_mode gripper(::arx5_arm_msg::msg::RobotCmd::_gripper_type arg)
  {
    msg_.gripper = std::move(arg);
    return Init_RobotCmd_mode(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotCmd msg_;
};

class Init_RobotCmd_joint_pos
{
public:
  explicit Init_RobotCmd_joint_pos(::arx5_arm_msg::msg::RobotCmd & msg)
  : msg_(msg)
  {}
  Init_RobotCmd_gripper joint_pos(::arx5_arm_msg::msg::RobotCmd::_joint_pos_type arg)
  {
    msg_.joint_pos = std::move(arg);
    return Init_RobotCmd_gripper(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotCmd msg_;
};

class Init_RobotCmd_end_pos
{
public:
  explicit Init_RobotCmd_end_pos(::arx5_arm_msg::msg::RobotCmd & msg)
  : msg_(msg)
  {}
  Init_RobotCmd_joint_pos end_pos(::arx5_arm_msg::msg::RobotCmd::_end_pos_type arg)
  {
    msg_.end_pos = std::move(arg);
    return Init_RobotCmd_joint_pos(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotCmd msg_;
};

class Init_RobotCmd_header
{
public:
  Init_RobotCmd_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotCmd_end_pos header(::arx5_arm_msg::msg::RobotCmd::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RobotCmd_end_pos(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotCmd msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::arx5_arm_msg::msg::RobotCmd>()
{
  return arx5_arm_msg::msg::builder::Init_RobotCmd_header();
}

}  // namespace arx5_arm_msg

#endif  // ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__BUILDER_HPP_

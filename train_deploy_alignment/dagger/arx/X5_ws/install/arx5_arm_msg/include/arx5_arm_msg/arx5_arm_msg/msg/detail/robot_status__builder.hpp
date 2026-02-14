// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from arx5_arm_msg:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_
#define ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "arx5_arm_msg/msg/detail/robot_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace arx5_arm_msg
{

namespace msg
{

namespace builder
{

class Init_RobotStatus_joint_cur
{
public:
  explicit Init_RobotStatus_joint_cur(::arx5_arm_msg::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  ::arx5_arm_msg::msg::RobotStatus joint_cur(::arx5_arm_msg::msg::RobotStatus::_joint_cur_type arg)
  {
    msg_.joint_cur = std::move(arg);
    return std::move(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotStatus msg_;
};

class Init_RobotStatus_joint_vel
{
public:
  explicit Init_RobotStatus_joint_vel(::arx5_arm_msg::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_joint_cur joint_vel(::arx5_arm_msg::msg::RobotStatus::_joint_vel_type arg)
  {
    msg_.joint_vel = std::move(arg);
    return Init_RobotStatus_joint_cur(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotStatus msg_;
};

class Init_RobotStatus_joint_pos
{
public:
  explicit Init_RobotStatus_joint_pos(::arx5_arm_msg::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_joint_vel joint_pos(::arx5_arm_msg::msg::RobotStatus::_joint_pos_type arg)
  {
    msg_.joint_pos = std::move(arg);
    return Init_RobotStatus_joint_vel(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotStatus msg_;
};

class Init_RobotStatus_end_pos
{
public:
  explicit Init_RobotStatus_end_pos(::arx5_arm_msg::msg::RobotStatus & msg)
  : msg_(msg)
  {}
  Init_RobotStatus_joint_pos end_pos(::arx5_arm_msg::msg::RobotStatus::_end_pos_type arg)
  {
    msg_.end_pos = std::move(arg);
    return Init_RobotStatus_joint_pos(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotStatus msg_;
};

class Init_RobotStatus_header
{
public:
  Init_RobotStatus_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStatus_end_pos header(::arx5_arm_msg::msg::RobotStatus::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RobotStatus_end_pos(msg_);
  }

private:
  ::arx5_arm_msg::msg::RobotStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::arx5_arm_msg::msg::RobotStatus>()
{
  return arx5_arm_msg::msg::builder::Init_RobotStatus_header();
}

}  // namespace arx5_arm_msg

#endif  // ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__BUILDER_HPP_

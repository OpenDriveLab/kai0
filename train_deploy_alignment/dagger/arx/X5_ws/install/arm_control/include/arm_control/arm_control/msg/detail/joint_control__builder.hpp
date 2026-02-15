// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__BUILDER_HPP_
#define ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "arm_control/msg/detail/joint_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace arm_control
{

namespace msg
{

namespace builder
{

class Init_JointControl_mode
{
public:
  explicit Init_JointControl_mode(::arm_control::msg::JointControl & msg)
  : msg_(msg)
  {}
  ::arm_control::msg::JointControl mode(::arm_control::msg::JointControl::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::arm_control::msg::JointControl msg_;
};

class Init_JointControl_joint_cur
{
public:
  explicit Init_JointControl_joint_cur(::arm_control::msg::JointControl & msg)
  : msg_(msg)
  {}
  Init_JointControl_mode joint_cur(::arm_control::msg::JointControl::_joint_cur_type arg)
  {
    msg_.joint_cur = std::move(arg);
    return Init_JointControl_mode(msg_);
  }

private:
  ::arm_control::msg::JointControl msg_;
};

class Init_JointControl_joint_vel
{
public:
  explicit Init_JointControl_joint_vel(::arm_control::msg::JointControl & msg)
  : msg_(msg)
  {}
  Init_JointControl_joint_cur joint_vel(::arm_control::msg::JointControl::_joint_vel_type arg)
  {
    msg_.joint_vel = std::move(arg);
    return Init_JointControl_joint_cur(msg_);
  }

private:
  ::arm_control::msg::JointControl msg_;
};

class Init_JointControl_joint_pos
{
public:
  Init_JointControl_joint_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointControl_joint_vel joint_pos(::arm_control::msg::JointControl::_joint_pos_type arg)
  {
    msg_.joint_pos = std::move(arg);
    return Init_JointControl_joint_vel(msg_);
  }

private:
  ::arm_control::msg::JointControl msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::arm_control::msg::JointControl>()
{
  return arm_control::msg::builder::Init_JointControl_joint_pos();
}

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__BUILDER_HPP_

// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__ARX_IMU__BUILDER_HPP_
#define ARM_CONTROL__MSG__DETAIL__ARX_IMU__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "arm_control/msg/detail/arx_imu__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace arm_control
{

namespace msg
{

namespace builder
{

class Init_ArxImu_angular_velocity
{
public:
  explicit Init_ArxImu_angular_velocity(::arm_control::msg::ArxImu & msg)
  : msg_(msg)
  {}
  ::arm_control::msg::ArxImu angular_velocity(::arm_control::msg::ArxImu::_angular_velocity_type arg)
  {
    msg_.angular_velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::arm_control::msg::ArxImu msg_;
};

class Init_ArxImu_yaw
{
public:
  explicit Init_ArxImu_yaw(::arm_control::msg::ArxImu & msg)
  : msg_(msg)
  {}
  Init_ArxImu_angular_velocity yaw(::arm_control::msg::ArxImu::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_ArxImu_angular_velocity(msg_);
  }

private:
  ::arm_control::msg::ArxImu msg_;
};

class Init_ArxImu_pitch
{
public:
  explicit Init_ArxImu_pitch(::arm_control::msg::ArxImu & msg)
  : msg_(msg)
  {}
  Init_ArxImu_yaw pitch(::arm_control::msg::ArxImu::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_ArxImu_yaw(msg_);
  }

private:
  ::arm_control::msg::ArxImu msg_;
};

class Init_ArxImu_roll
{
public:
  explicit Init_ArxImu_roll(::arm_control::msg::ArxImu & msg)
  : msg_(msg)
  {}
  Init_ArxImu_pitch roll(::arm_control::msg::ArxImu::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_ArxImu_pitch(msg_);
  }

private:
  ::arm_control::msg::ArxImu msg_;
};

class Init_ArxImu_stamp
{
public:
  Init_ArxImu_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ArxImu_roll stamp(::arm_control::msg::ArxImu::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_ArxImu_roll(msg_);
  }

private:
  ::arm_control::msg::ArxImu msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::arm_control::msg::ArxImu>()
{
  return arm_control::msg::builder::Init_ArxImu_stamp();
}

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__ARX_IMU__BUILDER_HPP_

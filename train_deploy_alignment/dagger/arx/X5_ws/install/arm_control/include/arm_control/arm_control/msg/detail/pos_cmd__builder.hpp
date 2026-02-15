// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from arm_control:msg/PosCmd.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__POS_CMD__BUILDER_HPP_
#define ARM_CONTROL__MSG__DETAIL__POS_CMD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "arm_control/msg/detail/pos_cmd__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace arm_control
{

namespace msg
{

namespace builder
{

class Init_PosCmd_time_count
{
public:
  explicit Init_PosCmd_time_count(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  ::arm_control::msg::PosCmd time_count(::arm_control::msg::PosCmd::_time_count_type arg)
  {
    msg_.time_count = std::move(arg);
    return std::move(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_mode2
{
public:
  explicit Init_PosCmd_mode2(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_time_count mode2(::arm_control::msg::PosCmd::_mode2_type arg)
  {
    msg_.mode2 = std::move(arg);
    return Init_PosCmd_time_count(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_mode1
{
public:
  explicit Init_PosCmd_mode1(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_mode2 mode1(::arm_control::msg::PosCmd::_mode1_type arg)
  {
    msg_.mode1 = std::move(arg);
    return Init_PosCmd_mode2(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_temp_int_data
{
public:
  explicit Init_PosCmd_temp_int_data(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_mode1 temp_int_data(::arm_control::msg::PosCmd::_temp_int_data_type arg)
  {
    msg_.temp_int_data = std::move(arg);
    return Init_PosCmd_mode1(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_temp_float_data
{
public:
  explicit Init_PosCmd_temp_float_data(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_temp_int_data temp_float_data(::arm_control::msg::PosCmd::_temp_float_data_type arg)
  {
    msg_.temp_float_data = std::move(arg);
    return Init_PosCmd_temp_int_data(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_head_yaw
{
public:
  explicit Init_PosCmd_head_yaw(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_temp_float_data head_yaw(::arm_control::msg::PosCmd::_head_yaw_type arg)
  {
    msg_.head_yaw = std::move(arg);
    return Init_PosCmd_temp_float_data(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_head_pit
{
public:
  explicit Init_PosCmd_head_pit(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_head_yaw head_pit(::arm_control::msg::PosCmd::_head_pit_type arg)
  {
    msg_.head_pit = std::move(arg);
    return Init_PosCmd_head_yaw(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_height
{
public:
  explicit Init_PosCmd_height(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_head_pit height(::arm_control::msg::PosCmd::_height_type arg)
  {
    msg_.height = std::move(arg);
    return Init_PosCmd_head_pit(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_vel_r
{
public:
  explicit Init_PosCmd_vel_r(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_height vel_r(::arm_control::msg::PosCmd::_vel_r_type arg)
  {
    msg_.vel_r = std::move(arg);
    return Init_PosCmd_height(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_vel_l
{
public:
  explicit Init_PosCmd_vel_l(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_vel_r vel_l(::arm_control::msg::PosCmd::_vel_l_type arg)
  {
    msg_.vel_l = std::move(arg);
    return Init_PosCmd_vel_r(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_chz
{
public:
  explicit Init_PosCmd_chz(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_vel_l chz(::arm_control::msg::PosCmd::_chz_type arg)
  {
    msg_.chz = std::move(arg);
    return Init_PosCmd_vel_l(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_chy
{
public:
  explicit Init_PosCmd_chy(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_chz chy(::arm_control::msg::PosCmd::_chy_type arg)
  {
    msg_.chy = std::move(arg);
    return Init_PosCmd_chz(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_chx
{
public:
  explicit Init_PosCmd_chx(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_chy chx(::arm_control::msg::PosCmd::_chx_type arg)
  {
    msg_.chx = std::move(arg);
    return Init_PosCmd_chy(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_quater_w
{
public:
  explicit Init_PosCmd_quater_w(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_chx quater_w(::arm_control::msg::PosCmd::_quater_w_type arg)
  {
    msg_.quater_w = std::move(arg);
    return Init_PosCmd_chx(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_quater_z
{
public:
  explicit Init_PosCmd_quater_z(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_quater_w quater_z(::arm_control::msg::PosCmd::_quater_z_type arg)
  {
    msg_.quater_z = std::move(arg);
    return Init_PosCmd_quater_w(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_quater_y
{
public:
  explicit Init_PosCmd_quater_y(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_quater_z quater_y(::arm_control::msg::PosCmd::_quater_y_type arg)
  {
    msg_.quater_y = std::move(arg);
    return Init_PosCmd_quater_z(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_quater_x
{
public:
  explicit Init_PosCmd_quater_x(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_quater_y quater_x(::arm_control::msg::PosCmd::_quater_x_type arg)
  {
    msg_.quater_x = std::move(arg);
    return Init_PosCmd_quater_y(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_gripper
{
public:
  explicit Init_PosCmd_gripper(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_quater_x gripper(::arm_control::msg::PosCmd::_gripper_type arg)
  {
    msg_.gripper = std::move(arg);
    return Init_PosCmd_quater_x(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_yaw
{
public:
  explicit Init_PosCmd_yaw(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_gripper yaw(::arm_control::msg::PosCmd::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_PosCmd_gripper(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_pitch
{
public:
  explicit Init_PosCmd_pitch(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_yaw pitch(::arm_control::msg::PosCmd::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_PosCmd_yaw(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_roll
{
public:
  explicit Init_PosCmd_roll(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_pitch roll(::arm_control::msg::PosCmd::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_PosCmd_pitch(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_z
{
public:
  explicit Init_PosCmd_z(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_roll z(::arm_control::msg::PosCmd::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_PosCmd_roll(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_y
{
public:
  explicit Init_PosCmd_y(::arm_control::msg::PosCmd & msg)
  : msg_(msg)
  {}
  Init_PosCmd_z y(::arm_control::msg::PosCmd::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_PosCmd_z(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

class Init_PosCmd_x
{
public:
  Init_PosCmd_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PosCmd_y x(::arm_control::msg::PosCmd::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_PosCmd_y(msg_);
  }

private:
  ::arm_control::msg::PosCmd msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::arm_control::msg::PosCmd>()
{
  return arm_control::msg::builder::Init_PosCmd_x();
}

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__POS_CMD__BUILDER_HPP_

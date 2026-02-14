// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from arm_control:msg/PosCmd.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__POS_CMD__STRUCT_HPP_
#define ARM_CONTROL__MSG__DETAIL__POS_CMD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__arm_control__msg__PosCmd __attribute__((deprecated))
#else
# define DEPRECATED__arm_control__msg__PosCmd __declspec(deprecated)
#endif

namespace arm_control
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PosCmd_
{
  using Type = PosCmd_<ContainerAllocator>;

  explicit PosCmd_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->gripper = 0.0;
      this->quater_x = 0.0;
      this->quater_y = 0.0;
      this->quater_z = 0.0;
      this->quater_w = 0.0;
      this->chx = 0.0;
      this->chy = 0.0;
      this->chz = 0.0;
      this->vel_l = 0.0;
      this->vel_r = 0.0;
      this->height = 0.0;
      this->head_pit = 0.0;
      this->head_yaw = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->temp_float_data.begin(), this->temp_float_data.end(), 0.0);
      std::fill<typename std::array<int32_t, 6>::iterator, int32_t>(this->temp_int_data.begin(), this->temp_int_data.end(), 0l);
      this->mode1 = 0l;
      this->mode2 = 0l;
      this->time_count = 0l;
    }
  }

  explicit PosCmd_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : temp_float_data(_alloc),
    temp_int_data(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
      this->gripper = 0.0;
      this->quater_x = 0.0;
      this->quater_y = 0.0;
      this->quater_z = 0.0;
      this->quater_w = 0.0;
      this->chx = 0.0;
      this->chy = 0.0;
      this->chz = 0.0;
      this->vel_l = 0.0;
      this->vel_r = 0.0;
      this->height = 0.0;
      this->head_pit = 0.0;
      this->head_yaw = 0.0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->temp_float_data.begin(), this->temp_float_data.end(), 0.0);
      std::fill<typename std::array<int32_t, 6>::iterator, int32_t>(this->temp_int_data.begin(), this->temp_int_data.end(), 0l);
      this->mode1 = 0l;
      this->mode2 = 0l;
      this->time_count = 0l;
    }
  }

  // field types and members
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _z_type =
    double;
  _z_type z;
  using _roll_type =
    double;
  _roll_type roll;
  using _pitch_type =
    double;
  _pitch_type pitch;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _gripper_type =
    double;
  _gripper_type gripper;
  using _quater_x_type =
    double;
  _quater_x_type quater_x;
  using _quater_y_type =
    double;
  _quater_y_type quater_y;
  using _quater_z_type =
    double;
  _quater_z_type quater_z;
  using _quater_w_type =
    double;
  _quater_w_type quater_w;
  using _chx_type =
    double;
  _chx_type chx;
  using _chy_type =
    double;
  _chy_type chy;
  using _chz_type =
    double;
  _chz_type chz;
  using _vel_l_type =
    double;
  _vel_l_type vel_l;
  using _vel_r_type =
    double;
  _vel_r_type vel_r;
  using _height_type =
    double;
  _height_type height;
  using _head_pit_type =
    double;
  _head_pit_type head_pit;
  using _head_yaw_type =
    double;
  _head_yaw_type head_yaw;
  using _temp_float_data_type =
    std::array<double, 6>;
  _temp_float_data_type temp_float_data;
  using _temp_int_data_type =
    std::array<int32_t, 6>;
  _temp_int_data_type temp_int_data;
  using _mode1_type =
    int32_t;
  _mode1_type mode1;
  using _mode2_type =
    int32_t;
  _mode2_type mode2;
  using _time_count_type =
    int32_t;
  _time_count_type time_count;

  // setters for named parameter idiom
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const double & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__roll(
    const double & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__pitch(
    const double & _arg)
  {
    this->pitch = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__gripper(
    const double & _arg)
  {
    this->gripper = _arg;
    return *this;
  }
  Type & set__quater_x(
    const double & _arg)
  {
    this->quater_x = _arg;
    return *this;
  }
  Type & set__quater_y(
    const double & _arg)
  {
    this->quater_y = _arg;
    return *this;
  }
  Type & set__quater_z(
    const double & _arg)
  {
    this->quater_z = _arg;
    return *this;
  }
  Type & set__quater_w(
    const double & _arg)
  {
    this->quater_w = _arg;
    return *this;
  }
  Type & set__chx(
    const double & _arg)
  {
    this->chx = _arg;
    return *this;
  }
  Type & set__chy(
    const double & _arg)
  {
    this->chy = _arg;
    return *this;
  }
  Type & set__chz(
    const double & _arg)
  {
    this->chz = _arg;
    return *this;
  }
  Type & set__vel_l(
    const double & _arg)
  {
    this->vel_l = _arg;
    return *this;
  }
  Type & set__vel_r(
    const double & _arg)
  {
    this->vel_r = _arg;
    return *this;
  }
  Type & set__height(
    const double & _arg)
  {
    this->height = _arg;
    return *this;
  }
  Type & set__head_pit(
    const double & _arg)
  {
    this->head_pit = _arg;
    return *this;
  }
  Type & set__head_yaw(
    const double & _arg)
  {
    this->head_yaw = _arg;
    return *this;
  }
  Type & set__temp_float_data(
    const std::array<double, 6> & _arg)
  {
    this->temp_float_data = _arg;
    return *this;
  }
  Type & set__temp_int_data(
    const std::array<int32_t, 6> & _arg)
  {
    this->temp_int_data = _arg;
    return *this;
  }
  Type & set__mode1(
    const int32_t & _arg)
  {
    this->mode1 = _arg;
    return *this;
  }
  Type & set__mode2(
    const int32_t & _arg)
  {
    this->mode2 = _arg;
    return *this;
  }
  Type & set__time_count(
    const int32_t & _arg)
  {
    this->time_count = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    arm_control::msg::PosCmd_<ContainerAllocator> *;
  using ConstRawPtr =
    const arm_control::msg::PosCmd_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<arm_control::msg::PosCmd_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<arm_control::msg::PosCmd_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::PosCmd_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::PosCmd_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::PosCmd_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::PosCmd_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<arm_control::msg::PosCmd_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<arm_control::msg::PosCmd_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__arm_control__msg__PosCmd
    std::shared_ptr<arm_control::msg::PosCmd_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__arm_control__msg__PosCmd
    std::shared_ptr<arm_control::msg::PosCmd_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PosCmd_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->pitch != other.pitch) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->gripper != other.gripper) {
      return false;
    }
    if (this->quater_x != other.quater_x) {
      return false;
    }
    if (this->quater_y != other.quater_y) {
      return false;
    }
    if (this->quater_z != other.quater_z) {
      return false;
    }
    if (this->quater_w != other.quater_w) {
      return false;
    }
    if (this->chx != other.chx) {
      return false;
    }
    if (this->chy != other.chy) {
      return false;
    }
    if (this->chz != other.chz) {
      return false;
    }
    if (this->vel_l != other.vel_l) {
      return false;
    }
    if (this->vel_r != other.vel_r) {
      return false;
    }
    if (this->height != other.height) {
      return false;
    }
    if (this->head_pit != other.head_pit) {
      return false;
    }
    if (this->head_yaw != other.head_yaw) {
      return false;
    }
    if (this->temp_float_data != other.temp_float_data) {
      return false;
    }
    if (this->temp_int_data != other.temp_int_data) {
      return false;
    }
    if (this->mode1 != other.mode1) {
      return false;
    }
    if (this->mode2 != other.mode2) {
      return false;
    }
    if (this->time_count != other.time_count) {
      return false;
    }
    return true;
  }
  bool operator!=(const PosCmd_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PosCmd_

// alias to use template instance with default allocator
using PosCmd =
  arm_control::msg::PosCmd_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__POS_CMD__STRUCT_HPP_

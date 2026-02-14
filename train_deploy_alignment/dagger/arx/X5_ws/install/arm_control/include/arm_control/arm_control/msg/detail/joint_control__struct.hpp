// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_HPP_
#define ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__arm_control__msg__JointControl __attribute__((deprecated))
#else
# define DEPRECATED__arm_control__msg__JointControl __declspec(deprecated)
#endif

namespace arm_control
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct JointControl_
{
  using Type = JointControl_<ContainerAllocator>;

  explicit JointControl_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_pos.begin(), this->joint_pos.end(), 0.0f);
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_vel.begin(), this->joint_vel.end(), 0.0f);
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_cur.begin(), this->joint_cur.end(), 0.0f);
      this->mode = 0l;
    }
  }

  explicit JointControl_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : joint_pos(_alloc),
    joint_vel(_alloc),
    joint_cur(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_pos.begin(), this->joint_pos.end(), 0.0f);
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_vel.begin(), this->joint_vel.end(), 0.0f);
      std::fill<typename std::array<float, 8>::iterator, float>(this->joint_cur.begin(), this->joint_cur.end(), 0.0f);
      this->mode = 0l;
    }
  }

  // field types and members
  using _joint_pos_type =
    std::array<float, 8>;
  _joint_pos_type joint_pos;
  using _joint_vel_type =
    std::array<float, 8>;
  _joint_vel_type joint_vel;
  using _joint_cur_type =
    std::array<float, 8>;
  _joint_cur_type joint_cur;
  using _mode_type =
    int32_t;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__joint_pos(
    const std::array<float, 8> & _arg)
  {
    this->joint_pos = _arg;
    return *this;
  }
  Type & set__joint_vel(
    const std::array<float, 8> & _arg)
  {
    this->joint_vel = _arg;
    return *this;
  }
  Type & set__joint_cur(
    const std::array<float, 8> & _arg)
  {
    this->joint_cur = _arg;
    return *this;
  }
  Type & set__mode(
    const int32_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    arm_control::msg::JointControl_<ContainerAllocator> *;
  using ConstRawPtr =
    const arm_control::msg::JointControl_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<arm_control::msg::JointControl_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<arm_control::msg::JointControl_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::JointControl_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::JointControl_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::JointControl_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::JointControl_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<arm_control::msg::JointControl_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<arm_control::msg::JointControl_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__arm_control__msg__JointControl
    std::shared_ptr<arm_control::msg::JointControl_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__arm_control__msg__JointControl
    std::shared_ptr<arm_control::msg::JointControl_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointControl_ & other) const
  {
    if (this->joint_pos != other.joint_pos) {
      return false;
    }
    if (this->joint_vel != other.joint_vel) {
      return false;
    }
    if (this->joint_cur != other.joint_cur) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointControl_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointControl_

// alias to use template instance with default allocator
using JointControl =
  arm_control::msg::JointControl_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_HPP_

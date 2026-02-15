// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from arx5_arm_msg:msg/RobotCmd.idl
// generated code does not contain a copyright notice

#ifndef ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__STRUCT_HPP_
#define ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__arx5_arm_msg__msg__RobotCmd __attribute__((deprecated))
#else
# define DEPRECATED__arx5_arm_msg__msg__RobotCmd __declspec(deprecated)
#endif

namespace arx5_arm_msg
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RobotCmd_
{
  using Type = RobotCmd_<ContainerAllocator>;

  explicit RobotCmd_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->end_pos.begin(), this->end_pos.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_pos.begin(), this->joint_pos.end(), 0.0);
      this->gripper = 0.0;
      this->mode = 0ll;
    }
  }

  explicit RobotCmd_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    end_pos(_alloc),
    joint_pos(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->end_pos.begin(), this->end_pos.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_pos.begin(), this->joint_pos.end(), 0.0);
      this->gripper = 0.0;
      this->mode = 0ll;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _end_pos_type =
    std::array<double, 6>;
  _end_pos_type end_pos;
  using _joint_pos_type =
    std::array<double, 6>;
  _joint_pos_type joint_pos;
  using _gripper_type =
    double;
  _gripper_type gripper;
  using _mode_type =
    int64_t;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__end_pos(
    const std::array<double, 6> & _arg)
  {
    this->end_pos = _arg;
    return *this;
  }
  Type & set__joint_pos(
    const std::array<double, 6> & _arg)
  {
    this->joint_pos = _arg;
    return *this;
  }
  Type & set__gripper(
    const double & _arg)
  {
    this->gripper = _arg;
    return *this;
  }
  Type & set__mode(
    const int64_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> *;
  using ConstRawPtr =
    const arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__arx5_arm_msg__msg__RobotCmd
    std::shared_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__arx5_arm_msg__msg__RobotCmd
    std::shared_ptr<arx5_arm_msg::msg::RobotCmd_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RobotCmd_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->end_pos != other.end_pos) {
      return false;
    }
    if (this->joint_pos != other.joint_pos) {
      return false;
    }
    if (this->gripper != other.gripper) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const RobotCmd_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RobotCmd_

// alias to use template instance with default allocator
using RobotCmd =
  arx5_arm_msg::msg::RobotCmd_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace arx5_arm_msg

#endif  // ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__STRUCT_HPP_

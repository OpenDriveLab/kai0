// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_HPP_
#define ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__arm_control__msg__ArxImu __attribute__((deprecated))
#else
# define DEPRECATED__arm_control__msg__ArxImu __declspec(deprecated)
#endif

namespace arm_control
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ArxImu_
{
  using Type = ArxImu_<ContainerAllocator>;

  explicit ArxImu_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    angular_velocity(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
    }
  }

  explicit ArxImu_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    angular_velocity(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->roll = 0.0;
      this->pitch = 0.0;
      this->yaw = 0.0;
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _roll_type =
    double;
  _roll_type roll;
  using _pitch_type =
    double;
  _pitch_type pitch;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _angular_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _angular_velocity_type angular_velocity;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
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
  Type & set__angular_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->angular_velocity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    arm_control::msg::ArxImu_<ContainerAllocator> *;
  using ConstRawPtr =
    const arm_control::msg::ArxImu_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<arm_control::msg::ArxImu_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<arm_control::msg::ArxImu_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::ArxImu_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::ArxImu_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      arm_control::msg::ArxImu_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<arm_control::msg::ArxImu_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<arm_control::msg::ArxImu_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<arm_control::msg::ArxImu_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__arm_control__msg__ArxImu
    std::shared_ptr<arm_control::msg::ArxImu_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__arm_control__msg__ArxImu
    std::shared_ptr<arm_control::msg::ArxImu_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ArxImu_ & other) const
  {
    if (this->stamp != other.stamp) {
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
    if (this->angular_velocity != other.angular_velocity) {
      return false;
    }
    return true;
  }
  bool operator!=(const ArxImu_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ArxImu_

// alias to use template instance with default allocator
using ArxImu =
  arm_control::msg::ArxImu_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace arm_control

#endif  // ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_HPP_

// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__ARX_IMU__TRAITS_HPP_
#define ARM_CONTROL__MSG__DETAIL__ARX_IMU__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "arm_control/msg/detail/arx_imu__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace arm_control
{

namespace msg
{

inline void to_flow_style_yaml(
  const ArxImu & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: roll
  {
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << ", ";
  }

  // member: pitch
  {
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << ", ";
  }

  // member: angular_velocity
  {
    out << "angular_velocity: ";
    to_flow_style_yaml(msg.angular_velocity, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ArxImu & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    rosidl_generator_traits::value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    rosidl_generator_traits::value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw, out);
    out << "\n";
  }

  // member: angular_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular_velocity:\n";
    to_block_style_yaml(msg.angular_velocity, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ArxImu & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace arm_control

namespace rosidl_generator_traits
{

[[deprecated("use arm_control::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const arm_control::msg::ArxImu & msg,
  std::ostream & out, size_t indentation = 0)
{
  arm_control::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use arm_control::msg::to_yaml() instead")]]
inline std::string to_yaml(const arm_control::msg::ArxImu & msg)
{
  return arm_control::msg::to_yaml(msg);
}

template<>
inline const char * data_type<arm_control::msg::ArxImu>()
{
  return "arm_control::msg::ArxImu";
}

template<>
inline const char * name<arm_control::msg::ArxImu>()
{
  return "arm_control/msg/ArxImu";
}

template<>
struct has_fixed_size<arm_control::msg::ArxImu>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct has_bounded_size<arm_control::msg::ArxImu>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct is_message<arm_control::msg::ArxImu>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ARM_CONTROL__MSG__DETAIL__ARX_IMU__TRAITS_HPP_

// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from arx5_arm_msg:msg/RobotCmd.idl
// generated code does not contain a copyright notice

#ifndef ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__TRAITS_HPP_
#define ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "arx5_arm_msg/msg/detail/robot_cmd__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace arx5_arm_msg
{

namespace msg
{

inline void to_flow_style_yaml(
  const RobotCmd & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: end_pos
  {
    if (msg.end_pos.size() == 0) {
      out << "end_pos: []";
    } else {
      out << "end_pos: [";
      size_t pending_items = msg.end_pos.size();
      for (auto item : msg.end_pos) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_pos
  {
    if (msg.joint_pos.size() == 0) {
      out << "joint_pos: []";
    } else {
      out << "joint_pos: [";
      size_t pending_items = msg.joint_pos.size();
      for (auto item : msg.joint_pos) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: gripper
  {
    out << "gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.gripper, out);
    out << ", ";
  }

  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotCmd & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: end_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.end_pos.size() == 0) {
      out << "end_pos: []\n";
    } else {
      out << "end_pos:\n";
      for (auto item : msg.end_pos) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_pos.size() == 0) {
      out << "joint_pos: []\n";
    } else {
      out << "joint_pos:\n";
      for (auto item : msg.joint_pos) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: gripper
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.gripper, out);
    out << "\n";
  }

  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotCmd & msg, bool use_flow_style = false)
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

}  // namespace arx5_arm_msg

namespace rosidl_generator_traits
{

[[deprecated("use arx5_arm_msg::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const arx5_arm_msg::msg::RobotCmd & msg,
  std::ostream & out, size_t indentation = 0)
{
  arx5_arm_msg::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use arx5_arm_msg::msg::to_yaml() instead")]]
inline std::string to_yaml(const arx5_arm_msg::msg::RobotCmd & msg)
{
  return arx5_arm_msg::msg::to_yaml(msg);
}

template<>
inline const char * data_type<arx5_arm_msg::msg::RobotCmd>()
{
  return "arx5_arm_msg::msg::RobotCmd";
}

template<>
inline const char * name<arx5_arm_msg::msg::RobotCmd>()
{
  return "arx5_arm_msg/msg/RobotCmd";
}

template<>
struct has_fixed_size<arx5_arm_msg::msg::RobotCmd>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<arx5_arm_msg::msg::RobotCmd>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<arx5_arm_msg::msg::RobotCmd>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ARX5_ARM_MSG__MSG__DETAIL__ROBOT_CMD__TRAITS_HPP_

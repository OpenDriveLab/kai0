// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__TRAITS_HPP_
#define ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "arm_control/msg/detail/joint_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace arm_control
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointControl & msg,
  std::ostream & out)
{
  out << "{";
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

  // member: joint_vel
  {
    if (msg.joint_vel.size() == 0) {
      out << "joint_vel: []";
    } else {
      out << "joint_vel: [";
      size_t pending_items = msg.joint_vel.size();
      for (auto item : msg.joint_vel) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_cur
  {
    if (msg.joint_cur.size() == 0) {
      out << "joint_cur: []";
    } else {
      out << "joint_cur: [";
      size_t pending_items = msg.joint_cur.size();
      for (auto item : msg.joint_cur) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
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
  const JointControl & msg,
  std::ostream & out, size_t indentation = 0)
{
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

  // member: joint_vel
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_vel.size() == 0) {
      out << "joint_vel: []\n";
    } else {
      out << "joint_vel:\n";
      for (auto item : msg.joint_vel) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_cur
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_cur.size() == 0) {
      out << "joint_cur: []\n";
    } else {
      out << "joint_cur:\n";
      for (auto item : msg.joint_cur) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
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

inline std::string to_yaml(const JointControl & msg, bool use_flow_style = false)
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
  const arm_control::msg::JointControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  arm_control::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use arm_control::msg::to_yaml() instead")]]
inline std::string to_yaml(const arm_control::msg::JointControl & msg)
{
  return arm_control::msg::to_yaml(msg);
}

template<>
inline const char * data_type<arm_control::msg::JointControl>()
{
  return "arm_control::msg::JointControl";
}

template<>
inline const char * name<arm_control::msg::JointControl>()
{
  return "arm_control/msg/JointControl";
}

template<>
struct has_fixed_size<arm_control::msg::JointControl>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<arm_control::msg::JointControl>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<arm_control::msg::JointControl>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__TRAITS_HPP_

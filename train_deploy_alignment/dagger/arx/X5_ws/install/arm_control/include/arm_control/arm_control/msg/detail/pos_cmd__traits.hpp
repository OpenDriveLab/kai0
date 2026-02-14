// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from arm_control:msg/PosCmd.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__POS_CMD__TRAITS_HPP_
#define ARM_CONTROL__MSG__DETAIL__POS_CMD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "arm_control/msg/detail/pos_cmd__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace arm_control
{

namespace msg
{

inline void to_flow_style_yaml(
  const PosCmd & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
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

  // member: gripper
  {
    out << "gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.gripper, out);
    out << ", ";
  }

  // member: quater_x
  {
    out << "quater_x: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_x, out);
    out << ", ";
  }

  // member: quater_y
  {
    out << "quater_y: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_y, out);
    out << ", ";
  }

  // member: quater_z
  {
    out << "quater_z: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_z, out);
    out << ", ";
  }

  // member: quater_w
  {
    out << "quater_w: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_w, out);
    out << ", ";
  }

  // member: chx
  {
    out << "chx: ";
    rosidl_generator_traits::value_to_yaml(msg.chx, out);
    out << ", ";
  }

  // member: chy
  {
    out << "chy: ";
    rosidl_generator_traits::value_to_yaml(msg.chy, out);
    out << ", ";
  }

  // member: chz
  {
    out << "chz: ";
    rosidl_generator_traits::value_to_yaml(msg.chz, out);
    out << ", ";
  }

  // member: vel_l
  {
    out << "vel_l: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_l, out);
    out << ", ";
  }

  // member: vel_r
  {
    out << "vel_r: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_r, out);
    out << ", ";
  }

  // member: height
  {
    out << "height: ";
    rosidl_generator_traits::value_to_yaml(msg.height, out);
    out << ", ";
  }

  // member: head_pit
  {
    out << "head_pit: ";
    rosidl_generator_traits::value_to_yaml(msg.head_pit, out);
    out << ", ";
  }

  // member: head_yaw
  {
    out << "head_yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.head_yaw, out);
    out << ", ";
  }

  // member: temp_float_data
  {
    if (msg.temp_float_data.size() == 0) {
      out << "temp_float_data: []";
    } else {
      out << "temp_float_data: [";
      size_t pending_items = msg.temp_float_data.size();
      for (auto item : msg.temp_float_data) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: temp_int_data
  {
    if (msg.temp_int_data.size() == 0) {
      out << "temp_int_data: []";
    } else {
      out << "temp_int_data: [";
      size_t pending_items = msg.temp_int_data.size();
      for (auto item : msg.temp_int_data) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: mode1
  {
    out << "mode1: ";
    rosidl_generator_traits::value_to_yaml(msg.mode1, out);
    out << ", ";
  }

  // member: mode2
  {
    out << "mode2: ";
    rosidl_generator_traits::value_to_yaml(msg.mode2, out);
    out << ", ";
  }

  // member: time_count
  {
    out << "time_count: ";
    rosidl_generator_traits::value_to_yaml(msg.time_count, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PosCmd & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
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

  // member: gripper
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.gripper, out);
    out << "\n";
  }

  // member: quater_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quater_x: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_x, out);
    out << "\n";
  }

  // member: quater_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quater_y: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_y, out);
    out << "\n";
  }

  // member: quater_z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quater_z: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_z, out);
    out << "\n";
  }

  // member: quater_w
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "quater_w: ";
    rosidl_generator_traits::value_to_yaml(msg.quater_w, out);
    out << "\n";
  }

  // member: chx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "chx: ";
    rosidl_generator_traits::value_to_yaml(msg.chx, out);
    out << "\n";
  }

  // member: chy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "chy: ";
    rosidl_generator_traits::value_to_yaml(msg.chy, out);
    out << "\n";
  }

  // member: chz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "chz: ";
    rosidl_generator_traits::value_to_yaml(msg.chz, out);
    out << "\n";
  }

  // member: vel_l
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel_l: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_l, out);
    out << "\n";
  }

  // member: vel_r
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vel_r: ";
    rosidl_generator_traits::value_to_yaml(msg.vel_r, out);
    out << "\n";
  }

  // member: height
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "height: ";
    rosidl_generator_traits::value_to_yaml(msg.height, out);
    out << "\n";
  }

  // member: head_pit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "head_pit: ";
    rosidl_generator_traits::value_to_yaml(msg.head_pit, out);
    out << "\n";
  }

  // member: head_yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "head_yaw: ";
    rosidl_generator_traits::value_to_yaml(msg.head_yaw, out);
    out << "\n";
  }

  // member: temp_float_data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.temp_float_data.size() == 0) {
      out << "temp_float_data: []\n";
    } else {
      out << "temp_float_data:\n";
      for (auto item : msg.temp_float_data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: temp_int_data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.temp_int_data.size() == 0) {
      out << "temp_int_data: []\n";
    } else {
      out << "temp_int_data:\n";
      for (auto item : msg.temp_int_data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: mode1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode1: ";
    rosidl_generator_traits::value_to_yaml(msg.mode1, out);
    out << "\n";
  }

  // member: mode2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode2: ";
    rosidl_generator_traits::value_to_yaml(msg.mode2, out);
    out << "\n";
  }

  // member: time_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_count: ";
    rosidl_generator_traits::value_to_yaml(msg.time_count, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PosCmd & msg, bool use_flow_style = false)
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
  const arm_control::msg::PosCmd & msg,
  std::ostream & out, size_t indentation = 0)
{
  arm_control::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use arm_control::msg::to_yaml() instead")]]
inline std::string to_yaml(const arm_control::msg::PosCmd & msg)
{
  return arm_control::msg::to_yaml(msg);
}

template<>
inline const char * data_type<arm_control::msg::PosCmd>()
{
  return "arm_control::msg::PosCmd";
}

template<>
inline const char * name<arm_control::msg::PosCmd>()
{
  return "arm_control/msg/PosCmd";
}

template<>
struct has_fixed_size<arm_control::msg::PosCmd>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<arm_control::msg::PosCmd>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<arm_control::msg::PosCmd>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ARM_CONTROL__MSG__DETAIL__POS_CMD__TRAITS_HPP_

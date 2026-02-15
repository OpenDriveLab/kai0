// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from arx5_arm_msg:msg/RobotCmd.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "arx5_arm_msg/msg/detail/robot_cmd__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace arx5_arm_msg
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void RobotCmd_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) arx5_arm_msg::msg::RobotCmd(_init);
}

void RobotCmd_fini_function(void * message_memory)
{
  auto typed_message = static_cast<arx5_arm_msg::msg::RobotCmd *>(message_memory);
  typed_message->~RobotCmd();
}

size_t size_function__RobotCmd__end_pos(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotCmd__end_pos(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotCmd__end_pos(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotCmd__end_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotCmd__end_pos(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotCmd__end_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotCmd__end_pos(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__RobotCmd__joint_pos(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__RobotCmd__joint_pos(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__RobotCmd__joint_pos(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__RobotCmd__joint_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__RobotCmd__joint_pos(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__RobotCmd__joint_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__RobotCmd__joint_pos(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RobotCmd_message_member_array[5] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg::msg::RobotCmd, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "end_pos",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg::msg::RobotCmd, end_pos),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotCmd__end_pos,  // size() function pointer
    get_const_function__RobotCmd__end_pos,  // get_const(index) function pointer
    get_function__RobotCmd__end_pos,  // get(index) function pointer
    fetch_function__RobotCmd__end_pos,  // fetch(index, &value) function pointer
    assign_function__RobotCmd__end_pos,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joint_pos",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg::msg::RobotCmd, joint_pos),  // bytes offset in struct
    nullptr,  // default value
    size_function__RobotCmd__joint_pos,  // size() function pointer
    get_const_function__RobotCmd__joint_pos,  // get_const(index) function pointer
    get_function__RobotCmd__joint_pos,  // get(index) function pointer
    fetch_function__RobotCmd__joint_pos,  // fetch(index, &value) function pointer
    assign_function__RobotCmd__joint_pos,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "gripper",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg::msg::RobotCmd, gripper),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mode",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg::msg::RobotCmd, mode),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RobotCmd_message_members = {
  "arx5_arm_msg::msg",  // message namespace
  "RobotCmd",  // message name
  5,  // number of fields
  sizeof(arx5_arm_msg::msg::RobotCmd),
  RobotCmd_message_member_array,  // message members
  RobotCmd_init_function,  // function to initialize message memory (memory has to be allocated)
  RobotCmd_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RobotCmd_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RobotCmd_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace arx5_arm_msg


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<arx5_arm_msg::msg::RobotCmd>()
{
  return &::arx5_arm_msg::msg::rosidl_typesupport_introspection_cpp::RobotCmd_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, arx5_arm_msg, msg, RobotCmd)() {
  return &::arx5_arm_msg::msg::rosidl_typesupport_introspection_cpp::RobotCmd_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

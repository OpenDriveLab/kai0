// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "arm_control/msg/detail/joint_control__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace arm_control
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void JointControl_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) arm_control::msg::JointControl(_init);
}

void JointControl_fini_function(void * message_memory)
{
  auto typed_message = static_cast<arm_control::msg::JointControl *>(message_memory);
  typed_message->~JointControl();
}

size_t size_function__JointControl__joint_pos(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__JointControl__joint_pos(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__JointControl__joint_pos(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointControl__joint_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__JointControl__joint_pos(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__JointControl__joint_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__JointControl__joint_pos(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

size_t size_function__JointControl__joint_vel(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__JointControl__joint_vel(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__JointControl__joint_vel(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointControl__joint_vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__JointControl__joint_vel(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__JointControl__joint_vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__JointControl__joint_vel(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

size_t size_function__JointControl__joint_cur(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__JointControl__joint_cur(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__JointControl__joint_cur(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__JointControl__joint_cur(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__JointControl__joint_cur(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__JointControl__joint_cur(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__JointControl__joint_cur(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember JointControl_message_member_array[4] = {
  {
    "joint_pos",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control::msg::JointControl, joint_pos),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointControl__joint_pos,  // size() function pointer
    get_const_function__JointControl__joint_pos,  // get_const(index) function pointer
    get_function__JointControl__joint_pos,  // get(index) function pointer
    fetch_function__JointControl__joint_pos,  // fetch(index, &value) function pointer
    assign_function__JointControl__joint_pos,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joint_vel",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control::msg::JointControl, joint_vel),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointControl__joint_vel,  // size() function pointer
    get_const_function__JointControl__joint_vel,  // get_const(index) function pointer
    get_function__JointControl__joint_vel,  // get(index) function pointer
    fetch_function__JointControl__joint_vel,  // fetch(index, &value) function pointer
    assign_function__JointControl__joint_vel,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "joint_cur",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control::msg::JointControl, joint_cur),  // bytes offset in struct
    nullptr,  // default value
    size_function__JointControl__joint_cur,  // size() function pointer
    get_const_function__JointControl__joint_cur,  // get_const(index) function pointer
    get_function__JointControl__joint_cur,  // get(index) function pointer
    fetch_function__JointControl__joint_cur,  // fetch(index, &value) function pointer
    assign_function__JointControl__joint_cur,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mode",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arm_control::msg::JointControl, mode),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers JointControl_message_members = {
  "arm_control::msg",  // message namespace
  "JointControl",  // message name
  4,  // number of fields
  sizeof(arm_control::msg::JointControl),
  JointControl_message_member_array,  // message members
  JointControl_init_function,  // function to initialize message memory (memory has to be allocated)
  JointControl_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t JointControl_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &JointControl_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace arm_control


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<arm_control::msg::JointControl>()
{
  return &::arm_control::msg::rosidl_typesupport_introspection_cpp::JointControl_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, arm_control, msg, JointControl)() {
  return &::arm_control::msg::rosidl_typesupport_introspection_cpp::JointControl_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

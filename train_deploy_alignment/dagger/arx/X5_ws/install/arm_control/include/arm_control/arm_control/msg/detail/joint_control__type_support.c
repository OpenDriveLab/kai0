// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "arm_control/msg/detail/joint_control__rosidl_typesupport_introspection_c.h"
#include "arm_control/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "arm_control/msg/detail/joint_control__functions.h"
#include "arm_control/msg/detail/joint_control__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  arm_control__msg__JointControl__init(message_memory);
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_fini_function(void * message_memory)
{
  arm_control__msg__JointControl__fini(message_memory);
}

size_t arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_pos(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_pos(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_pos(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_pos(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_pos(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

size_t arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_vel(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_vel(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_vel(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_vel(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_vel(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

size_t arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_cur(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_cur(
  const void * untyped_member, size_t index)
{
  const float * member =
    (const float *)(untyped_member);
  return &member[index];
}

void * arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_cur(
  void * untyped_member, size_t index)
{
  float * member =
    (float *)(untyped_member);
  return &member[index];
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_cur(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_cur(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_cur(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_cur(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_member_array[4] = {
  {
    "joint_pos",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control__msg__JointControl, joint_pos),  // bytes offset in struct
    NULL,  // default value
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_pos,  // size() function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_pos,  // get_const(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_pos,  // get(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_pos,  // fetch(index, &value) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_pos,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_vel",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control__msg__JointControl, joint_vel),  // bytes offset in struct
    NULL,  // default value
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_vel,  // size() function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_vel,  // get_const(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_vel,  // get(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_vel,  // fetch(index, &value) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_vel,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_cur",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(arm_control__msg__JointControl, joint_cur),  // bytes offset in struct
    NULL,  // default value
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__size_function__JointControl__joint_cur,  // size() function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_const_function__JointControl__joint_cur,  // get_const(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__get_function__JointControl__joint_cur,  // get(index) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__fetch_function__JointControl__joint_cur,  // fetch(index, &value) function pointer
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__assign_function__JointControl__joint_cur,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arm_control__msg__JointControl, mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_members = {
  "arm_control__msg",  // message namespace
  "JointControl",  // message name
  4,  // number of fields
  sizeof(arm_control__msg__JointControl),
  arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_member_array,  // message members
  arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_init_function,  // function to initialize message memory (memory has to be allocated)
  arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_type_support_handle = {
  0,
  &arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_arm_control
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, arm_control, msg, JointControl)() {
  if (!arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_type_support_handle.typesupport_identifier) {
    arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &arm_control__msg__JointControl__rosidl_typesupport_introspection_c__JointControl_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

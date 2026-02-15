// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from arx5_arm_msg:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "arx5_arm_msg/msg/detail/robot_status__rosidl_typesupport_introspection_c.h"
#include "arx5_arm_msg/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "arx5_arm_msg/msg/detail/robot_status__functions.h"
#include "arx5_arm_msg/msg/detail/robot_status__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  arx5_arm_msg__msg__RobotStatus__init(message_memory);
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_fini_function(void * message_memory)
{
  arx5_arm_msg__msg__RobotStatus__fini(message_memory);
}

size_t arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__end_pos(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__end_pos(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__end_pos(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__end_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__end_pos(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__end_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__end_pos(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_pos(
  const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_pos(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_pos(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_pos(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_pos(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_vel(
  const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_vel(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_vel(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_vel(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_vel(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_cur(
  const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_cur(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_cur(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_cur(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_cur(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_cur(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_cur(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg__msg__RobotStatus, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "end_pos",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg__msg__RobotStatus, end_pos),  // bytes offset in struct
    NULL,  // default value
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__end_pos,  // size() function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__end_pos,  // get_const(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__end_pos,  // get(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__end_pos,  // fetch(index, &value) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__end_pos,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_pos",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg__msg__RobotStatus, joint_pos),  // bytes offset in struct
    NULL,  // default value
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_pos,  // size() function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_pos,  // get_const(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_pos,  // get(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_pos,  // fetch(index, &value) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_pos,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_vel",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg__msg__RobotStatus, joint_vel),  // bytes offset in struct
    NULL,  // default value
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_vel,  // size() function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_vel,  // get_const(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_vel,  // get(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_vel,  // fetch(index, &value) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_vel,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_cur",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(arx5_arm_msg__msg__RobotStatus, joint_cur),  // bytes offset in struct
    NULL,  // default value
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__size_function__RobotStatus__joint_cur,  // size() function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_const_function__RobotStatus__joint_cur,  // get_const(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__get_function__RobotStatus__joint_cur,  // get(index) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__fetch_function__RobotStatus__joint_cur,  // fetch(index, &value) function pointer
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__assign_function__RobotStatus__joint_cur,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_members = {
  "arx5_arm_msg__msg",  // message namespace
  "RobotStatus",  // message name
  5,  // number of fields
  sizeof(arx5_arm_msg__msg__RobotStatus),
  arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_member_array,  // message members
  arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_type_support_handle = {
  0,
  &arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_arx5_arm_msg
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, arx5_arm_msg, msg, RobotStatus)() {
  arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_type_support_handle.typesupport_identifier) {
    arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &arx5_arm_msg__msg__RobotStatus__rosidl_typesupport_introspection_c__RobotStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

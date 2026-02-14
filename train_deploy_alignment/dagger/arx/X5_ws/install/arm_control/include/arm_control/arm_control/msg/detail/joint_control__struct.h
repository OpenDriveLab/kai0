// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_H_
#define ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/JointControl in the package arm_control.
typedef struct arm_control__msg__JointControl
{
  float joint_pos[8];
  float joint_vel[8];
  float joint_cur[8];
  int32_t mode;
} arm_control__msg__JointControl;

// Struct for a sequence of arm_control__msg__JointControl.
typedef struct arm_control__msg__JointControl__Sequence
{
  arm_control__msg__JointControl * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} arm_control__msg__JointControl__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ARM_CONTROL__MSG__DETAIL__JOINT_CONTROL__STRUCT_H_

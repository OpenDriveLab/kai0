// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from arx5_arm_msg:msg/RobotStatus.idl
// generated code does not contain a copyright notice

#ifndef ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_
#define ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/RobotStatus in the package arx5_arm_msg.
typedef struct arx5_arm_msg__msg__RobotStatus
{
  std_msgs__msg__Header header;
  double end_pos[6];
  double joint_pos[7];
  double joint_vel[7];
  double joint_cur[7];
} arx5_arm_msg__msg__RobotStatus;

// Struct for a sequence of arx5_arm_msg__msg__RobotStatus.
typedef struct arx5_arm_msg__msg__RobotStatus__Sequence
{
  arx5_arm_msg__msg__RobotStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} arx5_arm_msg__msg__RobotStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ARX5_ARM_MSG__MSG__DETAIL__ROBOT_STATUS__STRUCT_H_

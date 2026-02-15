// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice

#ifndef ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_H_
#define ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/ArxImu in the package arm_control.
typedef struct arm_control__msg__ArxImu
{
  builtin_interfaces__msg__Time stamp;
  double roll;
  double pitch;
  double yaw;
  geometry_msgs__msg__Vector3 angular_velocity;
} arm_control__msg__ArxImu;

// Struct for a sequence of arm_control__msg__ArxImu.
typedef struct arm_control__msg__ArxImu__Sequence
{
  arm_control__msg__ArxImu * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} arm_control__msg__ArxImu__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ARM_CONTROL__MSG__DETAIL__ARX_IMU__STRUCT_H_

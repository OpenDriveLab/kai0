// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from arx5_arm_msg:msg/RobotStatus.idl
// generated code does not contain a copyright notice
#include "arx5_arm_msg/msg/detail/robot_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
arx5_arm_msg__msg__RobotStatus__init(arx5_arm_msg__msg__RobotStatus * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    arx5_arm_msg__msg__RobotStatus__fini(msg);
    return false;
  }
  // end_pos
  // joint_pos
  // joint_vel
  // joint_cur
  return true;
}

void
arx5_arm_msg__msg__RobotStatus__fini(arx5_arm_msg__msg__RobotStatus * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // end_pos
  // joint_pos
  // joint_vel
  // joint_cur
}

bool
arx5_arm_msg__msg__RobotStatus__are_equal(const arx5_arm_msg__msg__RobotStatus * lhs, const arx5_arm_msg__msg__RobotStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // end_pos
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->end_pos[i] != rhs->end_pos[i]) {
      return false;
    }
  }
  // joint_pos
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->joint_pos[i] != rhs->joint_pos[i]) {
      return false;
    }
  }
  // joint_vel
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->joint_vel[i] != rhs->joint_vel[i]) {
      return false;
    }
  }
  // joint_cur
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->joint_cur[i] != rhs->joint_cur[i]) {
      return false;
    }
  }
  return true;
}

bool
arx5_arm_msg__msg__RobotStatus__copy(
  const arx5_arm_msg__msg__RobotStatus * input,
  arx5_arm_msg__msg__RobotStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // end_pos
  for (size_t i = 0; i < 6; ++i) {
    output->end_pos[i] = input->end_pos[i];
  }
  // joint_pos
  for (size_t i = 0; i < 7; ++i) {
    output->joint_pos[i] = input->joint_pos[i];
  }
  // joint_vel
  for (size_t i = 0; i < 7; ++i) {
    output->joint_vel[i] = input->joint_vel[i];
  }
  // joint_cur
  for (size_t i = 0; i < 7; ++i) {
    output->joint_cur[i] = input->joint_cur[i];
  }
  return true;
}

arx5_arm_msg__msg__RobotStatus *
arx5_arm_msg__msg__RobotStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotStatus * msg = (arx5_arm_msg__msg__RobotStatus *)allocator.allocate(sizeof(arx5_arm_msg__msg__RobotStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(arx5_arm_msg__msg__RobotStatus));
  bool success = arx5_arm_msg__msg__RobotStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
arx5_arm_msg__msg__RobotStatus__destroy(arx5_arm_msg__msg__RobotStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    arx5_arm_msg__msg__RobotStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
arx5_arm_msg__msg__RobotStatus__Sequence__init(arx5_arm_msg__msg__RobotStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotStatus * data = NULL;

  if (size) {
    data = (arx5_arm_msg__msg__RobotStatus *)allocator.zero_allocate(size, sizeof(arx5_arm_msg__msg__RobotStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = arx5_arm_msg__msg__RobotStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        arx5_arm_msg__msg__RobotStatus__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
arx5_arm_msg__msg__RobotStatus__Sequence__fini(arx5_arm_msg__msg__RobotStatus__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      arx5_arm_msg__msg__RobotStatus__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

arx5_arm_msg__msg__RobotStatus__Sequence *
arx5_arm_msg__msg__RobotStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotStatus__Sequence * array = (arx5_arm_msg__msg__RobotStatus__Sequence *)allocator.allocate(sizeof(arx5_arm_msg__msg__RobotStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = arx5_arm_msg__msg__RobotStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
arx5_arm_msg__msg__RobotStatus__Sequence__destroy(arx5_arm_msg__msg__RobotStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    arx5_arm_msg__msg__RobotStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
arx5_arm_msg__msg__RobotStatus__Sequence__are_equal(const arx5_arm_msg__msg__RobotStatus__Sequence * lhs, const arx5_arm_msg__msg__RobotStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!arx5_arm_msg__msg__RobotStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
arx5_arm_msg__msg__RobotStatus__Sequence__copy(
  const arx5_arm_msg__msg__RobotStatus__Sequence * input,
  arx5_arm_msg__msg__RobotStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(arx5_arm_msg__msg__RobotStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    arx5_arm_msg__msg__RobotStatus * data =
      (arx5_arm_msg__msg__RobotStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!arx5_arm_msg__msg__RobotStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          arx5_arm_msg__msg__RobotStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!arx5_arm_msg__msg__RobotStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

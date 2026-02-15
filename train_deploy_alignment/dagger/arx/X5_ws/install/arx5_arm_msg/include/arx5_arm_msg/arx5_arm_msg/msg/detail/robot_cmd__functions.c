// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from arx5_arm_msg:msg/RobotCmd.idl
// generated code does not contain a copyright notice
#include "arx5_arm_msg/msg/detail/robot_cmd__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
arx5_arm_msg__msg__RobotCmd__init(arx5_arm_msg__msg__RobotCmd * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    arx5_arm_msg__msg__RobotCmd__fini(msg);
    return false;
  }
  // end_pos
  // joint_pos
  // gripper
  // mode
  return true;
}

void
arx5_arm_msg__msg__RobotCmd__fini(arx5_arm_msg__msg__RobotCmd * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // end_pos
  // joint_pos
  // gripper
  // mode
}

bool
arx5_arm_msg__msg__RobotCmd__are_equal(const arx5_arm_msg__msg__RobotCmd * lhs, const arx5_arm_msg__msg__RobotCmd * rhs)
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
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->joint_pos[i] != rhs->joint_pos[i]) {
      return false;
    }
  }
  // gripper
  if (lhs->gripper != rhs->gripper) {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  return true;
}

bool
arx5_arm_msg__msg__RobotCmd__copy(
  const arx5_arm_msg__msg__RobotCmd * input,
  arx5_arm_msg__msg__RobotCmd * output)
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
  for (size_t i = 0; i < 6; ++i) {
    output->joint_pos[i] = input->joint_pos[i];
  }
  // gripper
  output->gripper = input->gripper;
  // mode
  output->mode = input->mode;
  return true;
}

arx5_arm_msg__msg__RobotCmd *
arx5_arm_msg__msg__RobotCmd__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotCmd * msg = (arx5_arm_msg__msg__RobotCmd *)allocator.allocate(sizeof(arx5_arm_msg__msg__RobotCmd), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(arx5_arm_msg__msg__RobotCmd));
  bool success = arx5_arm_msg__msg__RobotCmd__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
arx5_arm_msg__msg__RobotCmd__destroy(arx5_arm_msg__msg__RobotCmd * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    arx5_arm_msg__msg__RobotCmd__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
arx5_arm_msg__msg__RobotCmd__Sequence__init(arx5_arm_msg__msg__RobotCmd__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotCmd * data = NULL;

  if (size) {
    data = (arx5_arm_msg__msg__RobotCmd *)allocator.zero_allocate(size, sizeof(arx5_arm_msg__msg__RobotCmd), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = arx5_arm_msg__msg__RobotCmd__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        arx5_arm_msg__msg__RobotCmd__fini(&data[i - 1]);
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
arx5_arm_msg__msg__RobotCmd__Sequence__fini(arx5_arm_msg__msg__RobotCmd__Sequence * array)
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
      arx5_arm_msg__msg__RobotCmd__fini(&array->data[i]);
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

arx5_arm_msg__msg__RobotCmd__Sequence *
arx5_arm_msg__msg__RobotCmd__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arx5_arm_msg__msg__RobotCmd__Sequence * array = (arx5_arm_msg__msg__RobotCmd__Sequence *)allocator.allocate(sizeof(arx5_arm_msg__msg__RobotCmd__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = arx5_arm_msg__msg__RobotCmd__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
arx5_arm_msg__msg__RobotCmd__Sequence__destroy(arx5_arm_msg__msg__RobotCmd__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    arx5_arm_msg__msg__RobotCmd__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
arx5_arm_msg__msg__RobotCmd__Sequence__are_equal(const arx5_arm_msg__msg__RobotCmd__Sequence * lhs, const arx5_arm_msg__msg__RobotCmd__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!arx5_arm_msg__msg__RobotCmd__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
arx5_arm_msg__msg__RobotCmd__Sequence__copy(
  const arx5_arm_msg__msg__RobotCmd__Sequence * input,
  arx5_arm_msg__msg__RobotCmd__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(arx5_arm_msg__msg__RobotCmd);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    arx5_arm_msg__msg__RobotCmd * data =
      (arx5_arm_msg__msg__RobotCmd *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!arx5_arm_msg__msg__RobotCmd__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          arx5_arm_msg__msg__RobotCmd__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!arx5_arm_msg__msg__RobotCmd__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

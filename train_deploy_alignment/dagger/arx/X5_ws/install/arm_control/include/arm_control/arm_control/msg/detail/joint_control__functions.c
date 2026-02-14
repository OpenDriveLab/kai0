// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from arm_control:msg/JointControl.idl
// generated code does not contain a copyright notice
#include "arm_control/msg/detail/joint_control__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
arm_control__msg__JointControl__init(arm_control__msg__JointControl * msg)
{
  if (!msg) {
    return false;
  }
  // joint_pos
  // joint_vel
  // joint_cur
  // mode
  return true;
}

void
arm_control__msg__JointControl__fini(arm_control__msg__JointControl * msg)
{
  if (!msg) {
    return;
  }
  // joint_pos
  // joint_vel
  // joint_cur
  // mode
}

bool
arm_control__msg__JointControl__are_equal(const arm_control__msg__JointControl * lhs, const arm_control__msg__JointControl * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // joint_pos
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->joint_pos[i] != rhs->joint_pos[i]) {
      return false;
    }
  }
  // joint_vel
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->joint_vel[i] != rhs->joint_vel[i]) {
      return false;
    }
  }
  // joint_cur
  for (size_t i = 0; i < 8; ++i) {
    if (lhs->joint_cur[i] != rhs->joint_cur[i]) {
      return false;
    }
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  return true;
}

bool
arm_control__msg__JointControl__copy(
  const arm_control__msg__JointControl * input,
  arm_control__msg__JointControl * output)
{
  if (!input || !output) {
    return false;
  }
  // joint_pos
  for (size_t i = 0; i < 8; ++i) {
    output->joint_pos[i] = input->joint_pos[i];
  }
  // joint_vel
  for (size_t i = 0; i < 8; ++i) {
    output->joint_vel[i] = input->joint_vel[i];
  }
  // joint_cur
  for (size_t i = 0; i < 8; ++i) {
    output->joint_cur[i] = input->joint_cur[i];
  }
  // mode
  output->mode = input->mode;
  return true;
}

arm_control__msg__JointControl *
arm_control__msg__JointControl__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__JointControl * msg = (arm_control__msg__JointControl *)allocator.allocate(sizeof(arm_control__msg__JointControl), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(arm_control__msg__JointControl));
  bool success = arm_control__msg__JointControl__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
arm_control__msg__JointControl__destroy(arm_control__msg__JointControl * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    arm_control__msg__JointControl__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
arm_control__msg__JointControl__Sequence__init(arm_control__msg__JointControl__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__JointControl * data = NULL;

  if (size) {
    data = (arm_control__msg__JointControl *)allocator.zero_allocate(size, sizeof(arm_control__msg__JointControl), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = arm_control__msg__JointControl__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        arm_control__msg__JointControl__fini(&data[i - 1]);
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
arm_control__msg__JointControl__Sequence__fini(arm_control__msg__JointControl__Sequence * array)
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
      arm_control__msg__JointControl__fini(&array->data[i]);
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

arm_control__msg__JointControl__Sequence *
arm_control__msg__JointControl__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__JointControl__Sequence * array = (arm_control__msg__JointControl__Sequence *)allocator.allocate(sizeof(arm_control__msg__JointControl__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = arm_control__msg__JointControl__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
arm_control__msg__JointControl__Sequence__destroy(arm_control__msg__JointControl__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    arm_control__msg__JointControl__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
arm_control__msg__JointControl__Sequence__are_equal(const arm_control__msg__JointControl__Sequence * lhs, const arm_control__msg__JointControl__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!arm_control__msg__JointControl__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
arm_control__msg__JointControl__Sequence__copy(
  const arm_control__msg__JointControl__Sequence * input,
  arm_control__msg__JointControl__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(arm_control__msg__JointControl);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    arm_control__msg__JointControl * data =
      (arm_control__msg__JointControl *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!arm_control__msg__JointControl__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          arm_control__msg__JointControl__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!arm_control__msg__JointControl__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

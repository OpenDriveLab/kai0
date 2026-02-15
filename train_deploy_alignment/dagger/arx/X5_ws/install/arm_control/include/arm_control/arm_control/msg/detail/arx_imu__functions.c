// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from arm_control:msg/ArxImu.idl
// generated code does not contain a copyright notice
#include "arm_control/msg/detail/arx_imu__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `angular_velocity`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
arm_control__msg__ArxImu__init(arm_control__msg__ArxImu * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    arm_control__msg__ArxImu__fini(msg);
    return false;
  }
  // roll
  // pitch
  // yaw
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->angular_velocity)) {
    arm_control__msg__ArxImu__fini(msg);
    return false;
  }
  return true;
}

void
arm_control__msg__ArxImu__fini(arm_control__msg__ArxImu * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // roll
  // pitch
  // yaw
  // angular_velocity
  geometry_msgs__msg__Vector3__fini(&msg->angular_velocity);
}

bool
arm_control__msg__ArxImu__are_equal(const arm_control__msg__ArxImu * lhs, const arm_control__msg__ArxImu * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // pitch
  if (lhs->pitch != rhs->pitch) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->angular_velocity), &(rhs->angular_velocity)))
  {
    return false;
  }
  return true;
}

bool
arm_control__msg__ArxImu__copy(
  const arm_control__msg__ArxImu * input,
  arm_control__msg__ArxImu * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->angular_velocity), &(output->angular_velocity)))
  {
    return false;
  }
  return true;
}

arm_control__msg__ArxImu *
arm_control__msg__ArxImu__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__ArxImu * msg = (arm_control__msg__ArxImu *)allocator.allocate(sizeof(arm_control__msg__ArxImu), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(arm_control__msg__ArxImu));
  bool success = arm_control__msg__ArxImu__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
arm_control__msg__ArxImu__destroy(arm_control__msg__ArxImu * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    arm_control__msg__ArxImu__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
arm_control__msg__ArxImu__Sequence__init(arm_control__msg__ArxImu__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__ArxImu * data = NULL;

  if (size) {
    data = (arm_control__msg__ArxImu *)allocator.zero_allocate(size, sizeof(arm_control__msg__ArxImu), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = arm_control__msg__ArxImu__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        arm_control__msg__ArxImu__fini(&data[i - 1]);
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
arm_control__msg__ArxImu__Sequence__fini(arm_control__msg__ArxImu__Sequence * array)
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
      arm_control__msg__ArxImu__fini(&array->data[i]);
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

arm_control__msg__ArxImu__Sequence *
arm_control__msg__ArxImu__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__ArxImu__Sequence * array = (arm_control__msg__ArxImu__Sequence *)allocator.allocate(sizeof(arm_control__msg__ArxImu__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = arm_control__msg__ArxImu__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
arm_control__msg__ArxImu__Sequence__destroy(arm_control__msg__ArxImu__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    arm_control__msg__ArxImu__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
arm_control__msg__ArxImu__Sequence__are_equal(const arm_control__msg__ArxImu__Sequence * lhs, const arm_control__msg__ArxImu__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!arm_control__msg__ArxImu__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
arm_control__msg__ArxImu__Sequence__copy(
  const arm_control__msg__ArxImu__Sequence * input,
  arm_control__msg__ArxImu__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(arm_control__msg__ArxImu);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    arm_control__msg__ArxImu * data =
      (arm_control__msg__ArxImu *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!arm_control__msg__ArxImu__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          arm_control__msg__ArxImu__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!arm_control__msg__ArxImu__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from arm_control:msg/PosCmd.idl
// generated code does not contain a copyright notice
#include "arm_control/msg/detail/pos_cmd__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
arm_control__msg__PosCmd__init(arm_control__msg__PosCmd * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
  // gripper
  // quater_x
  // quater_y
  // quater_z
  // quater_w
  // chx
  // chy
  // chz
  // vel_l
  // vel_r
  // height
  // head_pit
  // head_yaw
  // temp_float_data
  // temp_int_data
  // mode1
  // mode2
  // time_count
  return true;
}

void
arm_control__msg__PosCmd__fini(arm_control__msg__PosCmd * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
  // z
  // roll
  // pitch
  // yaw
  // gripper
  // quater_x
  // quater_y
  // quater_z
  // quater_w
  // chx
  // chy
  // chz
  // vel_l
  // vel_r
  // height
  // head_pit
  // head_yaw
  // temp_float_data
  // temp_int_data
  // mode1
  // mode2
  // time_count
}

bool
arm_control__msg__PosCmd__are_equal(const arm_control__msg__PosCmd * lhs, const arm_control__msg__PosCmd * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
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
  // gripper
  if (lhs->gripper != rhs->gripper) {
    return false;
  }
  // quater_x
  if (lhs->quater_x != rhs->quater_x) {
    return false;
  }
  // quater_y
  if (lhs->quater_y != rhs->quater_y) {
    return false;
  }
  // quater_z
  if (lhs->quater_z != rhs->quater_z) {
    return false;
  }
  // quater_w
  if (lhs->quater_w != rhs->quater_w) {
    return false;
  }
  // chx
  if (lhs->chx != rhs->chx) {
    return false;
  }
  // chy
  if (lhs->chy != rhs->chy) {
    return false;
  }
  // chz
  if (lhs->chz != rhs->chz) {
    return false;
  }
  // vel_l
  if (lhs->vel_l != rhs->vel_l) {
    return false;
  }
  // vel_r
  if (lhs->vel_r != rhs->vel_r) {
    return false;
  }
  // height
  if (lhs->height != rhs->height) {
    return false;
  }
  // head_pit
  if (lhs->head_pit != rhs->head_pit) {
    return false;
  }
  // head_yaw
  if (lhs->head_yaw != rhs->head_yaw) {
    return false;
  }
  // temp_float_data
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->temp_float_data[i] != rhs->temp_float_data[i]) {
      return false;
    }
  }
  // temp_int_data
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->temp_int_data[i] != rhs->temp_int_data[i]) {
      return false;
    }
  }
  // mode1
  if (lhs->mode1 != rhs->mode1) {
    return false;
  }
  // mode2
  if (lhs->mode2 != rhs->mode2) {
    return false;
  }
  // time_count
  if (lhs->time_count != rhs->time_count) {
    return false;
  }
  return true;
}

bool
arm_control__msg__PosCmd__copy(
  const arm_control__msg__PosCmd * input,
  arm_control__msg__PosCmd * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  // roll
  output->roll = input->roll;
  // pitch
  output->pitch = input->pitch;
  // yaw
  output->yaw = input->yaw;
  // gripper
  output->gripper = input->gripper;
  // quater_x
  output->quater_x = input->quater_x;
  // quater_y
  output->quater_y = input->quater_y;
  // quater_z
  output->quater_z = input->quater_z;
  // quater_w
  output->quater_w = input->quater_w;
  // chx
  output->chx = input->chx;
  // chy
  output->chy = input->chy;
  // chz
  output->chz = input->chz;
  // vel_l
  output->vel_l = input->vel_l;
  // vel_r
  output->vel_r = input->vel_r;
  // height
  output->height = input->height;
  // head_pit
  output->head_pit = input->head_pit;
  // head_yaw
  output->head_yaw = input->head_yaw;
  // temp_float_data
  for (size_t i = 0; i < 6; ++i) {
    output->temp_float_data[i] = input->temp_float_data[i];
  }
  // temp_int_data
  for (size_t i = 0; i < 6; ++i) {
    output->temp_int_data[i] = input->temp_int_data[i];
  }
  // mode1
  output->mode1 = input->mode1;
  // mode2
  output->mode2 = input->mode2;
  // time_count
  output->time_count = input->time_count;
  return true;
}

arm_control__msg__PosCmd *
arm_control__msg__PosCmd__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__PosCmd * msg = (arm_control__msg__PosCmd *)allocator.allocate(sizeof(arm_control__msg__PosCmd), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(arm_control__msg__PosCmd));
  bool success = arm_control__msg__PosCmd__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
arm_control__msg__PosCmd__destroy(arm_control__msg__PosCmd * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    arm_control__msg__PosCmd__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
arm_control__msg__PosCmd__Sequence__init(arm_control__msg__PosCmd__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__PosCmd * data = NULL;

  if (size) {
    data = (arm_control__msg__PosCmd *)allocator.zero_allocate(size, sizeof(arm_control__msg__PosCmd), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = arm_control__msg__PosCmd__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        arm_control__msg__PosCmd__fini(&data[i - 1]);
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
arm_control__msg__PosCmd__Sequence__fini(arm_control__msg__PosCmd__Sequence * array)
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
      arm_control__msg__PosCmd__fini(&array->data[i]);
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

arm_control__msg__PosCmd__Sequence *
arm_control__msg__PosCmd__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  arm_control__msg__PosCmd__Sequence * array = (arm_control__msg__PosCmd__Sequence *)allocator.allocate(sizeof(arm_control__msg__PosCmd__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = arm_control__msg__PosCmd__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
arm_control__msg__PosCmd__Sequence__destroy(arm_control__msg__PosCmd__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    arm_control__msg__PosCmd__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
arm_control__msg__PosCmd__Sequence__are_equal(const arm_control__msg__PosCmd__Sequence * lhs, const arm_control__msg__PosCmd__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!arm_control__msg__PosCmd__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
arm_control__msg__PosCmd__Sequence__copy(
  const arm_control__msg__PosCmd__Sequence * input,
  arm_control__msg__PosCmd__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(arm_control__msg__PosCmd);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    arm_control__msg__PosCmd * data =
      (arm_control__msg__PosCmd *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!arm_control__msg__PosCmd__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          arm_control__msg__PosCmd__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!arm_control__msg__PosCmd__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

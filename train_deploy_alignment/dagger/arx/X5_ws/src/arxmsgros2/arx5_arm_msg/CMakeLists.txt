cmake_minimum_required(VERSION 3.8)
project(arx5_arm_msg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotCmd.msg"
  "msg/RobotStatus.msg"
  DEPENDENCIES 
  sensor_msgs
  std_msgs
  geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
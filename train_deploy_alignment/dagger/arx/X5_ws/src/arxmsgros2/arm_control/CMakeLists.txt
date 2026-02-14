cmake_minimum_required(VERSION 3.8)
project(arm_control)

# 找到需要的依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 声明消息文件
rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/PosCmd.msg"
        "msg/JointControl.msg"
        "msg/ArxImu.msg"
        DEPENDENCIES
        std_msgs
        geometry_msgs
        builtin_interfaces
)

# 添加对接口的依赖
ament_export_dependencies(rosidl_default_runtime)

ament_package()


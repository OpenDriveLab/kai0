import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

params_file = os.path.join(
    get_package_share_directory('arx_x5_controller'), 'config', 'vr_single_arm.yaml')

arm_node = Node(
    package='arx_x5_controller',
    executable='X5Controller',
    name='vr_arm',
    output='screen',
    parameters=[params_file],
)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(name='params_file',
                              default_value=params_file),
        arm_node,
    ])

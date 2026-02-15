import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

params_file = os.path.join(
    get_package_share_directory('arx_x5_controller'), 'config', 'double_arm.yaml')

arm_l =Node(
    package='arx_x5_controller',
    executable='X5Controller',
    name='arm_l',
    output='screen',
    parameters=[params_file],
)

arm_r =Node(
    package='arx_x5_controller',
    executable='X5Controller',
    name='arm_r',
    output='screen',
    parameters=[params_file],
)


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(name='params_file',
                              default_value=params_file),
        arm_l,
        arm_r,
    ])
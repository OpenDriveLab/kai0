from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import os
import sys
import arx_x5_python as arx


def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    参数：
        quat: np.ndarray, 长度为 4 的数组 [w, x, y, z]
    返回：
        roll, pitch, yaw: 以弧度为单位的欧拉角
    """
    w, x, y, z = quat

    # 计算 roll (x 轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # 计算 pitch (y 轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # 使用 90 度限制
    else:
        pitch = np.arcsin(sinp)

    # 计算 yaw (z 轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    将欧拉角（roll, pitch, yaw）转换为四元数。

    参数：
        roll: 绕 x 轴的旋转角（弧度）
        pitch: 绕 y 轴的旋转角（弧度）
        yaw: 绕 z 轴的旋转角（弧度）

    返回：
        np.ndarray: 长度为 4 的四元数数组 [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


class SingleArm:
    """
    Base class for a single robot arm.

    Args:
        config (Dict[str, sAny]): Configuration dictionary for the robot arm

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the robot arm
        num_joints (int): Number of joints in the arm
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_joints = config.get(
            "num_joints", 7
        )  # Default to 7 joints if not specified
        self.dt = config.get(
            "dt", 0.05
        )  # Default to 0.05s time step if not specified. set up the control frequency

        current_dir = os.path.dirname(os.path.abspath(__file__))
        type = config.get("type",0)
        if(type == 0):
            urdf_path = os.path.join(current_dir,"x5.urdf")
        elif(type == 1):
            urdf_path = os.path.join(current_dir,"x5_master.urdf")
        else:
            urdf_path = os.path.join(current_dir,"x5_2025.urdf")
        self.arm = arx.InterfacesPy(urdf_path,config.get("can_port", "can0"),type)
        self.arm.arx_x(500,2000,10)

    def get_joint_names(self) -> List[str]:
        """
        Get the names of all joints in the arm.

        Returns:
            List[str]: List of joint names. Shape: (num_joints,)
        """
        return NotImplementedError

    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        self.arm.set_arm_status(1)
        return True

    def gravity_compensation(self) -> bool:
        self.arm.set_arm_status(3)
        return True

    def protect_mode(self) -> bool:
        self.arm.set_arm_status(2)
        return True

    def set_joint_positions(
        self,
        positions: Union[float, List[float], np.ndarray],  # Shape: (num_joints,)
        **kwargs
    ) -> bool:
        """
        Move the arm to the given joint position(s).

        Args:
            positions: Desired joint position(s). Shape: (6)
            **kwargs: Additional arguments

        """
        self.arm.set_joint_positions(positions)
        self.arm.set_arm_status(5)

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (3,)
        quat: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (4,)
        **kwargs
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            pos: Desired position [x, y, z]. Shape: (3,)
            ori: Desired orientation (quaternion).
                 Shape: (4,) (w, x, y, z)
            **kwargs: Additional arguments

        """

        self.arm.set_ee_pose([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (6,)
        **kwargs
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, rol, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments

        """
        quat = euler_to_quaternion(xyzrpy[3], xyzrpy[4], xyzrpy[5])

        self.arm.set_ee_pose(
            [xyzrpy[0], xyzrpy[1], xyzrpy[2], quat[0], quat[1], quat[2], quat[3]]
        )
        self.arm.set_arm_status(4)

    def set_catch_pos(self, pos: float):
        self.arm.set_catch(pos)

    def get_joint_positions(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for. Shape: (num_joints,) or single string. If None,
                            return positions for all joints.

        """
        return self.arm.get_joint_positions()

    def get_joint_velocities(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for. Shape: (num_joints,) or single string. If None,
                            return velocities for all joints.

        """
        return self.arm.get_joint_velocities()

    def get_joint_currents(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        return self.arm.get_joint_currents()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current end effector pose of the arm.

        Returns:
            End effector pose as (position, quaternion)
            Shapes: position (3,), quaternion (4,) [w, x, y, z]
        """
        xyzwxyz = self.arm.get_ee_pose()

        return xyzwxyz

    def get_ee_pose_xyzrpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        xyzwxyz = self.arm.get_ee_pose()

        array = np.array([xyzwxyz[3], xyzwxyz[4], xyzwxyz[5], xyzwxyz[6]])

        roll, pitch, yaw = quaternion_to_euler(array)

        xyzrpy = np.array([xyzwxyz[0], xyzwxyz[1], xyzwxyz[2], roll, pitch, yaw])

        return xyzrpy

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'arm'):
            # 可以在这里添加必要的清理代码
            pass
    
    def __del__(self):
        # 或者可以直接在析构函数中释放资源
        print("销毁 SingleArm 对象")
        self.cleanup()

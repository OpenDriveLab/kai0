from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
from .single_arm import SingleArm
import time

class BimanualArm:
    """
    Class for bimanual robot arm teleoperation. We assume the base/root link of the URDF model
    is in the center of shoulder (the midpoint between the bases of the two arms).
    x-axis pointing forward, y-axis pointing right, z-axis pointing down.

    Args:
        left_arm_config (Dict[str, Any]): Configuration dictionary for the left arm
        right_arm_config (Dict[str, Any]): Configuration dictionary for the right arm

    Attributes:
        left_arm (SingleArm): Left arm instance
        right_arm (SingleArm): Right arm instance
    """

    def __init__(
        self, left_arm_config: Dict[str, Any], right_arm_config: Dict[str, Any]
    ):
        import threading

        def init_left():
            self.left_arm = SingleArm(left_arm_config)
        def init_right():
            self.right_arm = SingleArm(right_arm_config)

        t_left = threading.Thread(target=init_left)
        t_right = threading.Thread(target=init_right)
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

    def go_home(self) -> Dict[str, bool]:
        """
        Move both robot arms to pre-defined home poses.

        Returns:
            Dict[str, bool]: Success status for each arm
        """
        return {"left": self.left_arm.go_home(), "right": self.right_arm.go_home()}

    def gravity_compensation(self) -> Dict[str, bool]:
        return {
            "left": self.left_arm.gravity_compensation(),
            "right": self.right_arm.gravity_compensation(),
        }

    def get_joint_names(self, arm: str = "both") -> Dict[str, List[str]]:
        """
        Get the names of all joints for the specified arm(s).

        Args:
            arm (str): Which arm to get joint names for ("left", "right", or "both")

        Returns:
            Dict[str, List[str]]: Dictionary containing joint names for the specified arm(s).
                                  Shape: Dict with keys 'left' and/or 'right',
                                         each with a list of joint names of shape (num_joints,)
        """
        if arm == "both":
            return {
                "left": self.left_arm.get_joint_names(),
                "right": self.right_arm.get_joint_names(),
            }
        elif arm == "left":
            return {"left": self.left_arm.get_joint_names()}
        elif arm == "right":
            return {"right": self.right_arm.get_joint_names()}
        else:
            raise ValueError("Invalid arm specified. Use 'left', 'right', or 'both'.")

    def set_joint_positions(
        self,
        positions: Dict[str, Union[float, List[float], np.ndarray]],
        joint_names: Optional[Dict[str, Union[str, List[str]]]] = None,
        **kwargs
    ):
        """
        Set the target joint position(s) for the specified arm(s).

        Args:
            positions: Dictionary with keys 'left' and/or 'right', each containing desired joint position(s).
                        Shape of each arm's positions: (num_joints,)
            joint_names: Dictionary with keys 'left' and/or 'right', each containing name(s) of the joint(s) to set position for.
                         Shape of each arm's joint_names: (num_joints,) or single string
            **kwargs: Additional arguments

        """
        if "left" in positions:
            self.left_arm.set_joint_positions(
                positions["left"],
                joint_names.get("left") if joint_names else None,
                **kwargs
            )
        if "right" in positions:
            self.right_arm.set_joint_positions(
                positions["right"],
                joint_names.get("right") if joint_names else None,
                **kwargs
            )

    def get_joint_positions(
        self,
        arm: str = "both",
        joint_names: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get the current joint position(s) for the specified arm(s).

        Args:
            arm: Which arm(s) to get joint position(s) for ("left", "right", or "both")
            joint_names: Dictionary with keys 'left' and/or 'right', each containing name(s) of the joint(s) to get velocity for.
                         If None, returns all joint position(s) for the specified arm(s).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing current joint velocity(ies) for the specified arm(s).
                                   Shape of each arm's position(s): (num_requested_joints,)
        """
        result = {}
        if arm in ["left", "both"]:
            result["left"] = self.left_arm.get_joint_positions(
                joint_names.get("left") if joint_names else None
            )
        if arm in ["right", "both"]:
            result["right"] = self.right_arm.get_joint_positions(
                joint_names.get("right") if joint_names else None
            )
        return result

    def get_joint_velocities(
        self,
        arm: str = "both",
        joint_names: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get the current joint velocity(ies) for the specified arm(s).

        Args:
            arm: Which arm(s) to get joint velocities for ("left", "right", or "both")
            joint_names: Dictionary with keys 'left' and/or 'right', each containing name(s) of the joint(s) to get velocity for.
                         If None, returns all joint velocities for the specified arm(s).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing current joint velocity(ies) for the specified arm(s).
                                   Shape of each arm's velocities: (num_requested_joints,)
        """
        result = {}
        if arm in ["left", "both"]:
            result["left"] = self.left_arm.get_joint_velocities(
                joint_names.get("left") if joint_names else None
            )
        if arm in ["right", "both"]:
            result["right"] = self.right_arm.get_joint_velocities(
                joint_names.get("right") if joint_names else None
            )
        return result

    def set_ee_pose(self, poses: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Set the end-effector poses for both arms.

        Args:
            poses: Dict with keys 'left' and 'right', each containing a tuple of (position, orientation)
        """
        if "left" in poses:
            position, orientation = poses["left"]
            self.left_arm.set_ee_pose(position, orientation)

        if "right" in poses:
            position, orientation = poses["right"]
            self.right_arm.set_ee_pose(position, orientation)

    def set_ee_pose_rpy(
        self,
        xyzrpy: Dict[str, Tuple[np.ndarray]],
        **kwargs
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, rol, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments

        """
        if "left" in xyzrpy:
            position, orientation = xyzrpy["left"]
            self.left_arm.set_ee_pose_rpy(position, orientation)

        if "right" in xyzrpy:
            position, orientation = xyzrpy["right"]
            self.right_arm.set_ee_pose_rpy(position, orientation)

    def get_ee_pose(
        self, arm: str = "both"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the current end-effector pose(s) for the specified arm(s).

        Args:
            arm (str): Which arm(s) to get the end-effector pose for ("left", "right", or "both")

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary containing current end-effector pose(s) for the specified arm(s).
                                                      Each pose is a tuple of (position, orientation).
                                                      Position shape: (3,)
                                                      Orientation shape: (4,) (quaternion [x, y, z, w])
        """
        result = {}
        if arm in ["left", "both"]:
            result["left"] = self.left_arm.get_ee_pose()
        if arm in ["right", "both"]:
            result["right"] = self.right_arm.get_ee_pose()
        return result

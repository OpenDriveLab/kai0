#!/usr/bin/env python3
# -*-coding:utf8-*-
# 本文件为同时打开主从臂的节点，通过mode参数控制是读取还是控制
# 默认认为从臂有夹爪
# mode为0时为发送主从臂消息，
# mode为1时为控制从臂，不发送主臂消息，此时如果要控制从臂，需要给主臂的topic发送消息
from typing import (
    Optional,
)
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import argparse
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_msgs.msg import PiperStatusMsg, PosCmd
from geometry_msgs.msg import Pose,PoseStamped
from tf.transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换

#"""Arm IK
import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
import os, sys
try:
    import termios
    import tty
except ImportError:
    import msvcrt
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        last_path = os.path.dirname(current_dir)
        last_path = os.path.dirname(last_path)
        last_path = os.path.dirname(last_path)
        urdf_dir = os.path.join("/home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf")
        # urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'
        urdf_path = urdf_dir
    
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # q = quaternion_from_euler(0, -1.57, 0)
        # q = quaternion_from_euler(0, -1.57, -1.57)
        q = quaternion_from_euler(0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        rospy.loginfo("self.reduced_robot.model.lowerPositionLimit: %s", self.reduced_robot.model.lowerPositionLimit)
        rospy.loginfo("self.reduced_robot.model.upperPositionLimit: %s", self.reduced_robot.model.upperPositionLimit)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(200 * self.totalcost + 0.001 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth
        
        # least accurate
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)        
        # opts = {
        #     'ipopt': {
        #         'print_level': 0,
        #         'max_iter': 10, #400, #50,
        #         'tol': 1e-2 #1e-8 #1e-4
        #     },
        #     'print_time': False
        # }

        # original
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)        
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50, #400, #50,
                'tol': 1e-4 #1e-8 #1e-4
            },
            'print_time': False
        }

        # most accurate (false)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)        
        # opts = {
        #     'ipopt': {
        #         'print_level': 0,
        #         'max_iter': 1500, #400, #50,
        #         'tol': 1e-8 #1e-8 #1e-4
        #     },
        #     'print_time': False
        # }

        # most accurate:
        # self.opti.minimize(200 * self.totalcost + 0.001 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth
        # opts = {
        #     'ipopt': {
        #         'print_level': 0,
        #         'max_iter': 400, #400, #50,
        #         'tol': 1e-4 #1e-8 #1e-4
        #     },
        #     'print_time': False
        # }

        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision, eef_euler = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision, eef_euler

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        
        
        # 假设末端执行器是机器人模型中的最后一个链节
        eef_pose = self.robot.data.oMi[-1]  # 获取末端执行器的位姿矩阵

        # 提取末端执行器的位置 (从 4x4 位姿矩阵中提取平移部分)
        eef_position = eef_pose.translation  # 这是一个 3x1 向量，表示末端执行器的位置
        eef_rotation = eef_pose.rotation
        
        euler = R.from_matrix(eef_rotation).as_euler('xyz', degrees=False)
        return collision, np.concatenate([eef_position.reshape(1,-1), euler.reshape(1,-1)], axis=-1).squeeze(0)
    
    def forward_k(self, q):# -> ndarray[Any, dtype]:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        # 假设末端执行器是机器人模型中的最后一个链节
        eef_pose = self.reduced_robot.data.oMi[-1]  # 获取末端执行器的位姿矩阵

        # 提取末端执行器的位置 (从 4x4 位姿矩阵中提取平移部分)
        eef_position = eef_pose.translation  # 这是一个 3x1 向量，表示末端执行器的位置
        eef_rotation = eef_pose.rotation
        
        euler = R.from_matrix(eef_rotation).as_euler('xyz', degrees=False)
        return np.concatenate([eef_position.reshape(1,-1), euler.reshape(1,-1)], axis=-1).squeeze(0)


    def get_ik_solution(self, x,y,z,roll,pitch,yaw):
        
        q = quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        # print(target)
        sol_q, tau_ff, get_result, sol_eef = self.ik_fun(target.homogeneous,0)
        print("solved qpos:", sol_q)
        
        if get_result :
            # print(sol_q)
            print("get result")
            # piper_control.joint_control_piper(sol_q[0],sol_q[1],sol_q[2],sol_q[3],sol_q[4],sol_q[5],0)
        else :
            print("collision!!!")
        
        return sol_q, tau_ff, sol_eef, target.homogeneous
    
    def get_target(self, eef_euler):
        q = quaternion_from_euler(eef_euler[3], eef_euler[4], eef_euler[5], axes='rzyx')
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([eef_euler[0], eef_euler[1], eef_euler[2]]),
        )
        return target.homogeneous

#Arm IK"""




def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

class C_PiperRosNode():
    """机械臂ros节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('piper_start_all_node', anonymous=True)
        
        self.arm_ik = Arm_IK()
        rospy.loginfo("Initializing Arm_IK...")

        self.can_port = "can0"
        if rospy.has_param('~can_port'):
            self.can_port = rospy.get_param("~can_port")
            rospy.loginfo("%s is %s", rospy.resolve_name('~can_port'), self.can_port)
        else: 
            rospy.loginfo("未找到can_port参数,请输入 _can_port:=can0 类似的格式")
            exit(0)
        # 模式，模式为1的时候，才能够控制从臂
        self.mode = 0
        if rospy.has_param('~mode'):
            self.mode = rospy.get_param("~mode")
            rospy.loginfo("%s is %s", rospy.resolve_name('~mode'), self.mode)
        else:
            rospy.loginfo("未找到mode参数,请输入 _mode:=0 类似的格式")
            exit(0)

        # 是否自动使能，默认不自动使能，只有模式为1的时候才能够被设置为自动使能
        self.auto_enable = False
        if rospy.has_param('~auto_enable'):
            if(rospy.get_param("~auto_enable") and self.mode == 1):
                self.auto_enable = True
        rospy.loginfo("%s is %s", rospy.resolve_name('~auto_enable'), self.auto_enable)
        # publish
        self.joint_std_pub_puppet = rospy.Publisher('/puppet/joint_states', JointState, queue_size=10)
        # 默认模式为0，读取主从臂消息
        if(self.mode == 0):
            self.joint_std_pub_master = rospy.Publisher('/master/joint_states', JointState, queue_size=10)
        self.arm_status_pub = rospy.Publisher('/puppet/arm_status', PiperStatusMsg, queue_size=10)
        self.end_pose_pub = rospy.Publisher('/puppet/end_pose', PoseStamped, queue_size=10)
        
        self.__enable_flag = False
        self.gripper_exist = True
        # 从臂消息
        self.joint_state_slave = JointState()
        self.joint_state_slave.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_slave.position = [0.0] * 7
        self.joint_state_slave.velocity = [0.0] * 7
        self.joint_state_slave.effort = [0.0] * 7
        # 主臂消息
        self.joint_state_master = JointState()
        self.joint_state_master.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_master.position = [0.0] * 7
        self.joint_state_master.velocity = [0.0] * 7
        self.joint_state_master.effort = [0.0] * 7

        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        # 模式为1的时候，订阅控制消息
        if(self.mode == 1):
            sub_pos_th = threading.Thread(target=self.SubPosThread)
            sub_joint_th = threading.Thread(target=self.SubJointThread)
            sub_enable_th = threading.Thread(target=self.SubEnableThread)
            
            sub_pos_th.daemon = True
            sub_joint_th.daemon = True
            sub_enable_th.daemon = True
            
            sub_pos_th.start()
            sub_joint_th.start()
            sub_enable_th.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def Publish(self):
        """机械臂消息发布
        """
        rate = rospy.Rate(200)  # 200 Hz
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not rospy.is_shutdown():
            if(self.auto_enable and self.mode == 1):
                while not (enable_flag):
                    elapsed_time = time.time() - start_time
                    print("--------------------")
                    enable_flag = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                        self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
                    print("使能状态:",enable_flag)
                    if(enable_flag):
                        self.__enable_flag = True
                    self.piper.EnableArm(7)
                    self.piper.GripperCtrl(0,1000,0x01, 0)
                    print("--------------------")
                    # 检查是否超过超时时间
                    if elapsed_time > timeout:
                        print("超时....")
                        elapsed_time_flag = True
                        enable_flag = True
                        break
                    time.sleep(1)
                    pass
            if(elapsed_time_flag):
                print("程序自动使能超时,退出程序")
                exit(0)
            # 发布消息
            self.PublishSlaveArmJointAndGripper()
            self.PublishSlaveArmState()
            self.PublishSlaveArmEndPose()
            # 模式为0的时候，发布主臂消息
            if(self.mode == 0):
                self.PublishMasterArmJointAndGripper()

            rate.sleep()
    
    def PublishSlaveArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        self.arm_status_pub.publish(arm_status)
    
    def PublishSlaveArmEndPose(self):
        # 末端位姿
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis/1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis/1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis/1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis/1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis/1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis/1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        # 为末端位姿增加时间戳
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)
    
    def PublishSlaveArmJointAndGripper(self):
        # 从臂反馈消息
        self.joint_state_slave.header.stamp = rospy.Time.now()
        joint_0:float = (self.piper.GetArmJointMsgs().joint_state.joint_1/1000) * 0.017444
        joint_1:float = (self.piper.GetArmJointMsgs().joint_state.joint_2/1000) * 0.017444
        joint_2:float = (self.piper.GetArmJointMsgs().joint_state.joint_3/1000) * 0.017444
        joint_3:float = (self.piper.GetArmJointMsgs().joint_state.joint_4/1000) * 0.017444
        joint_4:float = (self.piper.GetArmJointMsgs().joint_state.joint_5/1000) * 0.017444
        joint_5:float = (self.piper.GetArmJointMsgs().joint_state.joint_6/1000) * 0.017444
        joint_6:float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle/1000000
        vel_0:float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed/1000
        vel_1:float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed/1000
        vel_2:float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed/1000
        vel_3:float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed/1000
        vel_4:float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed/1000
        vel_5:float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed/1000
        effort_6:float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort/1000
        self.joint_state_slave.position = [joint_0,joint_1, joint_2, joint_3, joint_4, joint_5,joint_6]  # Example values
        self.joint_state_slave.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]  # Example values
        self.joint_state_slave.effort[6] = effort_6
        self.joint_std_pub_puppet.publish(self.joint_state_slave)
    
    def PublishMasterArmJointAndGripper(self):
        # 主臂控制消息
        self.joint_state_master.header.stamp = rospy.Time.now()
        joint_0:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_1/1000) * 0.017444
        joint_1:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_2/1000) * 0.017444
        joint_2:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_3/1000) * 0.017444
        joint_3:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_4/1000) * 0.017444
        joint_4:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_5/1000) * 0.017444
        joint_5:float = (self.piper.GetArmJointCtrl().joint_ctrl.joint_6/1000) * 0.017444
        joint_6:float = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle/1000000
        self.joint_state_master.position = [joint_0,joint_1, joint_2, joint_3, joint_4, joint_5,joint_6]  # Example values
        self.joint_std_pub_master.publish(self.joint_state_master)
    
    def SubPosThread(self):
        """机械臂末端位姿订阅
        
        """
        rospy.Subscriber('pos_cmd', PosCmd, self.pos_callback)
        rospy.spin()
    
    def SubJointThread(self):
        """机械臂关节订阅
        
        """
        rospy.Subscriber('/master/joint_states', JointState, self.joint_callback)
        rospy.spin()
    
    def SubEnableThread(self):
        """机械臂使能
        
        """
        rospy.Subscriber('/enable_flag', Bool, self.enable_callback)
        rospy.spin()

    def pos_callback(self, pos_data):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data (): 
        """
        x = pos_data.x
        y = pos_data.y
        z = pos_data.z
        roll = pos_data.roll
        pitch = pos_data.pitch
        yaw = pos_data.yaw

        # 调用Arm_IK类的逆解函数
        joint, _, _, _ = self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw)
        
        factor = 57324.840764 #1000*180/3.14
        factor1 = 57.32484
        rospy.loginfo("Received Joint States:")
        rospy.loginfo("joint_0: %f", joint[0]*1)
        rospy.loginfo("joint_1: %f", joint[1]*1)
        rospy.loginfo("joint_2: %f", joint[2]*1)
        rospy.loginfo("joint_3: %f", joint[3]*1)
        rospy.loginfo("joint_4: %f", joint[4]*1)
        rospy.loginfo("joint_5: %f", joint[5]*1)
        rospy.loginfo("gripper: %f", pos_data.gripper)
        joint_0 = round(joint[0]*factor)
        joint_1 = round(joint[1]*factor)
        joint_2 = round(joint[2]*factor)
        joint_3 = round(joint[3]*factor)
        joint_4 = round(joint[4]*factor)
        joint_5 = round(joint[5]*factor)
        gripper = round(pos_data.gripper*1000*1000)
        if(gripper>80000): gripper = 80000
        if(gripper<0): gripper = 0
        if(self.GetEnableFlag()):
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, 
                                    joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            pass
        
        # factor = 180 / 3.1415926
        # x = round(pos_data.x*1000) * 1000
        # y = round(pos_data.y*1000) * 1000
        # z = round(pos_data.z*1000) * 1000
        # rx = round(pos_data.roll*1000*factor) 
        # ry = round(pos_data.pitch*1000*factor)
        # rz = round(pos_data.yaw*1000*factor)
        # rospy.loginfo("Received PosCmd:")
        # rospy.loginfo("x: %f", x)
        # rospy.loginfo("y: %f", y)
        # rospy.loginfo("z: %f", z)
        # rospy.loginfo("roll: %f", rx)
        # rospy.loginfo("pitch: %f", ry)
        # rospy.loginfo("yaw: %f", rz)
        # rospy.loginfo("gripper: %f", pos_data.gripper)
        # # rospy.loginfo("mode1: %d", pos_data.mode1)
        # # rospy.loginfo("mode2: %d", pos_data.mode2)
        # if(self.GetEnableFlag()):
        #     self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
        #     self.piper.MotionCtrl_2(0x01, 0x00, 100)
        #     self.piper.EndPoseCtrl(x, y, z, 
        #                             rx, ry, rz)
        #     gripper = round(pos_data.gripper*1000*1000)
        #     # if(pos_data.gripper>80000): gripper = 80000
        #     # if(pos_data.gripper<0): gripper = 0
        #     if(gripper>80000): gripper = 80000
        #     if(gripper<0): gripper = 0
        #     if(self.gripper_exist):
        #         self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
        #         self.piper.MotionCtrl_2(0x01, 0x00, 100)
    
    def joint_callback(self, joint_data):
        """机械臂关节角回调函数

        Args:
            joint_data (): 
        """
        factor = 57324.840764 #1000*180/3.14
        factor1 = 57.32484
        rospy.loginfo("Received Joint States:")
        rospy.loginfo("joint_0: %f", joint_data.position[0]*1)
        rospy.loginfo("joint_1: %f", joint_data.position[1]*1)
        rospy.loginfo("joint_2: %f", joint_data.position[2]*1)
        rospy.loginfo("joint_3: %f", joint_data.position[3]*1)
        rospy.loginfo("joint_4: %f", joint_data.position[4]*1)
        rospy.loginfo("joint_5: %f", joint_data.position[5]*1)
        rospy.loginfo("joint_6: %f", joint_data.position[6]*1)
        joint_0 = round(joint_data.position[0]*factor)
        joint_1 = round(joint_data.position[1]*factor)
        joint_2 = round(joint_data.position[2]*factor)
        joint_3 = round(joint_data.position[3]*factor)
        joint_4 = round(joint_data.position[4]*factor)
        joint_5 = round(joint_data.position[5]*factor)
        joint_6 = round(joint_data.position[6]*1000*1000)
        if(joint_6>80000): joint_6 = 80000
        if(joint_6<0): joint_6 = 0
        if(self.GetEnableFlag()):
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, 
                                    joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x01, 100)
            pass
    
    def enable_callback(self, enable_flag:Bool):
        """机械臂使能回调函数

        Args:
            enable_flag (): 
        """
        rospy.loginfo("Received enable flag:")
        rospy.loginfo("enable_flag: %s", enable_flag.data)
        if(enable_flag.data):
            self.__enable_flag = True
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0,1000,0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            self.piper.GripperCtrl(0,1000,0x00, 0)

if __name__ == '__main__':
    try:
        piper_ms = C_PiperRosNode()
        piper_ms.Publish()
    except rospy.ROSInterruptException:
        pass


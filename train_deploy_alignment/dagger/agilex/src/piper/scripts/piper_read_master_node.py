#!/usr/bin/env python3
# -*-coding:utf8-*-
# 本文件为读取主臂发送的消息，当机械臂设置为主臂，主臂只发送关节角消息和控制指令
from typing import (
    Optional,
)
from piper_sdk import *

import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

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
        rospy.init_node('piper_read_master_node', anonymous=True)

        self.can_port = "can0"
        if rospy.has_param('~can_port'):
            self.can_port = rospy.get_param("~can_port")
            rospy.loginfo("%s is %s", rospy.resolve_name('~can_port'), self.can_port)
        else: 
            rospy.loginfo("未找到can_port参数,请输入 _can_port:=can0 类似的格式")
            exit(0)

        self.joint_std_pub_master = rospy.Publisher('/master/joint_states', JointState, queue_size=1, tcp_nodelay=True)

        self.joint_state_master = JointState()
        self.joint_state_master.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_master.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0]
        self.joint_state_master.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0]
        self.joint_state_master.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0]

        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

    def Pubilsh(self):
        """机械臂消息发布
        """
        rate = rospy.Rate(200)  # 200 Hz
        while not rospy.is_shutdown():
            self.joint_state_master.header.stamp = rospy.Time.now()
            # Here, you can set the joint positions to any value you want
            joint_0 = self.piper.GetArmJointCtrl().joint_ctrl.joint_1/1000 * 0.017444
            joint_1 = self.piper.GetArmJointCtrl().joint_ctrl.joint_2/1000 * 0.017444
            joint_2 = self.piper.GetArmJointCtrl().joint_ctrl.joint_3/1000 * 0.017444
            joint_3 = self.piper.GetArmJointCtrl().joint_ctrl.joint_4/1000 * 0.017444
            joint_4 = self.piper.GetArmJointCtrl().joint_ctrl.joint_5/1000 * 0.017444
            joint_5 = self.piper.GetArmJointCtrl().joint_ctrl.joint_6/1000 * 0.017444
            joint_6 = self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle/1000000
            self.joint_state_master.position = [joint_0,joint_1, joint_2, joint_3, joint_4, joint_5,joint_6]  # Example values

            self.joint_std_pub_master.publish(self.joint_state_master)
            rate.sleep()

if __name__ == '__main__':
    try:
        piper_all = C_PiperRosNode()
        piper_all.Pubilsh()
    except rospy.ROSInterruptException:
        pass


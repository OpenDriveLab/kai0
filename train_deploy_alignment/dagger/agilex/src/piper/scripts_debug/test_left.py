#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单展示版：直接让从臂抬高到展示位置
"""

import rospy
import time
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header

class SimpleArmDemo:
    def __init__(self):
        rospy.init_node('simple_arm_demo', anonymous=True)
        rospy.sleep(1)
        
        # 发布器 /master/joint_left  /master_controled/joint_left
        self.slave_joint_pub = rospy.Publisher('/master_controled/joint_left', JointState, queue_size=10)
        self.slave_enable_pub = rospy.Publisher('/puppet/enable_left', Bool, queue_size=10)
        
        # 订阅器
        self.slave_joints_sub = rospy.Subscriber('/puppet_master/joint_right', JointState, self.slave_joints_callback)
        
        # 状态变量
        self.current_joints = None
        
        # 🎯 简单的展示位置：抬高手臂
        self.demo_position = [0.3, -0.5, -0.8, -0.5, -0.3, 0.0, 0.5]  # 抬高展示位置
        
        rospy.loginfo("Simple arm demo initialized")
    
    def slave_joints_callback(self, msg):
        """从臂状态回调"""
        self.current_joints = msg
        if len(msg.position) >= 7:
            degrees = [math.degrees(p) for p in msg.position[:6]]
            rospy.loginfo_throttle(2, "Current: J0=%.1f° J1=%.1f° J2=%.1f° J3=%.1f° J4=%.1f° J5=%.1f°", 
                                 degrees[0], degrees[1], degrees[2], degrees[3], degrees[4], degrees[5])
    
    def wait_for_connection(self):
        """等待连接"""
        rospy.loginfo("🔍 Waiting for slave arm...")
        
        while not rospy.is_shutdown() and self.current_joints is None:
            rospy.sleep(0.1)
        
        if self.current_joints:
            rospy.loginfo("✅ Connected to slave arm")
            return True
        return False
    
    def enable_slave_arm(self):
        """使能从臂"""
        rospy.loginfo("🔧 Enabling slave arm...")
        enable_msg = Bool(data=True)
        
        for i in range(8):
            self.slave_enable_pub.publish(enable_msg)
            rospy.loginfo("Enable command %d/8", i+1)
            rospy.sleep(0.5)
        
        rospy.loginfo("✅ Slave arm enabled")
        rospy.sleep(2)
    
    def move_to_demo_position(self):
        """移动到展示位置"""
        if self.current_joints is None:
            rospy.logerr("❌ No joint data")
            return False
        
        rospy.loginfo("🎯 Moving to demo position...")
        
        # 显示目标位置
        target_degrees = [math.degrees(p) for p in self.demo_position[:6]]
        rospy.loginfo("Target: J0=%.1f° J1=%.1f° J2=%.1f° J3=%.1f° J4=%.1f° J5=%.1f°", 
                     target_degrees[0], target_degrees[1], target_degrees[2], 
                     target_degrees[3], target_degrees[4], target_degrees[5])
        
        # 创建关节指令
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        joint_msg.position = self.demo_position
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        
        # 连续发送指令
        rospy.loginfo("🚀 Sending movement commands...")
        rate = rospy.Rate(10)  # 10Hz
        
        for i in range(50):  # 发送5秒
            self.slave_joint_pub.publish(joint_msg)
            
            if i % 10 == 0:
                rospy.loginfo("Sending command %d/50", i+1)
            
            rate.sleep()
        
        rospy.loginfo("✅ Movement commands completed")
        return True
    
    def hold_position(self, duration=10):
        """保持位置"""
        rospy.loginfo("📍 Holding demo position for %d seconds...", duration)
        
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        joint_msg.position = self.demo_position
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        
        rate = rospy.Rate(5)  # 5Hz
        start_time = time.time()
        
        while not rospy.is_shutdown() and (time.time() - start_time) < duration:
            joint_msg.header.stamp = rospy.Time.now()
            self.slave_joint_pub.publish(joint_msg)
            
            remaining = duration - (time.time() - start_time)
            if int(remaining) % 2 == 0 and remaining != int(remaining):  # 每2秒显示一次
                rospy.loginfo("Holding position... %.1f seconds remaining", remaining)
            
            rate.sleep()
        
        rospy.loginfo("✅ Position holding completed")
    
    def return_to_home(self):
        """回到初始位置"""
        rospy.loginfo("🏠 Returning to home position...")
        
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
        
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        joint_msg.position = home_position
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        
        rate = rospy.Rate(10)
        for i in range(30):  # 发送3秒
            joint_msg.header.stamp = rospy.Time.now()
            self.slave_joint_pub.publish(joint_msg)
            rate.sleep()
        
        rospy.loginfo("✅ Returned to home")
    
    def run_demo(self):
        """运行完整演示"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("🎭 STARTING SIMPLE ARM DEMO")
        rospy.loginfo("=" * 50)
        
        # 1. 连接
        if not self.wait_for_connection():
            rospy.logerr("❌ Failed to connect")
            return False
        
        # 2. 使能
        self.enable_slave_arm()
        
        # 3. 移动到展示位置
        rospy.loginfo("\n🎯 STEP 1: Moving to demo position...")
        if not self.move_to_demo_position():
            rospy.logerr("❌ Failed to move")
            return False
        
        # 4. 保持展示位置
        rospy.loginfo("\n📍 STEP 2: Holding demo position...")
        self.hold_position(10)  # 保持10秒
        
        # 5. 返回初始位置
        rospy.loginfo("\n🏠 STEP 3: Returning home...")
        self.return_to_home()
        
        rospy.loginfo("\n✅ DEMO COMPLETED!")
        rospy.loginfo("=" * 50)
        
        return True

def main():
    try:
        demo = SimpleArmDemo()
        
        print("=" * 50)
        print("🎭 SIMPLE ARM DEMO")
        print("=" * 50)
        print("This will:")
        print("1. Connect to slave arm")
        print("2. Enable the arm") 
        print("3. Move to demo position (arm raised)")
        print("4. Hold position for 10 seconds")
        print("5. Return to home position")
        print("=" * 50)
        
        input("Press Enter to start demo...")
        
        success = demo.run_demo()
        
        if success:
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed!")
            
    except KeyboardInterrupt:
        rospy.loginfo("🛑 User interrupted")
    except Exception as e:
        rospy.logerr("❌ Error: %s", str(e))

if __name__ == '__main__':
    main()
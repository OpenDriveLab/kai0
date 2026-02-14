#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂主从演示流程：
1. 左右从臂移动到展示pose
2. 左右主臂移动到相同pose  
3. 左右主臂切换为master模式
4. 手动拖拽主臂控制从臂
"""

import rospy
import time
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header, String, Int32
from geometry_msgs.msg import PoseStamped

class DualArmMasterSlaveDemo:
    def __init__(self):
        rospy.init_node('dual_arm_master_slave_demo', anonymous=True)
        rospy.sleep(1)
        
        # 🎯 展示位姿 (弧度) - 一个舒展的展示姿态
        self.demo_pose = [0.2, -0.4, -0.6, -0.8, -0.2, 0.0, 0.05]
        
        # ========== 左臂发布器 ==========
        # 左从臂控制
        self.slave_left_joint_pub = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
        self.slave_left_enable_pub = rospy.Publisher('/puppet/enable_left', Bool, queue_size=10)
        
        # 左主臂控制
        self.master_left_enable_pub = rospy.Publisher('/teach/master_enable_left', Bool, queue_size=10)
        self.master_left_config_pub = rospy.Publisher('/teach/master_config_left', String, queue_size=10)
        self.master_left_teach_mode_pub = rospy.Publisher('/teach/teach_mode_left', Int32, queue_size=10)
        self.master_left_joint_pub = rospy.Publisher('/master_controled/joint_left', JointState, queue_size=10)
        
        # ========== 右臂发布器 ==========
        # 右从臂控制
        self.slave_right_joint_pub = rospy.Publisher('/master/joint_right', JointState, queue_size=10)
        self.slave_right_enable_pub = rospy.Publisher('/puppet/enable_right', Bool, queue_size=10)
        
        # 右主臂控制
        self.master_right_enable_pub = rospy.Publisher('/teach/master_enable_right', Bool, queue_size=10)
        self.master_right_config_pub = rospy.Publisher('/teach/master_config_right', String, queue_size=10)
        self.master_right_teach_mode_pub = rospy.Publisher('/teach/teach_mode_right', Int32, queue_size=10)
        self.master_right_joint_pub = rospy.Publisher('/master_controled/joint_right', JointState, queue_size=10)
        
        # ========== 状态订阅器 ==========
        # 左臂状态监控
        self.slave_left_joints_sub = rospy.Subscriber('/puppet/joint_left', JointState, self.slave_left_joints_callback)
        self.master_left_joints_sub = rospy.Subscriber('/puppet_master/joint_left', JointState, self.master_left_joints_callback)
        
        # 右臂状态监控
        self.slave_right_joints_sub = rospy.Subscriber('/puppet/joint_right', JointState, self.slave_right_joints_callback)
        self.master_right_joints_sub = rospy.Subscriber('/puppet_master/joint_right', JointState, self.master_right_joints_callback)

        # ========== 状态变量 ==========
        self.slave_left_current_joints = None
        self.master_left_current_joints = None
        self.slave_right_current_joints = None
        self.master_right_current_joints = None
        
        rospy.loginfo("🎭 Dual-Arm Master-Slave Demo Controller Initialized")
        rospy.loginfo("🎯 Demo pose: %s", [f"{math.degrees(p):.1f}°" for p in self.demo_pose[:6]])

    # ========== 左臂回调函数 ==========
    def slave_left_joints_callback(self, msg):
        """左从臂状态回调"""
        self.slave_left_current_joints = msg
        if len(msg.position) >= 6:
            degrees = [math.degrees(p) for p in msg.position[:6]]
            rospy.loginfo_throttle(3, "🤖 Left Slave: J0=%.1f° J1=%.1f° J2=%.1f°", 
                                 degrees[0], degrees[1], degrees[2])
    
    def master_left_joints_callback(self, msg):
        """左主臂状态回调"""
        self.master_left_current_joints = msg
        if len(msg.position) >= 6:
            degrees = [math.degrees(p) for p in msg.position[:6]]
            rospy.loginfo_throttle(3, "🎮 Left Master: J0=%.1f° J1=%.1f° J2=%.1f°", 
                                 degrees[0], degrees[1], degrees[2])
    
    # ========== 右臂回调函数 ==========
    def slave_right_joints_callback(self, msg):
        """右从臂状态回调"""
        self.slave_right_current_joints = msg
        if len(msg.position) >= 6:
            degrees = [math.degrees(p) for p in msg.position[:6]]
            rospy.loginfo_throttle(3, "🤖 Right Slave: J0=%.1f° J1=%.1f° J2=%.1f°", 
                                 degrees[0], degrees[1], degrees[2])
    
    def master_right_joints_callback(self, msg):
        """右主臂状态回调"""
        self.master_right_current_joints = msg
        if len(msg.position) >= 6:
            degrees = [math.degrees(p) for p in msg.position[:6]]
            rospy.loginfo_throttle(3, "🎮 Right Master: J0=%.1f° J1=%.1f° J2=%.1f°", 
                                 degrees[0], degrees[1], degrees[2])
    
    def wait_for_connections(self, timeout=10):
        """等待左右主从臂连接"""
        rospy.loginfo("🔍 Waiting for dual-arm master and slave connections...")
        start_time = time.time()
        
        while not rospy.is_shutdown():
            if (self.slave_left_current_joints is not None and 
                self.master_left_current_joints is not None and
                self.slave_right_current_joints is not None and 
                self.master_right_current_joints is not None):
                rospy.loginfo("✅ All four arms connected (Left & Right Master-Slave)")
                return True
            
            if time.time() - start_time > timeout:
                rospy.logerr("❌ Connection timeout")
                return False
            
            missing = []
            if self.slave_left_current_joints is None:
                missing.append("left_slave(/puppet/joint_left)")
            if self.master_left_current_joints is None:
                missing.append("left_master(/puppet_master/joint_left)")
            if self.slave_right_current_joints is None:
                missing.append("right_slave(/puppet/joint_right)")
            if self.master_right_current_joints is None:
                missing.append("right_master(/puppet_master/joint_right)")
            
            rospy.loginfo("⏳ Waiting for: %s", ", ".join(missing))
            rospy.sleep(1)
        
        return False
    
    def create_joint_command(self, joint_positions):
        """创建关节指令消息"""
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        joint_msg.position = joint_positions
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 7
        return joint_msg
    
    # ========== 使能函数 ==========
    def enable_slave_arms(self):
        """使能左右从臂"""
        rospy.loginfo("🔧 Enabling left and right slave arms...")
        enable_msg = Bool(data=True)
        
        for i in range(8):
            # 左从臂使能
            self.slave_left_enable_pub.publish(enable_msg)
            # 右从臂使能
            self.slave_right_enable_pub.publish(enable_msg)
            rospy.loginfo("Slave arms enable %d/8", i+1)
            rospy.sleep(0.5)
        
        rospy.loginfo("✅ Both slave arms enabled")
        rospy.sleep(2)
    
    def enable_master_arms(self):
        """使能左右主臂"""
        rospy.loginfo("🔧 Enabling left and right master arms...")
        enable_msg = Bool(data=True)
        
        for i in range(5):
            # 左主臂使能
            self.master_left_enable_pub.publish(enable_msg)
            # 右主臂使能
            self.master_right_enable_pub.publish(enable_msg)
            rospy.loginfo("Master arms enable %d/5", i+1)
            rospy.sleep(0.5)
        
        rospy.loginfo("✅ Both master arms enabled")
        rospy.sleep(2)
    
    def move_slaves_to_demo_pose(self, duration=5.0):
        """移动左右从臂到展示位姿"""
        rospy.loginfo("🎯 Step 1: Moving left and right slave arms to demo pose...")
        
        # 显示目标位置
        target_degrees = [math.degrees(p) for p in self.demo_pose[:6]]
        rospy.loginfo("Target: J0=%.1f° J1=%.1f° J2=%.1f° J3=%.1f° J4=%.1f° J5=%.1f°", 
                     target_degrees[0], target_degrees[1], target_degrees[2], 
                     target_degrees[3], target_degrees[4], target_degrees[5])
        
        joint_cmd = self.create_joint_command(self.demo_pose)
        
        # 持续发送指令
        rate = rospy.Rate(10)  # 10Hz
        end_time = time.time() + duration
        
        while not rospy.is_shutdown() and time.time() < end_time:
            joint_cmd.header.stamp = rospy.Time.now()
            # 左从臂指令
            self.slave_left_joint_pub.publish(joint_cmd)
            # 右从臂指令
            self.slave_right_joint_pub.publish(joint_cmd)
            
            remaining = end_time - time.time()
            if int(remaining) % 2 == 1 and remaining != int(remaining):
                rospy.loginfo("Moving slaves... %.1fs remaining", remaining)
            
            rate.sleep()
        
        rospy.loginfo("✅ Both slave arms reached demo pose")
        rospy.sleep(1)
    
    def move_masters_to_demo_pose(self, duration=5.0):
        """移动左右主臂到展示位姿"""
        rospy.loginfo("🎮 Step 2: Moving left and right master arms to demo pose...")
        
        joint_cmd = self.create_joint_command(self.demo_pose)
        
        # 持续发送指令
        rate = rospy.Rate(10)  # 10Hz
        end_time = time.time() + duration
        
        while not rospy.is_shutdown() and time.time() < end_time:
            joint_cmd.header.stamp = rospy.Time.now()
            # 左主臂指令
            self.master_left_joint_pub.publish(joint_cmd)
            # 右主臂指令
            self.master_right_joint_pub.publish(joint_cmd)
            
            remaining = end_time - time.time()
            if int(remaining) % 2 == 1 and remaining != int(remaining):
                rospy.loginfo("Moving masters... %.1fs remaining", remaining)
            
            rate.sleep()
        
        rospy.loginfo("✅ Both master arms reached demo pose")
        rospy.sleep(1)
    
    def switch_masters_to_teach_mode(self):
        """切换左右主臂到示教模式"""
        rospy.loginfo("🔄 Step 3: Switching both master arms to teaching mode...")
        
        # 1. 切换到主臂配置 (0xFA)
        rospy.loginfo("📍 Setting master configuration (0xFA) for both arms...")
        config_msg = String(data="master")
        for i in range(5):
            # 左主臂配置
            self.master_left_config_pub.publish(config_msg)
            # 右主臂配置
            self.master_right_config_pub.publish(config_msg)
            rospy.loginfo("Config command %d/5", i+1)
            rospy.sleep(0.5)
        
        rospy.sleep(3)  # 等待配置生效
        
        # 2. 进入拖动示教模式
        rospy.loginfo("📍 Entering drag teaching mode for both arms...")
        teach_msg = Int32(data=1)
        for i in range(5):
            # 左主臂示教模式
            self.master_left_teach_mode_pub.publish(teach_msg)
            # 右主臂示教模式
            self.master_right_teach_mode_pub.publish(teach_msg)
            rospy.loginfo("Teach mode command %d/5", i+1)
            rospy.sleep(0.5)
        
        rospy.loginfo("✅ Both master arms switched to teaching mode")
        rospy.loginfo("🎉 You can now manually drag both master arms!")
        rospy.loginfo("🤖 The slave arms will follow master movements")
        rospy.sleep(2)
    
    def monitor_dual_arm_operation(self):
        """监控双臂主从操作"""
        rospy.loginfo("👁️ Monitoring dual-arm master-slave operation...")
        rospy.loginfo("🛑 Press Ctrl+C to stop monitoring")
        
        try:
            rate = rospy.Rate(1)  # 1Hz 监控频率
            while not rospy.is_shutdown():
                
                # 显示当前状态
                rospy.loginfo("=" * 60)
                
                # 左臂状态
                if (self.master_left_current_joints and self.slave_left_current_joints and 
                    len(self.master_left_current_joints.position) >= 6 and 
                    len(self.slave_left_current_joints.position) >= 6):
                    
                    master_left_deg = [math.degrees(p) for p in self.master_left_current_joints.position[:3]]
                    slave_left_deg = [math.degrees(p) for p in self.slave_left_current_joints.position[:3]]
                    
                    rospy.loginfo("🎮 Left Master:  J0=%.1f° J1=%.1f° J2=%.1f°", 
                                master_left_deg[0], master_left_deg[1], master_left_deg[2])
                    rospy.loginfo("🤖 Left Slave:   J0=%.1f° J1=%.1f° J2=%.1f°", 
                                slave_left_deg[0], slave_left_deg[1], slave_left_deg[2])
                else:
                    rospy.loginfo("⚠️ Left arm data missing")
                
                rospy.loginfo("-" * 30)
                
                # 右臂状态
                if (self.master_right_current_joints and self.slave_right_current_joints and 
                    len(self.master_right_current_joints.position) >= 6 and 
                    len(self.slave_right_current_joints.position) >= 6):
                    
                    master_right_deg = [math.degrees(p) for p in self.master_right_current_joints.position[:3]]
                    slave_right_deg = [math.degrees(p) for p in self.slave_right_current_joints.position[:3]]
                    
                    rospy.loginfo("🎮 Right Master: J0=%.1f° J1=%.1f° J2=%.1f°", 
                                master_right_deg[0], master_right_deg[1], master_right_deg[2])
                    rospy.loginfo("🤖 Right Slave:  J0=%.1f° J1=%.1f° J2=%.1f°", 
                                slave_right_deg[0], slave_right_deg[1], slave_right_deg[2])
                else:
                    rospy.loginfo("⚠️ Right arm data missing")
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("🛑 Monitoring stopped by user")
    
    def run_demo_sequence(self):
        """运行完整双臂演示序列"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("🎭 DUAL-ARM MASTER-SLAVE DEMONSTRATION SEQUENCE")
        rospy.loginfo("=" * 60)
        rospy.loginfo("📋 Sequence:")
        rospy.loginfo("  1. Move both slave arms to demo pose")
        rospy.loginfo("  2. Move both master arms to same pose")  
        rospy.loginfo("  3. Switch both master arms to teaching mode")
        rospy.loginfo("  4. Manual control enabled for both arms")
        rospy.loginfo("=" * 60)
        
        # 步骤0: 等待连接
        if not self.wait_for_connections():
            rospy.logerr("❌ Failed to connect to all arms")
            return False
        
        # 步骤1: 使能机械臂
        rospy.loginfo("\n🔧 Initializing all arms...")
        self.enable_slave_arms()
        self.enable_master_arms()
        
        # 步骤2: 移动从臂到展示位置
        rospy.loginfo("\n" + "="*50)
        self.move_slaves_to_demo_pose()
        
        # 步骤3: 移动主臂到展示位置
        rospy.loginfo("\n" + "="*50)
        self.move_masters_to_demo_pose()
        
        # 步骤4: 切换主臂到示教模式
        rospy.loginfo("\n" + "="*50)
        self.switch_masters_to_teach_mode()
        
        # 步骤5: 开始监控
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("🎉 DUAL-ARM SETUP COMPLETE - MANUAL CONTROL ACTIVE!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("📋 Instructions:")
        rospy.loginfo("  • Gently drag the LEFT MASTER arm to control left slave")
        rospy.loginfo("  • Gently drag the RIGHT MASTER arm to control right slave")
        rospy.loginfo("  • Both slave arms will follow their respective masters")
        rospy.loginfo("  • Press Ctrl+C to exit when finished")
        rospy.loginfo("=" * 60)
        
        self.monitor_dual_arm_operation()
        
        return True

def main():
    try:
        demo = DualArmMasterSlaveDemo()
        
        print("=" * 60)
        print("🎭 DUAL-ARM MASTER-SLAVE DEMONSTRATION")
        print("=" * 60)
        print("This demo will:")
        print("1. 🤖 Move both slave arms to demonstration pose")
        print("2. 🎮 Move both master arms to matching pose")
        print("3. 🔄 Switch both master arms to teaching mode")
        print("4. ✋ Enable manual drag control for both arms")
        print("=" * 60)
        print("Requirements:")
        print("• All four arms (L/R master & L/R slave) must be connected")
        print("• Arms should be in safe starting positions")
        print("• Ensure workspace is clear for movement")
        print("=" * 60)
        
        input("📋 Press Enter to start the dual-arm demonstration sequence...")
        
        success = demo.run_demo_sequence()
        
        if success:
            rospy.loginfo("✅ Dual-arm demonstration completed successfully!")
        else:
            rospy.logerr("❌ Dual-arm demonstration failed!")
            
    except KeyboardInterrupt:
        rospy.loginfo("🛑 Demo interrupted by user")
    except Exception as e:
        rospy.logerr("❌ Demo error: %s", str(e))

if __name__ == '__main__':
    main()
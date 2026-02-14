#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts_debug/piper_start_master_node_debug_fixed.py

import rospy
import time
from piper_sdk import C_PiperInterface
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
from geometry_msgs.msg import Twist

class MasterArmController:
    def __init__(self):
        rospy.init_node('master_arm_controller_test', anonymous=True)
        
        # 连接到独立的主臂CAN接口
        self.can_port = rospy.get_param('~can_port', 'can_left_master')
        rospy.loginfo(f"使用CAN端口: {self.can_port}")
        
        self.piper = C_PiperInterface(self.can_port)
        
        # ROS发布器
        self.joint_pub = rospy.Publisher('/master/joint_states', JointState, queue_size=1)
        self.status_pub = rospy.Publisher('/master/arm_status', Twist, queue_size=1)
        self.mode_status_pub = rospy.Publisher('/master/mode_status', String, queue_size=1)
        
        # ROS订阅器 - 用于接收控制指令
        self.joint_sub = rospy.Subscriber('/master/joint_cmd', JointState, self.joint_callback)
        self.enable_sub = rospy.Subscriber('/master/enable', Bool, self.enable_callback)
        
        # 新增：示教模式控制订阅器
        self.teach_mode_sub = rospy.Subscriber('/master/teach_mode', Int32, self.teach_mode_callback)
        
        # 新增：主从模式切换订阅器
        self.master_slave_mode_sub = rospy.Subscriber('/master/master_slave_mode', String, self.master_slave_mode_callback)
        
        # 状态变量
        self.is_enabled = False
        self.connection_ok = False
        self.in_teach_mode = False  # 默认不在示教模式
        self.current_master_slave_mode = "slave"  # 当前模式：只有master和slave
        
        # 控制参数
        self.control_rate = rospy.get_param('~control_rate', 10)  # 控制频率
        
        # 主从模式配置字典 - 修正版：删除independent模式
        self.master_slave_configs = {
            "master": {
                "linkage_config": 0xFA,  # 设置为示教输入臂
                "feedback_offset": 0x00,
                "ctrl_offset": 0x00,
                "linkage_offset": 0x00,
                "description": "示教输入臂模式 - 用于拖拽示教"
            },
            "slave": {
                "linkage_config": 0xFC,  # 设置为运动输出臂
                "feedback_offset": 0x00,
                "ctrl_offset": 0x00,
                "linkage_offset": 0x00,
                "description": "运动输出臂模式 - 接收控制指令"
            }
        }
        
    def connect_arm(self):
        """连接主臂"""
        try:
            rospy.loginfo("正在连接主臂...")
            self.piper.ConnectPort()
            rospy.loginfo(f"✅ 主臂连接成功: {self.can_port}")
            self.connection_ok = True
            return True
        except Exception as e:
            rospy.logerr(f"❌ 主臂连接失败: {e}")
            self.connection_ok = False
            return False
    
    def master_slave_mode_callback(self, msg):
        """主从模式切换回调 - 修复版"""
        mode = msg.data.lower().strip()
        
        # 只支持两种有效模式
        if mode not in self.master_slave_configs:
            rospy.logwarn(f"❌ 只支持两种模式: master（拖拽示教）, slave（接收控制指令）")
            rospy.logwarn(f"   您输入的是: {mode}")
            return
            
        if mode == self.current_master_slave_mode:
            rospy.loginfo(f"📍 当前已经是 {mode} 模式")
            return
            
        try:
            rospy.loginfo(f"🔄 切换主从模式: {self.current_master_slave_mode} -> {mode}")
            
            # 获取配置
            config = self.master_slave_configs[mode]
            
            # 记录当前使能状态
            was_enabled = self.is_enabled
            
            # 发送主从模式配置指令
            self.piper.MasterSlaveConfig(
                linkage_config=config["linkage_config"],
                feedback_offset=config["feedback_offset"],
                ctrl_offset=config["ctrl_offset"],
                linkage_offset=config["linkage_offset"]
            )
            
            time.sleep(2)  # 等待配置生效
            
            # 如果之前是使能状态，确保切换后也是使能的
            if was_enabled:
                rospy.loginfo("📍 确保切换后保持使能状态...")
                self.piper.EnableArm(7)
                time.sleep(2)
                self.is_enabled = True
            
            # 根据不同模式进行相应的初始化
            if mode == "master":
                self._init_master_mode()
            elif mode == "slave":
                self._init_slave_mode()
            
            self.current_master_slave_mode = mode
            rospy.loginfo(f"✅ 主从模式切换成功: {config['description']}")
            
            # 发布模式状态
            self._publish_mode_status()
            
        except Exception as e:
            rospy.logerr(f"❌ 主从模式切换失败: {e}")
    
    def _init_master_mode(self):
        """初始化主臂模式（示教输入臂）- 修复版"""
        rospy.loginfo("🎯 初始化主臂模式 (示教输入臂)")
        
        # 主臂模式需要确保使能状态
        if self.is_enabled:
            # 重新使能以确保状态正确
            rospy.loginfo("📍 重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)
            
            # 进入拖动示教模式
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)  # 进入拖动示教模式
            self.in_teach_mode = True
            rospy.loginfo("📍 主臂模式：已进入拖动示教模式，可以手动拖拽")
    
    def _init_slave_mode(self):
        """初始化从臂模式（运动输出臂）- 修复版"""
        rospy.loginfo("🎯 初始化从臂模式 (运动输出臂)")
        
        if self.is_enabled:
            # 从臂模式需要退出拖动示教模式
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)  # 退出拖动示教模式
            self.in_teach_mode = False
            
            time.sleep(1)
            
            # 重新使能以确保状态正确
            rospy.loginfo("📍 重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)
            
            # 设置为CAN控制模式
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,     # CAN控制模式
                move_mode=0x01,     # MOVE J
                move_spd_rate_ctrl=30
            )
            rospy.loginfo("📍 从臂模式：已退出示教模式，等待接收控制指令")
    
    def _publish_mode_status(self):
        """发布当前模式状态"""
        mode_msg = String()
        config = self.master_slave_configs[self.current_master_slave_mode]
        mode_msg.data = f"{self.current_master_slave_mode}:{config['description']}"
        self.mode_status_pub.publish(mode_msg)
    
    def initialize_arm_mode(self):
        """初始化机械臂模式：默认设置为从臂模式"""
        if not self.connection_ok:
            return
            
        try:
            rospy.loginfo("🔧 正在初始化机械臂模式...")
            
            # 默认初始化为从臂模式（可接收控制指令）
            config = self.master_slave_configs["slave"]
            
            rospy.loginfo("📍 设置为从臂模式（可接收控制指令）")
            self.piper.MasterSlaveConfig(
                linkage_config=config["linkage_config"],
                feedback_offset=config["feedback_offset"],
                ctrl_offset=config["ctrl_offset"],
                linkage_offset=config["linkage_offset"]
            )
            time.sleep(2)
            
            # 失能后重新使能
            rospy.loginfo("📍 重置使能状态")
            self.piper.DisableArm(7)
            time.sleep(1)
            self.piper.EnableArm(7)
            time.sleep(2)
            self.is_enabled = True  # 确保状态同步
            
            # 设置控制模式
            rospy.loginfo("📍 设置CAN控制模式")
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,     # CAN控制模式
                move_mode=0x01,     # MOVE J
                move_spd_rate_ctrl=30  # 30%速度
            )
            time.sleep(1)
            
            # 强制退出示教模式
            rospy.loginfo("📍 退出拖动示教模式")
            self.piper.MotionCtrl_1(
                emergency_stop=0x00,
                track_ctrl=0x06,        # 终止执行
                grag_teach_ctrl=0x02    # 退出示教模式
            )
            time.sleep(1)
            
            self.in_teach_mode = False
            self.current_master_slave_mode = "slave"  # 修正：默认为slave模式
            
            rospy.loginfo("✅ 机械臂模式初始化完成 - 从臂模式（可接收控制指令）")
            
        except Exception as e:
            rospy.logerr(f"❌ 机械臂模式初始化失败: {e}")
    
    def teach_mode_callback(self, msg):
        """示教模式切换回调"""
        try:
            if msg.data == 1:
                # 进入拖动示教模式
                rospy.loginfo("🔄 切换到拖动示教模式")
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)
                self.in_teach_mode = True
                
            elif msg.data == 0:
                # 退出拖动示教模式
                rospy.loginfo("🔒 退出拖动示教模式，进入控制模式")
                self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
                self.in_teach_mode = False
                
                # 重新设置为CAN控制模式
                time.sleep(0.5)
                self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30)
                
        except Exception as e:
            rospy.logerr(f"❌ 示教模式切换失败: {e}")
    
    def enable_callback(self, msg):
        """使能/失能回调"""
        try:
            if msg.data and not self.is_enabled:
                # 使能主臂
                self.piper.EnableArm(7)
                self.is_enabled = True
                rospy.loginfo("✅ 主臂已使能")
                
                # 使能后根据当前模式进行初始化
                time.sleep(2)
                if self.current_master_slave_mode == "master":
                    self._init_master_mode()
                elif self.current_master_slave_mode == "slave":
                    self._init_slave_mode()
                
            elif not msg.data and self.is_enabled:
                # 失能主臂
                self.piper.DisableArm(7)
                self.is_enabled = False
                rospy.loginfo("⏸️ 主臂已失能")
                
        except Exception as e:
            rospy.logerr(f"❌ 使能操作失败: {e}")
    
    def joint_callback(self, msg):
        """关节控制回调"""
        if not self.is_enabled:
            rospy.logwarn_throttle(5, "主臂未使能，忽略控制指令")
            return
            
        # 如果是主臂模式，通常不接收外部控制指令
        if self.current_master_slave_mode == "master":
            rospy.logwarn_throttle(5, "主臂模式下忽略关节控制指令（主臂用于拖拽示教）")
            return
            
        if self.in_teach_mode:
            rospy.logwarn_throttle(5, "主臂处于拖动示教模式，忽略控制指令")
            return
            
        if len(msg.position) < 6:
            rospy.logwarn("关节指令数据不完整，需要6个关节数据")
            return
            
        try:
            # 将ROS关节角度转换为机械臂控制单位 (弧度 -> 0.001度)
            joint_1 = int(msg.position[0] / 0.017444 * 1000)  # rad -> 0.001deg
            joint_2 = int(msg.position[1] / 0.017444 * 1000)
            joint_3 = int(msg.position[2] / 0.017444 * 1000)
            joint_4 = int(msg.position[3] / 0.017444 * 1000)
            joint_5 = int(msg.position[4] / 0.017444 * 1000)
            joint_6 = int(msg.position[5] / 0.017444 * 1000)
            
            # 发送控制指令
            self.piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
            rospy.loginfo_throttle(1, f"发送关节控制指令: [{joint_1}, {joint_2}, {joint_3}, {joint_4}, {joint_5}, {joint_6}]")
            
        except Exception as e:
            rospy.logerr(f"❌ 关节控制失败: {e}")
    
    def publish_joint_states(self):
        """发布关节状态"""
        if not self.connection_ok:
            return
            
        try:
            # 读取关节反馈数据
            joint_msgs = self.piper.GetArmJointMsgs()
            
            # 创建JointState消息
            joint_state = JointState()
            joint_state.header.stamp = rospy.Time.now()
            joint_state.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
            
            # 转换单位 (0.001度 -> 弧度)
            joint_state.position = [
                (joint_msgs.joint_state.joint_1 / 1000) * 0.017444,
                (joint_msgs.joint_state.joint_2 / 1000) * 0.017444,
                (joint_msgs.joint_state.joint_3 / 1000) * 0.017444,
                (joint_msgs.joint_state.joint_4 / 1000) * 0.017444,
                (joint_msgs.joint_state.joint_5 / 1000) * 0.017444,
                (joint_msgs.joint_state.joint_6 / 1000) * 0.017444
            ]
            
            # 添加速度和力矩（如果需要）
            joint_state.velocity = [0.0] * 6
            joint_state.effort = [0.0] * 6
            
            self.joint_pub.publish(joint_state)
            
        except Exception as e:
            rospy.logerr_throttle(5, f"❌ 发布关节状态失败: {e}")
    
    def publish_arm_status(self):
        """发布机械臂状态"""
        try:
            # 使用Twist消息避免PiperStatusMsg的问题
            status_msg = Twist()
            
            # 用Twist的线性和角速度字段来传递状态信息
            status_msg.linear.x = 1.0 if self.connection_ok else 0.0
            status_msg.linear.y = 1.0 if self.is_enabled else 0.0
            status_msg.linear.z = 1.0 if self.in_teach_mode else 0.0
            
            # 添加时间戳信息（用角速度字段）
            status_msg.angular.x = rospy.Time.now().to_sec()
            
            # 添加模式信息 (用角速度字段编码) - 修正版：只有两种模式
            mode_encoding = {"master": 1.0, "slave": 2.0}
            status_msg.angular.y = mode_encoding.get(self.current_master_slave_mode, 0.0)
            
            self.status_pub.publish(status_msg)
            
            # 发布模式状态
            self._publish_mode_status()
            
        except Exception as e:
            rospy.logerr_throttle(10, f"❌ 发布状态失败: {e}")
    
    def run(self):
        """运行主循环"""
        # 连接主臂
        if not self.connect_arm():
            rospy.logerr("无法连接主臂，退出")
            return
        
        # 等待一下让连接稳定
        rospy.sleep(2)
        
        # 初始化机械臂模式（默认从臂模式）
        self.initialize_arm_mode()
        
        rate = rospy.Rate(self.control_rate)  # 可配置的控制频率
        
        rospy.loginfo("🚀 主臂控制器启动成功")
        rospy.loginfo("=" * 60)
        rospy.loginfo("发布话题:")
        rospy.loginfo("  - /master/joint_states (关节状态)")
        rospy.loginfo("  - /master/arm_status (机械臂状态)")
        rospy.loginfo("  - /master/mode_status (模式状态)")
        rospy.loginfo("订阅话题:")
        rospy.loginfo("  - /master/joint_cmd (关节控制指令)")
        rospy.loginfo("  - /master/enable (使能控制)")
        rospy.loginfo("  - /master/teach_mode (示教模式切换)")
        rospy.loginfo("  - /master/master_slave_mode (主从模式切换)")
        rospy.loginfo("=" * 60)
        rospy.loginfo("📋 使用方法:")
        rospy.loginfo("  1. 使能: rostopic pub /master/enable std_msgs/Bool 'data: true'")
        rospy.loginfo("  2. 失能: rostopic pub /master/enable std_msgs/Bool 'data: false'")
        rospy.loginfo("  3. 示教: rostopic pub /master/teach_mode std_msgs/Int32 'data: 1'")
        rospy.loginfo("  4. 控制: rostopic pub /master/teach_mode std_msgs/Int32 'data: 0'")
        rospy.loginfo("  5. 关节控制: rostopic pub /master/joint_cmd sensor_msgs/JointState ...")
        rospy.loginfo("  🎯 主从模式切换（仅两种模式）:")
        rospy.loginfo("     - 主臂模式: rostopic pub /master/master_slave_mode std_msgs/String 'data: master'")
        rospy.loginfo("       （用于拖拽示教，自动进入示教模式）")
        rospy.loginfo("     - 从臂模式: rostopic pub /master/master_slave_mode std_msgs/String 'data: slave'") 
        rospy.loginfo("       （用于接收控制指令）")
        rospy.loginfo("=" * 60)
        
        while not rospy.is_shutdown():
            # 发布数据
            self.publish_joint_states()
            self.publish_arm_status()
            
            rate.sleep()

def main():
    try:
        # 设置日志级别
        rospy.loginfo("启动主臂独立控制器...")
        
        controller = MasterArmController()
        controller.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("收到中断信号，正在关闭...")
    except Exception as e:
        rospy.logerr(f"主臂控制器错误: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("主臂控制器已关闭")

if __name__ == '__main__':
    main()



# rostopic pub /master/master_slave_mode std_msgs/String 'data: master'

# rostopic pub /master/master_slave_mode std_msgs/String 'data: slave'

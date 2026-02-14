#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版：被控模式位姿控制 -> 示教模式切换测试
修复了角色配置和示教模式切换的问题
"""

import rospy
import time
from piper_sdk import C_PiperInterface
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
import math

class PiperPoseControlTest:
    def __init__(self, can_port="can_left_master"):
        """初始化测试类"""
        self.can_port = can_port
        self.piper = C_PiperInterface(self.can_port)
        
        # 连接状态
        self.connected = False
        self.enabled = False
        self.in_teach_mode = False
        self.current_mode = "unknown"  # 跟踪当前模式
        
        # 测试参数
        self.test_positions = [
            # 位置1：轻微抬起J2关节
            [0, 20000, 0, 0, 0, 0],  # J2抬高约20度
            
            # 位置2：多关节运动
            [30000, 30000, -20000, 0, 0, 0],  # J1右转30度，J2抬高30度，J3下压20度


        ]
        
        print(f"🔧 初始化Piper测试控制器 - CAN端口: {self.can_port}")
    
    def connect_and_setup(self):
        """连接并设置机械臂"""
        try:
            print("📡 正在连接机械臂...")
            self.piper.ConnectPort()
            self.connected = True
            print("✅ 机械臂连接成功")
            
            time.sleep(2)  # 等待连接稳定
            
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            self.connected = False
            return False
    
    def set_slave_mode(self):
        """配置为从臂模式（被控模式）"""
        try:
            print("🔄 配置为从臂模式（被控模式）...")
            self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)  # 设为从臂
            time.sleep(2)
            self.current_mode = "slave"
            print("✅ 从臂模式配置完成")
            return True
        except Exception as e:
            print(f"❌ 从臂模式配置失败: {e}")
            return False
    
    def set_master_mode(self):
        """配置为主臂模式（示教模式）"""
        try:
            print("🔄 配置为主臂模式（示教输入臂）...")
            self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)  # 设为主臂（示教输入臂）
            time.sleep(2)
            self.current_mode = "master"
            print("✅ 主臂模式配置完成")
            return True
        except Exception as e:
            print(f"❌ 主臂模式配置失败: {e}")
            return False
    
    def enable_arm(self):
        """使能机械臂"""
        try:
            print("🔋 使能机械臂...")
            self.piper.EnableArm(7)  # 使能所有关节
            time.sleep(3)  # 等待使能完成
            
            # 检查使能状态
            enable_status = self.piper.GetArmEnableStatus()
            self.enabled = all(enable_status)
            
            if self.enabled:
                print("✅ 机械臂使能成功")
                print(f"   使能状态: {enable_status}")
            else:
                print("⚠️ 机械臂使能可能不完全")
                print(f"   使能状态: {enable_status}")
                
            return self.enabled
            
        except Exception as e:
            print(f"❌ 使能失败: {e}")
            return False
    
    def setup_control_mode(self):
        """设置控制模式"""
        try:
            print("⚙️ 设置CAN控制模式...")
            
            # 确保退出示教模式
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
            time.sleep(1)
            
            # 设置为CAN控制模式
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,     # CAN控制模式
                move_mode=0x01,     # MOVE J (关节空间运动)
                move_spd_rate_ctrl=20  # 20%速度，慢一点更安全
            )
            time.sleep(1)
            
            self.in_teach_mode = False
            print("✅ 控制模式设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 控制模式设置失败: {e}")
            return False
    
    def check_enable_status(self):
        """检查当前使能状态"""
        try:
            enable_status = self.piper.GetArmEnableStatus()
            all_enabled = all(enable_status)
            
            print(f"🔍 当前使能状态: {enable_status}")
            print(f"   全部使能: {'✅' if all_enabled else '❌'}")
            print(f"   当前模式: {self.current_mode}")
            
            return all_enabled, enable_status
            
        except Exception as e:
            print(f"❌ 获取使能状态失败: {e}")
            return False, []
    
    def get_current_position(self):
        """获取当前关节位置"""
        try:
            joint_msgs = self.piper.GetArmJointMsgs()
            current_pos = [
                joint_msgs.joint_state.joint_1,
                joint_msgs.joint_state.joint_2,
                joint_msgs.joint_state.joint_3,
                joint_msgs.joint_state.joint_4,
                joint_msgs.joint_state.joint_5,
                joint_msgs.joint_state.joint_6
            ]
            return current_pos
        except Exception as e:
            print(f"❌ 获取当前位置失败: {e}")
            return [0, 0, 0, 0, 0, 0]
    
    def move_to_position(self, target_position, position_name="目标位置"):
        """移动到指定位置"""
        try:
            print(f"🎯 移动到{position_name}...")
            print(f"   目标位置: {target_position}")
            
            # 记录移动前状态
            before_enabled, before_status = self.check_enable_status()
            current_pos = self.get_current_position()
            print(f"   当前位置: {current_pos}")
            
            # 发送位置指令
            self.piper.JointCtrl(
                target_position[0], target_position[1], target_position[2],
                target_position[3], target_position[4], target_position[5]
            )
            
            print("⏳ 等待运动完成...")
            
            # 监控运动过程中的使能状态
            for i in range(30):  # 监控30次，每次0.5秒
                time.sleep(0.5)
                enabled, status = self.check_enable_status()
                
                if not enabled:
                    print(f"⚠️ 运动过程中检测到失能！第{i+1}次检查")
                    print(f"   使能状态变化: {before_status} -> {status}")
                    return False
                
                # 检查是否接近目标位置
                current_pos = self.get_current_position()
                max_error = max(abs(current_pos[j] - target_position[j]) for j in range(6))
                
                if max_error < 5000:  # 误差小于5度认为到达
                    print(f"✅ 已到达{position_name}")
                    print(f"   最终位置: {current_pos}")
                    print(f"   最大误差: {max_error/1000:.1f}度")
                    return True
            
            print(f"⚠️ 运动超时，可能未完全到达{position_name}")
            return False
            
        except Exception as e:
            print(f"❌ 移动到{position_name}失败: {e}")
            return False
    
    def switch_to_teach_mode(self):
        """切换到示教模式 - 修复版"""
        try:
            print("🔄 切换到示教模式...")
            
            # 记录切换前状态
            before_enabled, before_status = self.check_enable_status()
            
            # 步骤1：重新配置为主臂模式（示教输入臂）
            print("📍 重新配置为主臂模式...")
            if not self.set_master_mode():
                return False
            
            # 步骤2：使能机械臂
            print("📍 重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)
            
            # 步骤3：进入拖动示教模式
            print("📍 进入拖动示教模式...")
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x01)
            time.sleep(2)  # 等待切换完成
            
            # 检查切换后状态
            after_enabled, after_status = self.check_enable_status()
            
            self.in_teach_mode = True
            
            print("✅ 已切换到示教模式")
            print("🤏 现在应该可以手动拖拽机械臂了！")
            print("💡 请尝试轻轻移动机械臂关节")
            
            # 比较切换前后的使能状态
            if before_enabled and after_enabled:
                print("✅ 切换过程中使能状态保持正常")
            elif before_enabled and not after_enabled:
                print("⚠️ 切换到示教模式后出现失能")
                print(f"   使能状态变化: {before_status} -> {after_status}")
            else:
                print("⚠️ 切换前已存在失能问题")
            
            return True
            
        except Exception as e:
            print(f"❌ 切换到示教模式失败: {e}")
            return False
    
    def switch_to_control_mode(self):
        """切换回控制模式 - 修复版"""
        try:
            print("🔒 切换回控制模式...")
            
            # 记录切换前状态
            before_enabled, before_status = self.check_enable_status()
            
            # 步骤1：退出示教模式
            print("📍 退出拖动示教模式...")
            self.piper.MotionCtrl_1(grag_teach_ctrl=0x02)
            time.sleep(1)
            
            # 步骤2：重新配置为从臂模式
            print("📍 重新配置为从臂模式...")
            if not self.set_slave_mode():
                return False
            
            # 步骤3：重新使能机械臂
            print("📍 重新使能机械臂...")
            self.piper.EnableArm(7)
            time.sleep(2)
            
            # 步骤4：设置控制模式
            print("📍 设置CAN控制模式...")
            self.piper.MotionCtrl_2(
                ctrl_mode=0x01,
                move_mode=0x01,
                move_spd_rate_ctrl=20
            )
            time.sleep(1)
            
            # 检查切换后状态
            after_enabled, after_status = self.check_enable_status()
            
            self.in_teach_mode = False
            
            print("✅ 已切换回控制模式")
            
            # 比较切换前后的使能状态
            if before_enabled and after_enabled:
                print("✅ 切换过程中使能状态保持正常")
            elif before_enabled and not after_enabled:
                print("⚠️ 切换回控制模式后出现失能")
                print(f"   使能状态变化: {before_status} -> {after_status}")
            
            return True
            
        except Exception as e:
            print(f"❌ 切换回控制模式失败: {e}")
            return False
    
    def run_complete_test(self):
        """运行完整测试"""
        print("=" * 60)
        print("🧪 开始Piper位姿控制和示教模式切换测试")
        print("=" * 60)
        
        # 1. 连接
        if not self.connect_and_setup():
            return False
        
        # 2. 设置为从臂模式
        if not self.set_slave_mode():
            return False
        
        # 3. 使能机械臂
        if not self.enable_arm():
            return False
        
        # 4. 设置控制模式
        if not self.setup_control_mode():
            return False
        
        print("\n" + "=" * 40)
        print("🎯 开始位置控制测试")
        print("=" * 40)
        
        # 5. 依次移动到各个测试位置
        for i, position in enumerate(self.test_positions):
            print(f"\n--- 测试位置 {i+1}/{len(self.test_positions)} ---")
            
            success = self.move_to_position(position, f"位置{i+1}")
            if not success:
                print(f"⚠️ 位置{i+1}测试失败，继续下一个测试...")
            
            # 每次移动后检查状态
            time.sleep(2)
            enabled, status = self.check_enable_status()
            if not enabled:
                print("❌ 检测到失能，停止测试")
                return False
        
        print("\n" + "=" * 40)
        print("🔄 开始模式切换测试")
        print("=" * 40)
        
        # 6. 切换到示教模式
        if not self.switch_to_teach_mode():
            return False
        
        # 7. 在示教模式下等待用户测试
        print("⏳ 示教模式测试中，请手动拖拽机械臂...")
        print("💡 您有10秒时间测试手动控制")
        for i in range(10):
            time.sleep(1)
            enabled, status = self.check_enable_status()
            if not enabled:
                print("❌ 示教模式下检测到失能")
                return False
            print(f"   示教模式状态检查 {i+1}/10 ✅")
        
        # 8. 切换回控制模式
        if not self.switch_to_control_mode():
            return False
        
        # 9. 回到零位
        print("\n--- 最终回零测试 ---")
        success = self.move_to_position([0, 0, 0, 0, 0, 0], "零位")
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 测试完成！所有步骤成功")
        else:
            print("⚠️ 测试完成，但存在一些问题")
        print("=" * 60)
        
        return success
    
    def run_quick_test(self):
        """运行快速测试 - 修复版"""
        print("⚡ 快速测试：单次位置控制 + 示教模式切换")
        
        # 连接和基本设置
        if not self.connect_and_setup() or not self.set_slave_mode() or not self.enable_arm() or not self.setup_control_mode():
            return False
        
        # 简单位置测试
        print("\n🎯 移动到测试位置...")
        success = self.move_to_position([0, 30000, 0, 0, 0, 0], "测试位置")
        
        if success:
            # 切换到示教模式
            print("\n🔄 切换到示教模式...")
            self.switch_to_teach_mode()
            
            # 等待用户测试
            print("⏳ 请测试手动拖拽功能（10秒）...")
            for i in range(10):
                time.sleep(1)
                print(f"   倒计时: {10-i}秒 - 请尝试移动机械臂")
            
            # 切换回控制模式
            print("\n🔒 切换回控制模式...")
            self.switch_to_control_mode()
            
            # 回零
            print("\n🏠 回到零位...")
            self.move_to_position([0, 0, 0, 0, 0, 0], "零位")
        
        print("⚡ 快速测试完成")
        return success
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.connected:
                print("🧹 清理资源...")
                # 回到零位
                self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
                time.sleep(2)
                
                # 失能机械臂
                self.piper.DisableArm(7)
                time.sleep(1)
                
                # 断开连接
                self.piper.DisconnectPort()
                print("✅ 资源清理完成")
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}")

def main():
    # 创建测试实例
    tester = PiperPoseControlTest("can_left_master")  # 根据您的CAN端口名称调整
    
    try:
        print("选择测试模式:")
        print("1. 完整测试 (多个位置 + 模式切换)")
        print("2. 快速测试 (单个位置 + 模式切换)")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            tester.run_complete_test()
        elif choice == "2":
            tester.run_quick_test()
        else:
            print("无效选择，运行快速测试...")
            tester.run_quick_test()
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
#coding: utf-8
"""
键盘控制12自由度机械臂程序
控制方式：
- W/S：控制2号电机（前后）
- A/D：控制1号电机（左右转）
- 上/下键：控制3号电机（高低）
- Q/E：控制4号电机
- 长按按键可连续控制
- R：重置所有舵机到中位
- ESC/Ctrl+C：退出
"""

import sys
import os
import threading
from time import sleep, time
from pynput import keyboard

# 导入 Arm_Lib
from Arm_Lib import Arm_Device

class ArmController:
    def __init__(self, com_port="com4"):
        """初始化机械臂控制器"""
        try:
            self.arm = Arm_Device(com_port)
            print("[✓] 机械臂连接成功")
            self.get_version()
        except Exception as e:
            print(f"[✗] 机械臂连接失败: {e}")
            sys.exit(1)
        
        # 舵机的角度范围
        self.servo_ranges = {
            1: (0, 180),
            2: (0, 180),
            3: (0, 180),
            4: (0, 180),
            5: (0, 270),
            6: (0, 180)
        }
        # 角度步长
        self.angle_step = 5
        # 当前舵机角度缓存
        self.current_angles = {i: 90 for i in range(1, 7)}
        
        # 键盘状态跟踪
        self.keys_pressed = set()
        self.running = True
        self.last_update_time = {}
        self.update_interval = 0.05  # 50ms更新一次
        
        # 6号电机运动方向（1为增加，-1为减少）
        self.servo_6_direction = 1
        
    def get_version(self):
        """获取固件版本"""
        try:
            version = self.arm.get_version()
            if version is not None:
                print(f"[✓] 固件版本: {version}")
        except Exception as e:
            print(f"[!] 无法读取版本: {e}")
    
    def display_menu(self):
        """显示菜单"""
        print("\n" + "="*60)
        print("      12自由度机械臂 - 实时键盘控制程序")
        print("="*60)
        print("\n【控制映射】")
        print("  W / S          : 2号电机 前/后")
        print("  A / D          : 1号电机 左转/右转")
        print("  ↑ / ↓          : 3号电机 上升/下降")
        print("  Q / E          : 4号电机 减小/增大")
        print("  Space / X      : 6号电机 增加/减少")
        print("  R              : 重置所有舵机到中位")
        print("  H              : 显示帮助")
        print("  ESC / Ctrl+C   : 退出程序")
        print("\n【说明】")
        print("  - 长按按键可连续控制")
        print("  - S1-S4, S6: 0-180°")
        print("  - S5: 0-270°")
        print("  - 当前步长: {}°".format(self.angle_step))
        print("="*60 + "\n")
    
    def write_servo_angle(self, servo_id, angle, move_time=200):
        """写入舵机角度（不输出日志）"""
        try:
            min_angle, max_angle = self.servo_ranges[servo_id]
            angle = max(min_angle, min(angle, max_angle))
            
            self.arm.Arm_serial_servo_write(servo_id, angle, move_time)
            self.current_angles[servo_id] = angle
            return angle
        except Exception as e:
            pass
    
    def read_servo_angle(self, servo_id):
        """读取舵机角度"""
        try:
            angle = self.arm.Arm_serial_servo_read(servo_id)
            if angle is not None:
                return angle
            return None
        except Exception as e:
            return None
    
    def control_servo(self, servo_id, direction):
        """
        控制舵机
        direction: 1 为增加角度，-1 为减少角度
        """
        angle = self.current_angles.get(servo_id, 90) + direction * self.angle_step
        self.write_servo_angle(servo_id, angle)
    
    def show_all_angles(self):
        """显示所有舵机角度"""
        try:
            angles = []
            for i in range(1, 7):
                angle = self.read_servo_angle(i)
                if angle is not None:
                    angles.append(f"S{i}:{angle}°")
            if angles:
                print("[舵机角度] " + "  ".join(angles))
        except Exception as e:
            pass
    
    def reset_all(self):
        """重置所有舵机到中位"""
        try:
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 135, 90, 1000)
            for i in range(1, 7):
                self.current_angles[i] = 90 if i != 5 else 135
            print("[✓] 所有舵机已重置")
        except Exception as e:
            print(f"[✗] 重置失败: {e}")
    
    def on_press(self, key):
        """按键按下事件"""
        try:
            if hasattr(key, 'char') and key.char:
                self.keys_pressed.add(key.char.lower())
            else:
                self.keys_pressed.add(str(key).replace('Key.', '').lower())
        except:
            pass
    
    def on_release(self, key):
        """按键释放事件"""
        try:
            if hasattr(key, 'char') and key.char:
                self.keys_pressed.discard(key.char.lower())
            else:
                key_name = str(key).replace('Key.', '').lower()
                self.keys_pressed.discard(key_name)
                
                # ESC键退出
                if key_name == 'esc':
                    self.running = False
                    return False
        except:
            pass
    
    def update_controls(self):
        """更新控制状态"""
        current_time = time()
        
        # 1号电机 - A左转(减少) D右转(增加)
        if 'a' in self.keys_pressed:
            if current_time - self.last_update_time.get(1, 0) > self.update_interval:
                self.control_servo(1, -1)
                self.last_update_time[1] = current_time
        
        if 'd' in self.keys_pressed:
            if current_time - self.last_update_time.get(1, 0) > self.update_interval:
                self.control_servo(1, 1)
                self.last_update_time[1] = current_time
        
        # 2号电机 - W前进(增加) S后退(减少)
        if 'w' in self.keys_pressed:
            if current_time - self.last_update_time.get(2, 0) > self.update_interval:
                self.control_servo(2, 1)
                self.last_update_time[2] = current_time
        
        if 's' in self.keys_pressed:
            if current_time - self.last_update_time.get(2, 0) > self.update_interval:
                self.control_servo(2, -1)
                self.last_update_time[2] = current_time
        
        # 3号电机 - 上升(增加) 下降(减少)
        if 'up' in self.keys_pressed:
            if current_time - self.last_update_time.get(3, 0) > self.update_interval:
                self.control_servo(3, 1)
                self.last_update_time[3] = current_time
        
        if 'down' in self.keys_pressed:
            if current_time - self.last_update_time.get(3, 0) > self.update_interval:
                self.control_servo(3, -1)
                self.last_update_time[3] = current_time
        
        # 4号电机 - Q减小(减少) E增加(增加)
        if 'q' in self.keys_pressed:
            if current_time - self.last_update_time.get(4, 0) > self.update_interval:
                self.control_servo(4, -1)
                self.last_update_time[4] = current_time
        
        if 'e' in self.keys_pressed:
            if current_time - self.last_update_time.get(4, 0) > self.update_interval:
                self.control_servo(4, 1)
                self.last_update_time[4] = current_time
        
        # 6号电机 - Space增加(增加) X减少(减少)
        if 'space' in self.keys_pressed:
            if current_time - self.last_update_time.get(6, 0) > self.update_interval:
                self.control_servo(6, 1)
                self.last_update_time[6] = current_time
        
        if 'x' in self.keys_pressed:
            if current_time - self.last_update_time.get(6, 0) > self.update_interval:
                self.control_servo(6, -1)
                self.last_update_time[6] = current_time
    
    def run(self):
        """主循环"""
        self.display_menu()
        print(">>> 监听键盘输入...\n")
        
        # 启动键盘监听
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()
        
        try:
            while self.running:
                # 处理按键输入
                if 'r' in self.keys_pressed:
                    self.keys_pressed.discard('r')
                    self.reset_all()
                
                if 'h' in self.keys_pressed:
                    self.keys_pressed.discard('h')
                    self.display_menu()
                
                # 更新舵机控制
                self.update_controls()
                
                sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n>>> 程序已中断")
        finally:
            self.running = False
            listener.stop()
            print("[✓] 程序已退出")


def main():
    """主函数"""
    print("="*60)
    print("    12自由度机械臂 - 实时键盘控制程序")
    print("="*60)
    print("\n正在初始化机械臂...\n")
    
    com_port = input("输入COM端口 (默认com3): ").strip()
    if not com_port:
        com_port = "com3"
    
    controller = ArmController(com_port)
    controller.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BCI 超级控制中心 (Supercontrol)
集成小车、无人机、机械臂的多硬件控制 UI
支持键盘和脑机交互
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import pygame
except ImportError:
    print("缺少 pygame 库，请运行: pip install pygame")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from hardware import CarHttpController, TelloDroneController, SimulatedDroneController, ArmController


# ============================================================================
# 常量配置
# ============================================================================

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
FONT_LARGE = 28
FONT_MEDIUM = 20
FONT_SMALL = 14

COLOR_BG = (17, 24, 39)          # 深灰
COLOR_PANEL = (31, 41, 55)        # 面板灰
COLOR_BUTTON_IDLE = (59, 130, 246)    # 蓝色
COLOR_BUTTON_HOVER = (96, 165, 250)   # 浅蓝
COLOR_BUTTON_ACTIVE = (37, 99, 235)   # 深蓝
COLOR_BUTTON_DISABLED = (107, 114, 128)  # 灰色
COLOR_TEXT = (243, 244, 246)      # 亮灰
COLOR_SUCCESS = (34, 197, 94)     # 绿色
COLOR_ERROR = (239, 68, 68)       # 红色
COLOR_WARNING = (245, 158, 11)    # 黄色

HARDWARE_TYPES = ["小车", "无人机", "机械臂"]
CONTROL_MODES = ["键盘", "脑机"]


# ============================================================================
# 硬件管理器
# ============================================================================

class HardwareManager:
    def __init__(self):
        self.hardware_type = "小车"
        self.control_mode = "键盘"
        self.controller = None
        self.connected = False
        self.config = {
            "小车": {"host": "192.168.149.1", "port": 5000, "speed": 50},
            "无人机": {"drone_type": "tello"},
            "机械臂": {"arm_type": "dofbot"},
        }

    def set_hardware(self, hardware_type: str) -> None:
        self.hardware_type = hardware_type
        self.controller = None
        self.connected = False

    def set_mode(self, mode: str) -> None:
        self.control_mode = mode

    def connect(self) -> Tuple[bool, str]:
        try:
            if self.hardware_type == "小车":
                cfg = self.config["小车"]
                self.controller = CarHttpController(
                    host=cfg["host"],
                    port=cfg["port"],
                    speed=cfg["speed"],
                )
                # 尝试发送测试信号
                self.controller.send_signal("停止")
                self.connected = True
                return True, f"小车连接成功: {cfg['host']}:{cfg['port']}"

            elif self.hardware_type == "无人机":
                self.controller = TelloDroneController()
                self.controller.connect()
                self.connected = True
                return True, "无人机连接成功"

            elif self.hardware_type == "机械臂":
                self.controller = ArmController()
                self.controller.connect()
                self.connected = True
                return True, "机械臂连接成功"

            return False, f"不支持的硬件类型: {self.hardware_type}"

        except Exception as exc:
            self.connected = False
            return False, f"连接失败: {exc}"

    def send_command(self, command: str) -> Tuple[bool, str]:
        if not self.connected or not self.controller:
            return False, "未连接硬件"

        try:
            if self.hardware_type == "小车":
                result = self.controller.send_signal(command)
                return True, f"发送成功: {command}"

            elif self.hardware_type == "无人机":
                if hasattr(self.controller, command):
                    getattr(self.controller, command)()
                    return True, f"执行成功: {command}"
                return False, f"不支持的命令: {command}"

            elif self.hardware_type == "机械臂":
                if hasattr(self.controller, command):
                    getattr(self.controller, command)()
                    return True, f"执行成功: {command}"
                return False, f"不支持的命令: {command}"

        except Exception as exc:
            return False, f"命令执行失败: {exc}"

        return False, "未知错误"


# ============================================================================
# UI 按钮类
# ============================================================================

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
        self.pressed = False
        self.enabled = True

    def update(self, mouse_pos: Tuple[int, int]) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos) and self.enabled

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            color = COLOR_BUTTON_DISABLED
        elif self.pressed:
            color = COLOR_BUTTON_ACTIVE
        elif self.hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON_IDLE

        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, COLOR_TEXT, self.rect, 2)

        text_surf = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, mouse_pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos) and self.enabled


# ============================================================================
# Supercontrol UI 主类
# ============================================================================

class SupercontrolUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("BCI 超级控制中心 - Supercontrol")
        self.clock = pygame.time.Clock()
        self.running = True
        self.manager = HardwareManager()
        self.keyboard_enabled = False

        # 字体
        self.font_large = pygame.font.SysFont("msyh", FONT_LARGE, bold=True)
        self.font_medium = pygame.font.SysFont("msyh", FONT_MEDIUM)
        self.font_small = pygame.font.SysFont("msyh", FONT_SMALL)

        # 日志
        self.logs: list[str] = ["[系统] Supercontrol 已启动"]

        # UI 元素
        self._init_buttons()

    def _init_buttons(self) -> None:
        # 硬件选择按钮
        self.hw_buttons = {
            name: Button(80 + i * 160, 80, 140, 50, name)
            for i, name in enumerate(HARDWARE_TYPES)
        }
        self.hw_buttons[self.manager.hardware_type].pressed = True

        # 模式选择按钮
        self.mode_buttons = {
            name: Button(80 + i * 160, 160, 140, 50, name)
            for i, name in enumerate(CONTROL_MODES)
        }
        self.mode_buttons[self.manager.control_mode].pressed = True

        # 控制按钮（动态变化）
        self.control_buttons: Dict[str, Button] = {}
        self._update_control_buttons()

        # 功能按钮
        self.connect_button = Button(80, 240, 140, 50, "连接硬件")
        self.keyboard_toggle_button = Button(240, 240, 140, 50, "启用键盘")

    def _update_control_buttons(self) -> None:
        """根据硬件类型更新控制按钮"""
        self.control_buttons = {}

        if self.manager.hardware_type == "小车":
            commands = [("前进", "前进"), ("左转", "左转"), ("停止", "停止"),
                       ("右转", "右转"), ("后退", "后退")]
            positions = [
                (640, 350),  # 前进
                (500, 450),  # 左转
                (640, 450),  # 停止
                (780, 450),  # 右转
                (640, 550),  # 后退
            ]

        elif self.manager.hardware_type == "无人机":
            commands = [("起飞", "takeoff"), ("上升", "up"), ("着陆", "land"),
                       ("前进", "forward"), ("后退", "backward"),
                       ("左转", "left"), ("右转", "right")]
            positions = [
                (540, 350), (640, 350), (740, 350),
                (540, 450), (740, 450),
                (540, 550), (740, 550),
            ]

        elif self.manager.hardware_type == "机械臂":
            commands = [("连接", "connect"), ("握拳", "grip"), ("张开", "release"),
                       ("归零", "reset_pose")]
            positions = [
                (540, 350), (640, 350),
                (540, 450), (640, 450),
            ]
        else:
            commands = []
            positions = []

        for (label, cmd), pos in zip(commands, positions):
            self.control_buttons[cmd] = Button(pos[0] - 40, pos[1] - 25, 80, 50, label)

    def _log(self, message: str) -> None:
        self.logs.append(message)
        if len(self.logs) > 15:
            self.logs.pop(0)

    def _handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEMOTION:
                for btn in self.hw_buttons.values():
                    btn.update(mouse_pos)
                for btn in self.mode_buttons.values():
                    btn.update(mouse_pos)
                self.connect_button.update(mouse_pos)
                self.keyboard_toggle_button.update(mouse_pos)
                for btn in self.control_buttons.values():
                    btn.update(mouse_pos)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 硬件选择
                for hw_name, btn in self.hw_buttons.items():
                    if btn.is_clicked(mouse_pos):
                        for b in self.hw_buttons.values():
                            b.pressed = False
                        btn.pressed = True
                        self.manager.set_hardware(hw_name)
                        self._update_control_buttons()
                        self._log(f"[硬件] 已选择 {hw_name}")
                        self.manager.connected = False

                # 模式选择
                for mode_name, btn in self.mode_buttons.items():
                    if btn.is_clicked(mouse_pos):
                        for b in self.mode_buttons.values():
                            b.pressed = False
                        btn.pressed = True
                        self.manager.set_mode(mode_name)
                        self._log(f"[模式] 已选择 {mode_name}")

                # 连接硬件
                if self.connect_button.is_clicked(mouse_pos):
                    success, msg = self.manager.connect()
                    self._log(msg)
                    if success:
                        self.connect_button.text = "已连接"
                        self.connect_button.enabled = False
                    else:
                        self.connect_button.text = "重试"

                # 启用键盘
                if self.keyboard_toggle_button.is_clicked(mouse_pos):
                    self.keyboard_enabled = not self.keyboard_enabled
                    self.keyboard_toggle_button.text = "停用键盘" if self.keyboard_enabled else "启用键盘"
                    self._log(f"[键盘] {'已启用' if self.keyboard_enabled else '已停用'}")

                # 控制命令
                for cmd, btn in self.control_buttons.items():
                    if btn.is_clicked(mouse_pos):
                        success, msg = self.manager.send_command(cmd)
                        self._log(msg)

            elif event.type == pygame.KEYDOWN and self.keyboard_enabled and self.manager.connected:
                # 键盘控制映射
                key_map = {
                    pygame.K_i: "前进",
                    pygame.K_k: "后退",
                    pygame.K_j: "左转",
                    pygame.K_l: "右转",
                    pygame.K_SPACE: "停止",
                }
                if event.key in key_map:
                    cmd = key_map[event.key]
                    success, msg = self.manager.send_command(cmd)
                    self._log(msg)

    def _draw(self) -> None:
        self.screen.fill(COLOR_BG)

        # 标题
        title = self.font_large.render("BCI 超级控制中心", True, COLOR_TEXT)
        self.screen.blit(title, (20, 20))

        # 硬件选择面板
        hw_label = self.font_medium.render("硬件选择:", True, COLOR_TEXT)
        self.screen.blit(hw_label, (20, 85))
        for btn in self.hw_buttons.values():
            btn.draw(self.screen, self.font_small)

        # 模式选择面板
        mode_label = self.font_medium.render("控制方式:", True, COLOR_TEXT)
        self.screen.blit(mode_label, (20, 165))
        for btn in self.mode_buttons.values():
            btn.draw(self.screen, self.font_small)

        # 连接和键盘按钮
        self.connect_button.draw(self.screen, self.font_small)
        self.keyboard_toggle_button.draw(self.screen, self.font_small)

        # 连接状态显示
        status = "已连接" if self.manager.connected else "未连接"
        status_color = COLOR_SUCCESS if self.manager.connected else COLOR_ERROR
        status_text = self.font_small.render(f"状态: {status}", True, status_color)
        self.screen.blit(status_text, (400, 250))

        # 控制按钮
        control_label = self.font_medium.render("控制面板:", True, COLOR_TEXT)
        self.screen.blit(control_label, (20, 320))
        for btn in self.control_buttons.values():
            btn.draw(self.screen, self.font_small)

        # 日志区
        log_label = self.font_medium.render("运行日志:", True, COLOR_TEXT)
        self.screen.blit(log_label, (20, 650))

        log_y = 680
        for log in self.logs[-5:]:  # 显示最后 5 条日志
            log_color = COLOR_SUCCESS if "成功" in log else COLOR_ERROR if "失败" in log else COLOR_TEXT
            log_text = self.font_small.render(log, True, log_color)
            self.screen.blit(log_text, (20, log_y))
            log_y += 20

        # 快捷键提示
        if self.keyboard_enabled:
            hint = "键盘快捷: i前进 k后退 j左转 l右转 空格停止"
            hint_text = self.font_small.render(hint, True, COLOR_WARNING)
            self.screen.blit(hint_text, (20, WINDOW_HEIGHT - 30))

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._draw()
            self.clock.tick(60)

        pygame.quit()


# ============================================================================
# 主函数
# ============================================================================

def main() -> int:
    try:
        app = SupercontrolUI()
        app.run()
        return 0
    except Exception as exc:
        print(f"✗ 程序出错: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

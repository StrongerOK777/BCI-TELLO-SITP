"""Tello adapter for the unified brain-control session (``bin.control``).

Wraps any ``drone_hardware.DroneController`` (real Tello or the simulated one)
and owns the height bookkeeping: ``up`` on the ground takes off, ``down`` at
take-off height lands, and horizontal / yaw moves are refused until airborne.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.control import MOVE_INTENTS, DeviceBase  # noqa: E402


class DroneDevice(DeviceBase):
    name = "drone"
    capabilities = frozenset(MOVE_INTENTS)

    def __init__(self, controller: Any, *, takeoff_height: int = 100, max_height: int = 150,
                 height_step: int = 10) -> None:
        super().__init__()
        self.controller = controller
        self.takeoff_height = takeoff_height
        self.max_height = max_height
        self.height_step = height_step
        self.height = 0

    @property
    def airborne(self) -> bool:
        return self.height > 0

    def connect(self) -> None:
        self.controller.connect()

    def close(self) -> None:
        try:
            if self.airborne:
                self.controller.land()
                self.height = 0
        except Exception as exc:
            print(f"无人机降落失败：{exc}")
        finally:
            self.controller.close()

    def _refuse(self, intent: str) -> str:
        if intent == "up" and self.height >= self.max_height:
            return "已到达最大高度"
        if intent == "down" and not self.airborne:
            return "未起飞，无法下降"
        if intent not in ("up", "down") and not self.airborne:
            return "未起飞，先在升降模式起飞"
        return ""

    def _perform(self, intent: str) -> str:
        if intent == "up":
            if not self.airborne:
                self.controller.takeoff()
                self.height = self.takeoff_height
                return "起飞"
            self.controller.up()
            self.height += self.height_step
            return f"上升 {self.height_step}cm（{self.height}cm）"
        if intent == "down":
            if self.height <= self.takeoff_height:
                self.controller.land()
                self.height = 0
                return "降落"
            self.controller.down()
            self.height -= self.height_step
            return f"下降 {self.height_step}cm（{self.height}cm）"
        getattr(self.controller, intent)()
        return {"forward": "前进", "backward": "后退", "left": "左转", "right": "右转"}[intent]

    def _rest(self) -> str:
        return "悬停" if self.airborne else "停在地面"

    def _emergency(self) -> str:
        if not self.airborne:
            return "停在地面"
        self.controller.land()
        self.height = 0
        return "紧急降落"

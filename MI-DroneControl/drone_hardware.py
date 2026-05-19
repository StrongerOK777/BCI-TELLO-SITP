"""Tello-specific hardware adapters."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, List, Protocol


class DroneController(Protocol):
    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def takeoff(self) -> None:
        ...

    def land(self) -> None:
        ...

    def up(self) -> None:
        ...

    def down(self) -> None:
        ...

    def forward(self) -> None:
        ...

    def backward(self) -> None:
        ...

    def left(self) -> None:
        ...

    def right(self) -> None:
        ...


class TelloDroneController:
    def __init__(
        self,
        *,
        vertical_speed: int = 10,
        horizontal_speed: int = 10,
        yaw_speed: int = 30,
        action_sleep: float = 3.0,
        wait_for_state: bool = False,
    ) -> None:
        self.vertical_speed = vertical_speed
        self.horizontal_speed = horizontal_speed
        self.yaw_speed = yaw_speed
        self.action_sleep = action_sleep
        self.wait_for_state = wait_for_state
        self.drone: Any = None

    def connect(self) -> None:
        tello_module = importlib.import_module("djitellopy.tello")
        self.drone = tello_module.Tello()
        self.drone.connect(wait_for_state=self.wait_for_state)
        print("无人机连接成功！")
        print(f"无人机目前电量：{self.drone.get_battery()}")

    def close(self) -> None:
        pass

    def takeoff(self) -> None:
        self._require_drone().takeoff()

    def land(self) -> None:
        self._require_drone().land()

    def up(self) -> None:
        self._send_rc(0, 0, self.vertical_speed, 0)

    def down(self) -> None:
        self._send_rc(0, 0, -self.vertical_speed, 0)

    def forward(self) -> None:
        self._send_rc(0, self.horizontal_speed, 0, 0)

    def backward(self) -> None:
        self._send_rc(0, -self.horizontal_speed, 0, 0)

    def left(self) -> None:
        self._send_rc(0, 0, 0, -self.yaw_speed)

    def right(self) -> None:
        self._send_rc(0, 0, 0, self.yaw_speed)

    def _send_rc(self, lr: int, fb: int, ud: int, yv: int) -> None:
        self._require_drone().send_rc_control(lr, fb, ud, yv)
        time.sleep(self.action_sleep)

    def _require_drone(self) -> Any:
        if self.drone is None:
            raise RuntimeError("TelloDroneController.connect() must be called first")
        return self.drone


@dataclass
class SimulatedDroneController:
    action_sleep: float = 0.0
    actions: List[str] = field(default_factory=list)

    def connect(self) -> None:
        self.actions.append("connect")

    def close(self) -> None:
        self.actions.append("close")

    def takeoff(self) -> None:
        self.actions.append("takeoff")

    def land(self) -> None:
        self.actions.append("land")

    def up(self) -> None:
        self.actions.append("up")

    def down(self) -> None:
        self.actions.append("down")

    def forward(self) -> None:
        self.actions.append("forward")

    def backward(self) -> None:
        self.actions.append("backward")

    def left(self) -> None:
        self.actions.append("left")

    def right(self) -> None:
        self.actions.append("right")

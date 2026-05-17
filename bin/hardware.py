"""Hardware adapters used by BCI control state machines."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

try:
    from .eeg import add_sys_path
except ImportError:
    # 支持直接运行
    def add_sys_path(*_):
        pass


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

    def execute(self, action: str) -> None:
        getattr(self, action)()

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

    def execute(self, action: str) -> None:
        getattr(self, action)()
        if self.action_sleep > 0:
            time.sleep(self.action_sleep)


SERVO_RANGES = {
    1: (0, 180),
    2: (0, 180),
    3: (0, 180),
    4: (0, 180),
    5: (0, 270),
    6: (0, 180),
}
HOME_POSE = [90, 90, 90, 90, 135, 90]


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def clamp_servo_angle(servo_id: int, angle: int) -> int:
    min_angle, max_angle = SERVO_RANGES[servo_id]
    return clamp(int(angle), min_angle, max_angle)


class ArmController:
    def __init__(
        self,
        com_port: str = "COM4",
        *,
        arm_lib_dir: Optional[str] = None,
        dry_run: bool = False,
        angle_step: int = 5,
        move_time_ms: int = 200,
    ) -> None:
        self.com_port = com_port
        self.arm_lib_dir = arm_lib_dir
        self.dry_run = dry_run
        self.angle_step = max(1, min(int(angle_step), 10))
        self.move_time_ms = move_time_ms
        self.arm: Any = None
        self.current_angles: Dict[int, int] = {index: angle for index, angle in enumerate(HOME_POSE, start=1)}

    def connect(self) -> None:
        if self.dry_run:
            return
        add_sys_path(self.arm_lib_dir)
        module = importlib.import_module("Arm_Lib")
        self.arm = module.Arm_Device(self.com_port)

    def close(self) -> None:
        serial_obj = getattr(self.arm, "ser", None)
        if serial_obj is not None:
            serial_obj.close()

    def home(self) -> None:
        if self.dry_run:
            self._sync_pose(HOME_POSE)
            return
        self._require_arm().Arm_serial_servo_write6(*HOME_POSE, 1000)
        self._sync_pose(HOME_POSE)

    def write_servo_angle(self, servo_id: int, angle: int, move_time_ms: Optional[int] = None) -> None:
        angle = clamp_servo_angle(servo_id, angle)
        if not self.dry_run:
            self._require_arm().Arm_serial_servo_write(servo_id, angle, move_time_ms or self.move_time_ms)
        self.current_angles[servo_id] = angle

    def execute_action(self, action_name: str) -> None:
        actions = {
            "home": self.home,
            "base_left": lambda: self._step(1, -1),
            "base_right": lambda: self._step(1, 1),
            "arm_forward": lambda: self._step(2, 1),
            "arm_backward": lambda: self._step(2, -1),
            "arm_up": lambda: self._step(3, 1),
            "arm_down": lambda: self._step(3, -1),
            "joint4_decrease": lambda: self._step(4, -1),
            "joint4_increase": lambda: self._step(4, 1),
        }
        action = actions.get(action_name)
        if action is None:
            raise ValueError(f"Unknown arm action: {action_name}")
        action()

    def _step(self, servo_id: int, direction: int) -> None:
        self.write_servo_angle(servo_id, self.current_angles[servo_id] + direction * self.angle_step)

    def _sync_pose(self, pose: List[int]) -> None:
        for servo_id, angle in enumerate(pose, start=1):
            self.current_angles[servo_id] = clamp_servo_angle(servo_id, angle)

    def _require_arm(self) -> Any:
        if self.arm is None:
            raise RuntimeError("ArmController.connect() must be called first")
        return self.arm


class CarHttpController:
    def __init__(self, host: str = "192.168.149.1", port: int = 5000, speed: int = 50) -> None:
        self.host = host
        self.port = port
        self.speed = speed

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/signal"

    def send_signal(self, signal: str, speed: Optional[int] = None) -> Any:
        import requests

        response = requests.post(self.url, json={"signal": signal, "speed": speed or self.speed})
        response.raise_for_status()
        return response.json()

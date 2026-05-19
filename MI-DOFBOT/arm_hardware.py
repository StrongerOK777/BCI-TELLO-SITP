"""DOFBOT-specific arm adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        self.arm_lib_dir = arm_lib_dir or str(Path(__file__).resolve().parent / "Arm_Lib (Windows)")
        self.dry_run = dry_run
        self.angle_step = max(1, min(int(angle_step), 10))
        self.move_time_ms = move_time_ms
        self.arm: Any = None
        self.current_angles: Dict[int, int] = {index: angle for index, angle in enumerate(HOME_POSE, start=1)}

    def connect(self) -> None:
        if self.dry_run:
            return
        if self.arm_lib_dir:
            import sys
            from pathlib import Path

            path = str(Path(self.arm_lib_dir).expanduser())
            if path not in sys.path:
                sys.path.insert(0, path)
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

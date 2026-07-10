"""DOFBOT-specific arm adapter and its simulated twin.

``ArmController`` drives the real DOFBOT over serial via ``Arm_Lib``;
``SimulatedArmController`` records the same discrete action vocabulary in memory
so the control logic can be exercised in tests and ``--simulated`` runs without
any hardware. Both satisfy the :class:`ArmDevice` protocol consumed by
``brain_control.py`` and ``keyboard_control.py`` — the golden-sample "real +
simulated sharing one interface" pattern used by ``MI-DroneControl``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


SERVO_RANGES = {
    1: (0, 180),
    2: (0, 180),
    3: (0, 180),
    4: (0, 180),
    5: (0, 270),
    6: (0, 180),
}
HOME_POSE = [90, 90, 90, 90, 135, 90]

# The discrete action vocabulary every ArmDevice understands. Keeping it in one
# place lets the controller, the keyboard entrypoint and the tests agree on the
# exact command names.
ARM_ACTIONS = (
    "home",
    "safe_stop",
    "base_left",
    "base_right",
    "arm_forward",
    "arm_backward",
    "arm_up",
    "arm_down",
    "joint4_decrease",
    "joint4_increase",
    "grip_open",
    "grip_close",
)

# servo id, direction (+1 / -1) for each jog action.
_STEP_ACTIONS = {
    "base_left": (1, -1),
    "base_right": (1, 1),
    "arm_forward": (2, 1),
    "arm_backward": (2, -1),
    "arm_up": (3, 1),
    "arm_down": (3, -1),
    "joint4_decrease": (4, -1),
    "joint4_increase": (4, 1),
}


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def clamp_servo_angle(servo_id: int, angle: int) -> int:
    min_angle, max_angle = SERVO_RANGES[servo_id]
    return clamp(int(angle), min_angle, max_angle)


class ArmDevice(Protocol):
    """Interface the brain/keyboard controllers depend on."""

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def home(self) -> None:
        ...

    def safe_stop(self) -> None:
        ...

    def execute_action(self, action_name: str) -> None:
        ...


class ArmController:
    def __init__(
        self,
        com_port: str = "COM4",
        *,
        arm_lib_dir: Optional[str] = None,
        dry_run: bool = False,
        angle_step: int = 5,
        move_time_ms: int = 200,
        gripper_servo_id: int = 6,
        gripper_open_angle: int = 120,
        gripper_close_angle: int = 60,
    ) -> None:
        self.com_port = com_port
        self.arm_lib_dir = arm_lib_dir or str(Path(__file__).resolve().parent / "Arm_Lib (Windows)")
        self.dry_run = dry_run
        self.angle_step = max(1, min(int(angle_step), 10))
        self.move_time_ms = move_time_ms
        self.gripper_servo_id = gripper_servo_id
        self.gripper_open_angle = clamp_servo_angle(gripper_servo_id, gripper_open_angle)
        self.gripper_close_angle = clamp_servo_angle(gripper_servo_id, gripper_close_angle)
        self.gripper_open = False
        self.arm: Any = None
        self.current_angles: Dict[int, int] = {
            index: angle for index, angle in enumerate(HOME_POSE, start=1)
        }

    def connect(self) -> None:
        if self.dry_run:
            return
        if self.arm_lib_dir:
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

    def safe_stop(self) -> None:
        # Hold the current pose: the DOFBOT servos keep their last commanded
        # angle, so "do nothing" is the safe response to a rest decision.
        return

    def write_servo_angle(self, servo_id: int, angle: int, move_time_ms: Optional[int] = None) -> None:
        angle = clamp_servo_angle(servo_id, angle)
        if not self.dry_run:
            self._require_arm().Arm_serial_servo_write(servo_id, angle, move_time_ms or self.move_time_ms)
        self.current_angles[servo_id] = angle

    def grip_open(self) -> None:
        self.write_servo_angle(self.gripper_servo_id, self.gripper_open_angle)
        self.gripper_open = True

    def grip_close(self) -> None:
        self.write_servo_angle(self.gripper_servo_id, self.gripper_close_angle)
        self.gripper_open = False

    def execute_action(self, action_name: str) -> None:
        if action_name == "home":
            self.home()
            return
        if action_name == "safe_stop":
            self.safe_stop()
            return
        if action_name == "grip_open":
            self.grip_open()
            return
        if action_name == "grip_close":
            self.grip_close()
            return
        step = _STEP_ACTIONS.get(action_name)
        if step is None:
            raise ValueError(f"Unknown arm action: {action_name}")
        servo_id, direction = step
        self.write_servo_angle(servo_id, self.current_angles[servo_id] + direction * self.angle_step)

    def _sync_pose(self, pose: List[int]) -> None:
        for servo_id, angle in enumerate(pose, start=1):
            self.current_angles[servo_id] = clamp_servo_angle(servo_id, angle)
        self.gripper_open = self.current_angles[self.gripper_servo_id] >= self.gripper_open_angle

    def _require_arm(self) -> Any:
        if self.arm is None:
            raise RuntimeError("ArmController.connect() must be called first")
        return self.arm


@dataclass
class SimulatedArmController:
    """In-memory ArmDevice that records the exact action sequence it receives."""

    angle_step: int = 5
    actions: List[str] = field(default_factory=list)
    current_angles: Dict[int, int] = field(
        default_factory=lambda: {index: angle for index, angle in enumerate(HOME_POSE, start=1)}
    )
    gripper_open: bool = False

    def connect(self) -> None:
        self.actions.append("connect")

    def close(self) -> None:
        self.actions.append("close")

    def home(self) -> None:
        for servo_id, angle in enumerate(HOME_POSE, start=1):
            self.current_angles[servo_id] = clamp_servo_angle(servo_id, angle)
        self.gripper_open = False
        self.actions.append("home")

    def safe_stop(self) -> None:
        self.actions.append("safe_stop")

    def execute_action(self, action_name: str) -> None:
        if action_name == "home":
            self.home()
            return
        if action_name == "safe_stop":
            self.safe_stop()
            return
        if action_name in ("grip_open", "grip_close"):
            self.gripper_open = action_name == "grip_open"
            self.actions.append(action_name)
            return
        step = _STEP_ACTIONS.get(action_name)
        if step is None:
            raise ValueError(f"Unknown arm action: {action_name}")
        servo_id, direction = step
        self.current_angles[servo_id] = clamp_servo_angle(
            servo_id, self.current_angles[servo_id] + direction * self.angle_step
        )
        self.actions.append(action_name)

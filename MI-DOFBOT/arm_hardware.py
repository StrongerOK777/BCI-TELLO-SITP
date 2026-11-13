"""DOFBOT-specific arm adapter and its simulated twin.

``ArmController`` drives the real DOFBOT over serial via ``Arm_Lib``;
``SimulatedArmController`` records the same discrete action vocabulary in memory
so the control logic can be exercised in tests and ``--simulated`` runs without
any hardware. Both satisfy the :class:`ArmDevice` protocol consumed by
``brain_control.py`` and ``keyboard_control.py`` — the golden-sample "real +
simulated sharing one interface" pattern used by ``MI-DroneControl``.

Numbers that have to be measured on the physical arm (gripper angles, soft
joint limits, named poses, jog directions) live in :class:`ArmCalibration`,
loaded from ``arm_calibration.json`` so a hardware session never edits code.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


SERVO_RANGES = {
    1: (0, 180),
    2: (0, 180),
    3: (0, 180),
    4: (0, 180),
    5: (0, 270),
    6: (0, 180),
}
HOME_POSE = [90, 90, 90, 90, 135, 90]
GRIPPER_SERVO_ID = 6
READY_POSE = "ready"
POSE_PREFIX = "pose:"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "arm_calibration.json"

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
    "joint5_decrease": (5, -1),
    "joint5_increase": (5, 1),
    "grip_decrease": (6, -1),
    "grip_increase": (6, 1),
}

# The discrete action vocabulary every ArmDevice understands (plus
# ``pose:<name>`` for calibrated poses). Keeping it in one place lets the
# controller, the keyboard entrypoint and the tests agree on the command names.
ARM_ACTIONS = ("home", "safe_stop", *_STEP_ACTIONS, "grip_open", "grip_close")


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def clamp_servo_angle(servo_id: int, angle: int) -> int:
    min_angle, max_angle = SERVO_RANGES[servo_id]
    return clamp(int(angle), min_angle, max_angle)


@dataclass
class ArmCalibration:
    """Per-arm values measured on the real DOFBOT (defaults = uncalibrated).

    ``limits`` are soft joint limits inside ``SERVO_RANGES``; a jog never
    leaves them. ``poses`` maps a name to 5 angles (servos 1-5, gripper left
    alone) or 6 angles; the pose named ``ready`` is where brain control starts.
    ``steps`` overrides the servo / direction of a jog action.
    """

    gripper_open_angle: int = 120
    gripper_close_angle: int = 60
    limits: Dict[int, Tuple[int, int]] = field(default_factory=lambda: dict(SERVO_RANGES))
    poses: Dict[str, List[int]] = field(default_factory=dict)
    steps: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(_STEP_ACTIONS))

    def clamp(self, servo_id: int, angle: int) -> int:
        low, high = self.limits[servo_id]
        return clamp(int(angle), low, high)


def _servo_id(value: Any) -> int:
    servo_id = int(value)
    if servo_id not in SERVO_RANGES:
        raise ValueError(f"机械臂校准文件：舵机编号必须是 1-6，收到 {value!r}")
    return servo_id


def calibration_from_dict(data: Dict[str, Any]) -> ArmCalibration:
    """Build an :class:`ArmCalibration` from the JSON layout in the readme.

    Raises ``ValueError`` (Chinese message) on anything out of range, so a bad
    file stops the program before the arm moves.
    """

    calibration = ArmCalibration()
    gripper = data.get("gripper", {})
    for key, attr in (("open", "gripper_open_angle"), ("close", "gripper_close_angle")):
        if key in gripper:
            setattr(calibration, attr, clamp_servo_angle(GRIPPER_SERVO_ID, gripper[key]))

    for raw_id, bounds in data.get("limits", {}).items():
        servo_id = _servo_id(raw_id)
        low, high = (clamp_servo_angle(servo_id, value) for value in bounds)
        if low >= high:
            raise ValueError(f"机械臂校准文件：舵机 {servo_id} 的限位 {list(bounds)} 无效（需要 最小 < 最大）")
        calibration.limits[servo_id] = (low, high)

    for name, angles in data.get("poses", {}).items():
        if len(angles) not in (5, 6):
            raise ValueError(f"机械臂校准文件：姿态 {name} 需要 5 或 6 个角度，收到 {len(angles)} 个")
        for servo_id, angle in enumerate(angles, start=1):
            low, high = calibration.limits[servo_id]
            if not low <= int(angle) <= high:
                raise ValueError(f"机械臂校准文件：姿态 {name} 的舵机 {servo_id} 角度 {angle} 超出限位 {low}-{high}")
        calibration.poses[name] = [int(angle) for angle in angles]

    for action, (raw_id, direction) in data.get("steps", {}).items():
        if action not in _STEP_ACTIONS:
            raise ValueError(f"机械臂校准文件：未知动作 {action}")
        if int(direction) not in (1, -1):
            raise ValueError(f"机械臂校准文件：动作 {action} 的方向只能是 1 或 -1")
        calibration.steps[action] = (_servo_id(raw_id), int(direction))
    return calibration


def load_calibration(path: Optional[str] = None) -> ArmCalibration:
    """Load ``path``; no path, or a missing default file, means uncalibrated."""

    if not path:
        return ArmCalibration()
    file = Path(path).expanduser()
    if not file.exists():
        if file.resolve() == DEFAULT_CALIBRATION_PATH:
            return ArmCalibration()
        raise FileNotFoundError(f"找不到机械臂校准文件：{file}")
    calibration = calibration_from_dict(json.loads(file.read_text(encoding="utf-8")))
    print(f"已加载机械臂校准文件：{file}")
    return calibration


class ArmDevice(Protocol):
    """Interface the brain/keyboard controllers depend on."""

    calibration: ArmCalibration
    current_angles: Dict[int, int]
    gripper_open: bool

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

    def would_move(self, action_name: str) -> bool:
        ...


class _ArmBase:
    """Pose bookkeeping shared by the real and the simulated arm.

    Subclasses only say how a single servo and a full pose are written.
    """

    def __init__(
        self,
        *,
        angle_step: int = 5,
        move_time_ms: int = 200,
        calibration: Optional[ArmCalibration] = None,
        gripper_open_angle: Optional[int] = None,
        gripper_close_angle: Optional[int] = None,
    ) -> None:
        self.calibration = calibration or ArmCalibration()
        if gripper_open_angle is not None:
            self.calibration.gripper_open_angle = clamp_servo_angle(GRIPPER_SERVO_ID, gripper_open_angle)
        if gripper_close_angle is not None:
            self.calibration.gripper_close_angle = clamp_servo_angle(GRIPPER_SERVO_ID, gripper_close_angle)
        self.angle_step = max(1, min(int(angle_step), 10))
        self.move_time_ms = move_time_ms
        self.gripper_open = False
        self.current_angles: Dict[int, int] = {index: angle for index, angle in enumerate(HOME_POSE, start=1)}

    # -- hardware hooks --------------------------------------------------------
    def _write_servo(self, servo_id: int, angle: int, move_time_ms: int) -> None:
        raise NotImplementedError

    def _write_pose(self, angles: List[int], move_time_ms: int) -> None:
        raise NotImplementedError

    # -- shared behaviour ------------------------------------------------------
    def home(self) -> None:
        self._write_pose(list(HOME_POSE), 1000)
        self._sync_pose(HOME_POSE)

    def safe_stop(self) -> None:
        # Hold the current pose: the DOFBOT servos keep their last commanded
        # angle, so "do nothing" is the safe response to a rest decision.
        return

    def move_to_pose(self, name: str, move_time_ms: int = 1000) -> None:
        if name not in self.calibration.poses:
            raise ValueError(f"Unknown arm pose: {name}")
        angles = list(self.calibration.poses[name])
        if len(angles) == 5:  # 5-angle poses leave the gripper where it is
            angles.append(self.current_angles[GRIPPER_SERVO_ID])
        self._write_pose(angles, move_time_ms)
        self._sync_pose(angles)

    def write_servo_angle(self, servo_id: int, angle: int, move_time_ms: Optional[int] = None) -> None:
        angle = self.calibration.clamp(servo_id, angle)
        self._write_servo(servo_id, angle, move_time_ms or self.move_time_ms)
        self.current_angles[servo_id] = angle

    def grip_open(self) -> None:
        self.write_servo_angle(GRIPPER_SERVO_ID, self.calibration.gripper_open_angle)
        self.gripper_open = True

    def grip_close(self) -> None:
        self.write_servo_angle(GRIPPER_SERVO_ID, self.calibration.gripper_close_angle)
        self.gripper_open = False

    def would_move(self, action_name: str) -> bool:
        """False when a jog action is already parked at its soft limit."""

        step = self.calibration.steps.get(action_name)
        if step is None:
            return True
        servo_id, direction = step
        current = self.current_angles[servo_id]
        return self.calibration.clamp(servo_id, current + direction * self.angle_step) != current

    def execute_action(self, action_name: str) -> None:
        if action_name == "home":
            self.home()
        elif action_name == "safe_stop":
            self.safe_stop()
        elif action_name == "grip_open":
            self.grip_open()
        elif action_name == "grip_close":
            self.grip_close()
        elif action_name.startswith(POSE_PREFIX):
            self.move_to_pose(action_name[len(POSE_PREFIX):])
        else:
            step = self.calibration.steps.get(action_name)
            if step is None:
                raise ValueError(f"Unknown arm action: {action_name}")
            servo_id, direction = step
            self.write_servo_angle(servo_id, self.current_angles[servo_id] + direction * self.angle_step)

    def _sync_pose(self, pose: List[int]) -> None:
        for servo_id, angle in enumerate(pose, start=1):
            self.current_angles[servo_id] = clamp_servo_angle(servo_id, angle)
        self.gripper_open = self.current_angles[GRIPPER_SERVO_ID] == self.calibration.gripper_open_angle


class ArmController(_ArmBase):
    def __init__(
        self,
        com_port: str = "COM4",
        *,
        arm_lib_dir: Optional[str] = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.com_port = com_port
        self.arm_lib_dir = arm_lib_dir or str(Path(__file__).resolve().parent / "Arm_Lib (Windows)")
        self.dry_run = dry_run
        self.arm: Any = None

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

    def _write_servo(self, servo_id: int, angle: int, move_time_ms: int) -> None:
        if not self.dry_run:
            self._require_arm().Arm_serial_servo_write(servo_id, angle, move_time_ms)

    def _write_pose(self, angles: List[int], move_time_ms: int) -> None:
        if not self.dry_run:
            self._require_arm().Arm_serial_servo_write6(*angles, move_time_ms)

    def _require_arm(self) -> Any:
        if self.arm is None:
            raise RuntimeError("ArmController.connect() must be called first")
        return self.arm


class SimulatedArmController(_ArmBase):
    """In-memory ArmDevice that records the exact action sequence it receives."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.actions: List[str] = []

    def connect(self) -> None:
        self.actions.append("connect")

    def close(self) -> None:
        self.actions.append("close")

    def home(self) -> None:
        super().home()
        self.actions.append("home")

    def safe_stop(self) -> None:
        self.actions.append("safe_stop")

    def move_to_pose(self, name: str, move_time_ms: int = 1000) -> None:
        super().move_to_pose(name, move_time_ms)
        self.actions.append(POSE_PREFIX + name)

    def execute_action(self, action_name: str) -> None:
        super().execute_action(action_name)
        if action_name not in ("home", "safe_stop") and not action_name.startswith(POSE_PREFIX):
            self.actions.append(action_name)  # the others record themselves

    def _write_servo(self, servo_id: int, angle: int, move_time_ms: int) -> None:
        pass

    def _write_pose(self, angles: List[int], move_time_ms: int) -> None:
        pass

"""DOFBOT adapter for the unified brain-control session (``bin.control``).

Wraps any ``arm_hardware.ArmDevice`` (real ``ArmController`` or
``SimulatedArmController``): intents map onto the arm's jog actions, ``aux``
toggles the gripper and the emergency action returns to the home pose. A jog
that is already at its calibrated soft limit is refused, and a calibrated
``ready`` pose becomes the pose brain control starts from. The operator-only
extras (复位, 预备姿态, wrist / gripper trims) are for setup and calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.control import AUX, MOVE_INTENTS, DeviceBase  # noqa: E402

from arm_hardware import POSE_PREFIX, READY_POSE  # noqa: E402

ARM_INTENT_ACTIONS = {
    "up": "arm_up",
    "down": "arm_down",
    "forward": "arm_forward",
    "backward": "arm_backward",
    "left": "base_left",
    "right": "base_right",
}


READY_ACTION = POSE_PREFIX + READY_POSE
# operator-only manual actions (never triggered by the brain): key -> label
ARM_EXTRA_ACTIONS = {
    "home": "复位",
    READY_ACTION: "预备姿态",
    "joint4_decrease": "腕部 −",
    "joint4_increase": "腕部 +",
    "joint5_decrease": "夹爪旋转 −",
    "joint5_increase": "夹爪旋转 +",
    "grip_decrease": "夹爪角度 −",
    "grip_increase": "夹爪角度 +",
}


class ArmDeviceAdapter(DeviceBase):
    name = "arm"
    capabilities = frozenset(MOVE_INTENTS) | {AUX}

    def __init__(self, arm: Any) -> None:
        super().__init__()
        self.arm = arm
        self.has_ready_pose = READY_POSE in arm.calibration.poses
        self.extra_actions = {
            key: label for key, label in ARM_EXTRA_ACTIONS.items() if key != READY_ACTION or self.has_ready_pose
        }

    @property
    def gripper_open(self) -> bool:
        return self.arm.gripper_open

    def connect(self) -> None:
        self.arm.connect()
        if self.has_ready_pose:
            self.arm.move_to_pose(READY_POSE)
        else:
            self.arm.home()

    def close(self) -> None:
        try:
            self.arm.home()
        except Exception as exc:
            print(f"复位失败：{exc}")
        finally:
            self.arm.close()

    def status(self) -> Dict[str, Any]:
        calibration = self.arm.calibration
        return {
            "angles": {str(servo_id): angle for servo_id, angle in self.arm.current_angles.items()},
            "limits": {str(servo_id): list(bounds) for servo_id, bounds in calibration.limits.items()},
            "gripper_open": self.gripper_open,
            "gripper_angles": {"open": calibration.gripper_open_angle, "close": calibration.gripper_close_angle},
        }

    def _refuse(self, intent: str) -> str:
        if not self.arm.would_move(ARM_INTENT_ACTIONS[intent]):
            return "已到限位"
        return ""

    def _perform(self, intent: str) -> str:
        action = ARM_INTENT_ACTIONS[intent]
        self.arm.execute_action(action)
        return action

    def _rest(self) -> str:
        # DOFBOT servos hold their last commanded angle, so rest sends nothing.
        return "保持当前姿态"

    def _emergency(self) -> str:
        self.arm.home()
        return "机械臂复位"

    def _aux(self) -> str:
        self.arm.execute_action("grip_close" if self.gripper_open else "grip_open")
        return "夹爪张开" if self.gripper_open else "夹爪闭合"

    def _extra(self, key: str) -> str:
        self.arm.execute_action(key)
        return self.extra_actions[key]

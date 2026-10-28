"""DOFBOT adapter for the unified brain-control session (``bin.control``).

Wraps any ``arm_hardware.ArmDevice`` (real ``ArmController`` or
``SimulatedArmController``): intents map onto the arm's jog actions, ``aux``
toggles the gripper and the emergency action returns to the home pose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.control import AUX, MOVE_INTENTS, DeviceBase  # noqa: E402

ARM_INTENT_ACTIONS = {
    "up": "arm_up",
    "down": "arm_down",
    "forward": "arm_forward",
    "backward": "arm_backward",
    "left": "base_left",
    "right": "base_right",
}


class ArmDeviceAdapter(DeviceBase):
    name = "arm"
    capabilities = frozenset(MOVE_INTENTS) | {AUX}

    def __init__(self, arm: Any) -> None:
        super().__init__()
        self.arm = arm
        self.gripper_open = False

    def connect(self) -> None:
        self.arm.connect()
        self.arm.home()

    def close(self) -> None:
        try:
            self.arm.home()
        except Exception as exc:
            print(f"复位失败：{exc}")
        finally:
            self.arm.close()

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
        action = "grip_close" if self.gripper_open else "grip_open"
        self.arm.execute_action(action)
        self.gripper_open = not self.gripper_open
        return "夹爪张开" if self.gripper_open else "夹爪闭合"

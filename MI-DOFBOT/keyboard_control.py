"""Keyboard control entrypoint for the DOFBOT arm.

Hold a key to jog: A/D base, W/S joint 2, ↑/↓ joint 3, Q/E joint 4 (wrist),
Space/X gripper angle. Tap G to open / close the gripper at the calibrated
angles, P for the calibrated ready pose, R to home, Esc to quit. The current
servo angles are printed after every move so they can be copied into
``arm_calibration.json``.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence, Set

import sys
from pathlib import Path


DEVICE_DIR = Path(__file__).resolve().parent
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from arm_hardware import DEFAULT_CALIBRATION_PATH, READY_POSE, ArmController, load_calibration

JOG_KEYS = {
    "a": "base_left",
    "d": "base_right",
    "w": "arm_forward",
    "s": "arm_backward",
    "up": "arm_up",
    "down": "arm_down",
    "q": "joint4_decrease",
    "e": "joint4_increase",
    "space": "grip_increase",
    "x": "grip_decrease",
}


class KeyboardArmRunner:
    def __init__(self, controller: ArmController, update_interval: float = 0.05) -> None:
        self.controller = controller
        self.update_interval = update_interval
        self.keys_pressed: Set[str] = set()
        self.running = True
        self.last_update_time = {}

    def run(self) -> None:
        from pynput import keyboard

        self.controller.connect()
        self.controller.home()
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        try:
            while self.running:
                self.handle_taps()
                self.update_controls()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            listener.stop()
            self.controller.close()

    def on_press(self, key) -> None:
        name = self.key_name(key)
        if name:
            self.keys_pressed.add(name)

    def on_release(self, key):
        name = self.key_name(key)
        if name:
            self.keys_pressed.discard(name)
            if name == "esc":
                self.running = False
                return False
        return None

    def handle_taps(self) -> None:
        """One-shot keys: R home, G toggle gripper, P ready pose."""

        if "r" in self.keys_pressed:
            self.keys_pressed.discard("r")
            self.controller.home()
            self.report()
        if "g" in self.keys_pressed:
            self.keys_pressed.discard("g")
            self.controller.execute_action("grip_close" if self.controller.gripper_open else "grip_open")
            self.report()
        if "p" in self.keys_pressed:
            self.keys_pressed.discard("p")
            if READY_POSE in self.controller.calibration.poses:
                self.controller.move_to_pose(READY_POSE)
                self.report()
            else:
                print("校准文件里没有 ready 姿态")

    def update_controls(self) -> None:
        now = time.time()
        moved = False
        for key, action in JOG_KEYS.items():
            if key not in self.keys_pressed:
                continue
            if now - self.last_update_time.get(key, 0) <= self.update_interval:
                continue
            self.controller.execute_action(action)
            self.last_update_time[key] = now
            moved = True
        if moved:
            self.report()

    def report(self) -> None:
        angles = self.controller.current_angles
        print("舵机角度：" + "  ".join(f"S{servo_id}={angles[servo_id]}" for servo_id in sorted(angles)))

    @staticmethod
    def key_name(key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return str(key).replace("Key.", "").lower()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard control for the DOFBOT arm.")
    parser.add_argument("--arm-port", default="com3")
    parser.add_argument("--arm-lib-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH),
                        help="机械臂校准文件（夹爪角度 / 限位 / 预备姿态）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    controller = ArmController(
        com_port=args.arm_port,
        arm_lib_dir=args.arm_lib_dir,
        dry_run=args.dry_run,
        angle_step=args.angle_step,
        calibration=load_calibration(args.calibration),
    )
    KeyboardArmRunner(controller).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Keyboard control entrypoint for the DOFBOT arm."""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence, Set

from .hardware import ArmController


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
                if "r" in self.keys_pressed:
                    self.keys_pressed.discard("r")
                    self.controller.home()
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

    def update_controls(self) -> None:
        mapping = {
            "a": "base_left",
            "d": "base_right",
            "w": "arm_forward",
            "s": "arm_backward",
            "up": "arm_up",
            "down": "arm_down",
            "q": "joint4_decrease",
            "e": "joint4_increase",
        }
        now = time.time()
        for key, action in mapping.items():
            if key not in self.keys_pressed:
                continue
            if now - self.last_update_time.get(key, 0) <= self.update_interval:
                continue
            self.controller.execute_action(action)
            self.last_update_time[key] = now

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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    controller = ArmController(
        com_port=args.arm_port,
        arm_lib_dir=args.arm_lib_dir,
        dry_run=args.dry_run,
        angle_step=args.angle_step,
    )
    KeyboardArmRunner(controller).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

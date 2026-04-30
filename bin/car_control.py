"""Keyboard-to-HTTP car control helpers."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .hardware import CarHttpController


def key_to_signal(key_module) -> str:
    signal = "停止"
    if key_module.getKey("SPACE"):
        signal = "停止"
    if key_module.getKey("j"):
        signal = "左转"
    elif key_module.getKey("l"):
        signal = "右转"
    if key_module.getKey("i"):
        signal = "前进"
    elif key_module.getKey("k"):
        signal = "后退"
    return signal


def run_keyboard_car_control(
    *,
    host: str = "192.168.149.1",
    port: int = 5000,
    speed: int = 50,
    keypress_dir: Optional[str] = None,
    interval: float = 0.05,
) -> None:
    if keypress_dir:
        path = Path(keypress_dir).expanduser()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import KeyPressModule as kp

    controller = CarHttpController(host=host, port=port, speed=speed)
    kp.init()
    try:
        while True:
            signal = key_to_signal(kp)
            try:
                result = controller.send_signal(signal)
                print(f"✓ 发送成功: {signal} {speed}")
                print(f"  树莓派返回: {result}")
            except Exception as exc:
                print(f"✗ 发送出错: {exc}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n停止发送")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send keyboard car-control signals over HTTP.")
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--keypress-dir", default=None)
    parser.add_argument("--interval", type=float, default=0.05)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_keyboard_car_control(
        host=args.host,
        port=args.port,
        speed=args.speed,
        keypress_dir=args.keypress_dir,
        interval=args.interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Keyboard control entrypoint for the HTTP-driven car."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
for _path in (ROOT_DIR, DEVICE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bin.keyboard import KeyboardReader
from car_device import CarSignalClient  # noqa: F401  (re-exported for existing callers)


KEY_TO_SIGNAL = {
    "i": "前进",
    "k": "后退",
    "j": "左转",
    "l": "右转",
    " ": "停止",
}


def run_keyboard_control(
    client: CarSignalClient,
    *,
    reader: Optional[KeyboardReader] = None,
    interval: float = 0.05,
    dry_run: bool = False,
) -> None:
    reader = reader or KeyboardReader()
    print("🚗 键盘控制已启动：i 前进 / k 后退 / j 左转 / l 右转 / 空格 停止 / Ctrl+C 退出")
    with reader:
        while True:
            key = reader.read_key()
            signal = KEY_TO_SIGNAL.get(key or "")
            if signal is not None:
                if dry_run:
                    print(f"[dry-run] {signal}")
                else:
                    try:
                        print(f"{signal} -> {client.send_signal(signal)}")
                    except Exception as exc:
                        print(f"发送失败：{exc}", file=sys.stderr)
            time.sleep(interval)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard control for the HTTP-driven car.")
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_keyboard_control(
            CarSignalClient(host=args.host, port=args.port, speed=args.speed),
            interval=args.interval,
            dry_run=args.dry_run,
        )
        return 0
    except KeyboardInterrupt:
        print("\n已退出")
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Keyboard control entrypoint for the HTTP-driven car."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.keyboard import KeyboardReader
from bin.transport import JsonHttpClient


KEY_TO_SIGNAL = {
    "i": "前进",
    "k": "后退",
    "j": "左转",
    "l": "右转",
    " ": "停止",
}


@dataclass
class CarSignalClient:
    host: str = "192.168.149.1"
    port: int = 5000
    speed: int = 50
    transport: Optional[JsonHttpClient] = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = JsonHttpClient(f"http://{self.host}:{self.port}")

    def send_signal(self, signal: str) -> Any:
        assert self.transport is not None
        return self.transport.post_json("/signal", {"signal": signal, "speed": self.speed})


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

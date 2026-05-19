#!/usr/bin/env python3
"""Standalone MindWave signal test for the car-control workflow."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bci_interface import get_mindwave_interface


def test_brain_signal(
    *,
    port: Optional[str] = None,
    baud: Optional[int] = None,
    neuropy_dir: Optional[str] = None,
    duration: int = 10,
) -> bool:
    interface = get_mindwave_interface(port=port, baud=baud, neuropy_dir=neuropy_dir)
    driver_dir = Path(interface.neuropy_dir).expanduser()
    if str(driver_dir) not in sys.path:
        sys.path.insert(0, str(driver_dir))

    from neuropy import NeuroSkyPy

    print("测试脑环连接...")
    print(f"端口: {interface.port}, 波特率: {interface.baud}")
    print(f"测试时长: {duration} 秒\n")

    mindwave = None
    try:
        mindwave = NeuroSkyPy(interface.port, interface.baud)
        mindwave.start()
        print("✓ 脑环已连接\n")

        start_time = time.time()
        count = 0
        good_samples = 0

        print("采集中... (Ctrl+C 停止)")
        print("-" * 80)
        print(f"{'时间':<8} {'Attention':<12} {'Meditation':<12} {'PoorSignal':<12} {'BlinkStrength':<12}")
        print("-" * 80)

        while time.time() - start_time < duration:
            attention = int(getattr(mindwave, "attention", 0) or 0)
            meditation = int(getattr(mindwave, "meditation", 0) or 0)
            poor_signal = int(getattr(mindwave, "poorSignal", 0) or 0)
            blink_strength = int(getattr(mindwave, "blinkStrength", 0) or 0)

            elapsed = time.time() - start_time
            print(f"{elapsed:6.1f}s  {attention:<10}  {meditation:<10}  {poor_signal:<10}  {blink_strength:<10}")

            count += 1
            if attention > 0 and meditation > 0:
                good_samples += 1
            time.sleep(0.5)

        print("-" * 80)
        ratio = 100 * good_samples / max(1, count)
        print("\n✓ 测试完成!")
        print(f"总采样数: {count}")
        print(f"有效样本: {good_samples} ({ratio:.1f}%)")
        print("✓ 脑环工作正常！" if good_samples > 0 else "✗ 脑环未读取到有效数据")
        return good_samples > 0
    except KeyboardInterrupt:
        print("\n测试已停止")
        return True
    except Exception as exc:
        print(f"✗ 错误: {exc}", file=sys.stderr)
        return False
    finally:
        if mindwave is not None:
            try:
                mindwave.stop()
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    interface = get_mindwave_interface()
    parser = argparse.ArgumentParser(description="Test MindWave signal reception.")
    parser.add_argument("--port", default=interface.port, help="MindWave serial port")
    parser.add_argument("--baud", type=int, default=interface.baud, help="MindWave baud rate")
    parser.add_argument("--neuropy-dir", default=interface.neuropy_dir, help="Directory containing neuropy.py")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return 0 if test_brain_signal(
        port=args.port,
        baud=args.baud,
        neuropy_dir=args.neuropy_dir,
        duration=args.duration,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

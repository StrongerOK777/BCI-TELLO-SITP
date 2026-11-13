#!/usr/bin/env python3
"""Brain-signal control entrypoint for the HTTP-driven car.

The control logic lives in ``bin.control.BrainControlSession`` (shared by all
devices); this file only wires the car adapter (``car_device.CarDevice``) and
the CLI. Modes: 前后 -> 转弯 (double blink), see the root README.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from bin.cli import BrainSettings, add_brain_arguments, build_session, run_brain_signal_test, settings_from_args
from bin.control import BrainControlSession
from bci_interface import get_mindwave_interface
from car_device import CarDevice, CarSignalClient, RecordingSender


DEFAULT_MODEL_PATH = ROOT_DIR / "src" / "models" / "FinalModel.pth"
DEFAULT_INTERFACE = get_mindwave_interface(neuropy_dir=DEVICE_DIR)


@dataclass
class CarBrainConfig(BrainSettings):
    mindwave_port: Optional[str] = DEFAULT_INTERFACE.port
    mindwave_baud: Optional[int] = DEFAULT_INTERFACE.baud
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)
    neuropy_dir: Optional[str] = DEFAULT_INTERFACE.neuropy_dir
    host: str = "192.168.149.1"
    port: int = 5000
    speed: int = 50
    dry_run: bool = False


def build_device(config: CarBrainConfig, *, sender: Any = None) -> CarDevice:
    if sender is None:
        if config.dry_run:
            sender = RecordingSender()
        else:
            sender = CarSignalClient(host=config.host, port=config.port, speed=config.speed)
    return CarDevice(sender)


def build_controller(
    config: Optional[CarBrainConfig] = None,
    *,
    reader: Any = None,
    device: Any = None,
    predictor: Any = None,
    sender: Any = None,
    listeners=(),
) -> BrainControlSession:
    config = config or CarBrainConfig()
    device = device or build_device(config, sender=sender)
    return build_session(config, device, reader=reader, predictor=predictor, listeners=listeners)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain-signal control for the HTTP-driven car.")
    add_brain_arguments(parser, CarBrainConfig())
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="模拟控制命令，不连接真实小车")
    return parser


def config_from_args(args: argparse.Namespace) -> CarBrainConfig:
    return settings_from_args(CarBrainConfig, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    if args.test_brain:
        return run_brain_signal_test(config)
    try:
        if config.dry_run:
            print("小车脑控 dry-run 模式（不连接真实小车）")
        build_controller(config).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

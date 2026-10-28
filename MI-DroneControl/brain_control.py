"""Brain-signal control entrypoint for the Tello drone.

The control logic lives in ``bin.control.BrainControlSession`` (shared by all
devices); this file only wires the drone adapter (``drone_device.DroneDevice``
around ``drone_hardware``) and the CLI. Modes: 升降 -> 前后 -> 转弯 (double
blink); the drone must take off in 升降 before it accepts other moves.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from bin.cli import (
    BrainSettings,
    add_brain_arguments,
    build_predictor,
    build_reader,
    build_session,
    run_brain_signal_test,
    settings_from_args,
)
from bin.control import BrainControlSession
from bin.models import LABELS, report_model
from bci_interface import get_mindwave_interface
from drone_device import DroneDevice
from drone_hardware import SimulatedDroneController, TelloDroneController


DEFAULT_MODEL_PATH = ROOT_DIR / "src" / "models" / "FinalModel.pth"
DEFAULT_INTERFACE = get_mindwave_interface(neuropy_dir=DEVICE_DIR)


@dataclass
class MIDroneConfig(BrainSettings):
    mindwave_port: Optional[str] = DEFAULT_INTERFACE.port
    mindwave_baud: Optional[int] = DEFAULT_INTERFACE.baud
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)
    neuropy_dir: Optional[str] = DEFAULT_INTERFACE.neuropy_dir
    simulated: bool = False
    takeoff_height: int = 100
    max_height: int = 150
    height_step: int = 10


def build_device(config: MIDroneConfig, *, drone: Any = None) -> DroneDevice:
    drone = drone or (SimulatedDroneController() if config.simulated else TelloDroneController())
    return DroneDevice(
        drone,
        takeoff_height=config.takeoff_height,
        max_height=config.max_height,
        height_step=config.height_step,
    )


def build_controller(
    config: Optional[MIDroneConfig] = None,
    *,
    reader: Any = None,
    device: Any = None,
    drone: Any = None,
    predictor: Any = None,
    listeners=(),
) -> BrainControlSession:
    config = config or MIDroneConfig()
    device = device or build_device(config, drone=drone)
    return build_session(config, device, reader=reader, predictor=predictor, listeners=listeners)


def run_prediction_loop(config: MIDroneConfig) -> None:
    """Print the shared model's prediction for every window (no drone)."""

    reader = build_reader(config)
    predictor = build_predictor(config)
    if not report_model(predictor):
        return
    try:
        reader.start()
        while True:
            result = reader.collect_feature_window("预测")
            if not result.valid:
                print(f"脑环信号异常：{result.reason or result.poor_signal}，重新采集窗口...")
                time.sleep(1.0)
                continue
            predicted = predictor.predict_window(result.feature_window)
            print(f"预测结果：{LABELS[predicted]} | {config.window_size}组完成")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain-signal control for the Tello drone.")
    add_brain_arguments(parser, MIDroneConfig())
    parser.add_argument("--simulated", action="store_true", help="使用模拟无人机，不连接 Tello")
    parser.add_argument("--predict-only", action="store_true", help="只连续输出模型预测，不控制无人机")
    parser.add_argument("--takeoff-height", type=int, default=100)
    parser.add_argument("--max-height", type=int, default=150)
    parser.add_argument("--height-step", type=int, default=10)
    return parser


def config_from_args(args: argparse.Namespace) -> MIDroneConfig:
    return settings_from_args(MIDroneConfig, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    try:
        if args.test_brain:
            return run_brain_signal_test(config)
        if args.predict_only:
            run_prediction_loop(config)
        else:
            build_controller(config).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

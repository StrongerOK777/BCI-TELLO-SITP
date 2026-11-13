#!/usr/bin/env python3
"""Brain-signal control entrypoint for the DOFBOT arm.

The control logic lives in ``bin.control.BrainControlSession`` (shared by all
devices); this file only wires the arm adapter (``arm_device.ArmDeviceAdapter``
around ``arm_hardware``) and the CLI. Modes: 升降 -> 前后 -> 转弯 (double
blink); a single blink toggles the gripper (flips the rotation direction in
转弯). The arm keeps a 2-of-3 vote window by default so a single noisy window
cannot move a servo.
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
from arm_device import ArmDeviceAdapter
from arm_hardware import DEFAULT_CALIBRATION_PATH, ArmController, SimulatedArmController, load_calibration


DEFAULT_MODEL_PATH = ROOT_DIR / "src" / "models" / "FinalModel.pth"
# This folder ships no neuropy.py copy: use the shared driver from bci_interface.
DEFAULT_INTERFACE = get_mindwave_interface()


@dataclass
class ArmBrainConfig(BrainSettings):
    mindwave_port: Optional[str] = DEFAULT_INTERFACE.port
    mindwave_baud: Optional[int] = DEFAULT_INTERFACE.baud
    model_path: Optional[str] = str(DEFAULT_MODEL_PATH)
    neuropy_dir: Optional[str] = DEFAULT_INTERFACE.neuropy_dir
    attention_threshold: int = 45
    meditation_threshold: int = 55
    min_decision_count: int = 15
    vote_window: int = 3
    vote_min: int = 2
    simulated: bool = False
    dry_run: bool = False
    arm_port: str = "COM4"
    arm_lib_dir: Optional[str] = None
    angle_step: int = 5
    move_time_ms: int = 200
    # 真机测得的夹爪角度 / 限位 / 预备姿态；默认文件不存在时使用未校准的默认值
    calibration_path: Optional[str] = str(DEFAULT_CALIBRATION_PATH)
    gripper_open_angle: Optional[int] = None
    gripper_close_angle: Optional[int] = None


def build_device(config: ArmBrainConfig, *, arm: Any = None) -> ArmDeviceAdapter:
    if arm is None:
        shared = dict(
            angle_step=config.angle_step,
            move_time_ms=config.move_time_ms,
            calibration=load_calibration(config.calibration_path),
            gripper_open_angle=config.gripper_open_angle,
            gripper_close_angle=config.gripper_close_angle,
        )
        if config.simulated:
            arm = SimulatedArmController(**shared)
        else:
            arm = ArmController(
                com_port=config.arm_port, arm_lib_dir=config.arm_lib_dir, dry_run=config.dry_run, **shared
            )
    return ArmDeviceAdapter(arm)


def build_controller(
    config: Optional[ArmBrainConfig] = None,
    *,
    reader: Any = None,
    device: Any = None,
    arm: Any = None,
    predictor: Any = None,
    listeners=(),
) -> BrainControlSession:
    config = config or ArmBrainConfig()
    device = device or build_device(config, arm=arm)
    return build_session(config, device, reader=reader, predictor=predictor, listeners=listeners)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain-signal control for the DOFBOT arm.")
    add_brain_arguments(parser, ArmBrainConfig())
    parser.add_argument("--arm-port", default="COM4")
    parser.add_argument("--arm-lib-dir", default=None)
    parser.add_argument("--simulated", action="store_true", help="用内存中的仿真机械臂，不连接任何硬件库")
    parser.add_argument("--dry-run", action="store_true", help="连接真实适配器但不下发串口指令")
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--move-time-ms", type=int, default=200)
    parser.add_argument("--calibration", dest="calibration_path", default=str(DEFAULT_CALIBRATION_PATH),
                        help="机械臂校准文件（夹爪角度 / 限位 / 预备姿态），默认 MI-DOFBOT/arm_calibration.json")
    parser.add_argument("--gripper-open-angle", type=int, default=None, help="夹爪张开角度，覆盖校准文件")
    parser.add_argument("--gripper-close-angle", type=int, default=None, help="夹爪闭合角度，覆盖校准文件")
    return parser


def config_from_args(args: argparse.Namespace) -> ArmBrainConfig:
    return settings_from_args(ArmBrainConfig, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    if args.test_brain:
        return run_brain_signal_test(config)
    try:
        build_controller(config).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

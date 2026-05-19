#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect left / right / rest EEG training data for the arm rotation model."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from brain_arm_control import KNOWN_NEUROPY_DIR, add_sys_path, import_attr, safe_int


LOGGER = logging.getLogger("collect_training_data")

LABELS: Sequence[Tuple[str, int, str, str]] = (
    ("left", 0, "actionleft.txt", "请想象或执行左手/左方向动作"),
    ("right", 1, "actionright.txt", "请想象或执行右手/右方向动作"),
    ("rest", 2, "rest.txt", "请保持放松静止"),
)

EEG_FIELDS = (
    "attention",
    "meditation",
    "delta",
    "theta",
    "lowAlpha",
    "highAlpha",
    "lowBeta",
    "highBeta",
    "lowGamma",
    "midGamma",
    "poorSignal",
    "blinkStrength",
)


@dataclass
class CollectionConfig:
    mindwave_port: str
    mindwave_baud: int = 57600
    output_dir: Path = Path("data")
    rounds_per_label: int = 10
    samples_per_round: int = 30
    sample_interval: float = 0.1
    prepare_seconds: int = 3
    poor_signal_threshold: int = 20
    max_round_seconds: float = 15.0
    append: bool = False
    random_order: bool = True
    seed: int = 42
    neuropy_dir: Optional[str] = None
    log_level: str = "INFO"

    def validate(self) -> None:
        self.rounds_per_label = max(1, int(self.rounds_per_label))
        self.samples_per_round = max(1, int(self.samples_per_round))
        self.sample_interval = max(0.01, float(self.sample_interval))
        self.prepare_seconds = max(0, int(self.prepare_seconds))
        self.poor_signal_threshold = max(0, int(self.poor_signal_threshold))
        self.max_round_seconds = max(
            self.samples_per_round * self.sample_interval,
            float(self.max_round_seconds),
        )


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def prompt_required_port(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("脑环端口不能为空，请输入类似 COM6 或 /dev/ttyUSB0 的端口。")


def configure_neuropy_path(neuropy_dir: Optional[str]) -> None:
    script_dir = Path(__file__).resolve().parent
    add_sys_path(str(script_dir))
    add_sys_path(neuropy_dir)
    add_sys_path(os.getenv("NEUROPY_DIR"))
    add_sys_path(str(KNOWN_NEUROPY_DIR))


def read_eeg_values(device: Any) -> Dict[str, int]:
    return {field: safe_int(getattr(device, field, 0)) for field in EEG_FIELDS}


def is_valid_sample(values: Dict[str, int], poor_signal_threshold: int) -> bool:
    if values["poorSignal"] > poor_signal_threshold:
        return False
    if values["attention"] == 0 or values["meditation"] == 0:
        return False
    return True


def format_data_line(elapsed: float, values: Dict[str, int]) -> str:
    payload = "|".join(str(values[field]) for field in EEG_FIELDS)
    return f"{elapsed:.2f},{payload}\n"


def prepare_output_files(config: CollectionConfig) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: config.output_dir / filename for name, _label, filename, _tip in LABELS}
    if not config.append:
        for path in paths.values():
            path.write_text("", encoding="utf-8")
    return paths


def build_round_plan(config: CollectionConfig) -> List[Tuple[str, int, str, str]]:
    plan: List[Tuple[str, int, str, str]] = []
    for item in LABELS:
        plan.extend([item] * config.rounds_per_label)
    if config.random_order:
        random.Random(config.seed).shuffle(plan)
    return plan


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        LOGGER.info("准备 %d 秒后开始采集...", remaining)
        time.sleep(1)


def collect_one_round(
    device: Any,
    config: CollectionConfig,
    session_start: float,
    label_name: str,
    label_id: int,
    output_path: Path,
) -> int:
    collected = 0
    round_start = time.time()
    next_sample_time = round_start
    with output_path.open("a", encoding="utf-8") as handle:
        while collected < config.samples_per_round:
            if time.time() - round_start > config.max_round_seconds:
                LOGGER.warning(
                    "%s round timed out: collected %d/%d valid samples",
                    label_name,
                    collected,
                    config.samples_per_round,
                )
                break

            values = read_eeg_values(device)
            if is_valid_sample(values, config.poor_signal_threshold):
                elapsed = time.time() - session_start
                handle.write(format_data_line(elapsed, values))
                handle.flush()
                collected += 1
                LOGGER.info(
                    "recording %-5s label=%d sample=%02d/%02d attention=%s meditation=%s poorSignal=%s blink=%s",
                    label_name,
                    label_id,
                    collected,
                    config.samples_per_round,
                    values["attention"],
                    values["meditation"],
                    values["poorSignal"],
                    values["blinkStrength"],
                )
            else:
                LOGGER.warning(
                    "skip invalid sample: poorSignal=%s attention=%s meditation=%s",
                    values["poorSignal"],
                    values["attention"],
                    values["meditation"],
                )

            next_sample_time += config.sample_interval
            sleep_time = next_sample_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    return collected


def run_collection(config: CollectionConfig) -> None:
    configure_neuropy_path(config.neuropy_dir)
    NeuroSkyPy = import_attr("neuropy", "NeuroSkyPy", "--neuropy-dir")
    output_paths = prepare_output_files(config)
    round_plan = build_round_plan(config)

    LOGGER.info("输出目录: %s", config.output_dir.resolve())
    LOGGER.info("采集计划: %d 轮，每轮 %d 个有效样本", len(round_plan), config.samples_per_round)
    LOGGER.info("标签: 0=left, 1=right, 2=rest")

    device = NeuroSkyPy(config.mindwave_port, config.mindwave_baud)
    device.start()
    session_start = time.time()
    total_counts = {name: 0 for name, _label, _filename, _tip in LABELS}

    try:
        time.sleep(1.0)
        for index, (label_name, label_id, _filename, tip) in enumerate(round_plan, start=1):
            LOGGER.info("=" * 56)
            LOGGER.info("第 %d/%d 轮: %s (%s)", index, len(round_plan), label_name, tip)
            countdown(config.prepare_seconds)
            count = collect_one_round(
                device,
                config,
                session_start,
                label_name,
                label_id,
                output_paths[label_name],
            )
            total_counts[label_name] += count

        LOGGER.info("=" * 56)
        LOGGER.info("采集完成: %s", total_counts)
        LOGGER.info("下一步可运行: python train_eeg_model.py --data-dir %s --output model/FinalModel.pth", config.output_dir)
    except KeyboardInterrupt:
        LOGGER.warning("用户中断采集，已保留当前已写入的数据。")
    finally:
        try:
            device.stop()
        except Exception as exc:
            LOGGER.warning("停止脑环读取失败: %s", exc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect left/right/rest EEG data for the optional arm rotation model."
    )
    parser.add_argument(
        "--mindwave-port",
        default=os.getenv("MINDWAVE_PORT"),
        help="NeuroSky / MindWave serial port. If omitted, the script asks before start.",
    )
    parser.add_argument("--mindwave-baud", type=int, default=57600)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--rounds-per-label", type=int, default=10)
    parser.add_argument("--samples-per-round", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--prepare-seconds", type=int, default=3)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    parser.add_argument("--max-round-seconds", type=float, default=15.0)
    parser.add_argument("--append", action="store_true", help="Append to existing data files instead of clearing them.")
    parser.add_argument("--ordered", action="store_true", help="Collect labels in left/right/rest order instead of random order.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neuropy-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> CollectionConfig:
    return CollectionConfig(
        mindwave_port=args.mindwave_port,
        mindwave_baud=args.mindwave_baud,
        output_dir=Path(args.output_dir),
        rounds_per_label=args.rounds_per_label,
        samples_per_round=args.samples_per_round,
        sample_interval=args.sample_interval,
        prepare_seconds=args.prepare_seconds,
        poor_signal_threshold=args.poor_signal_threshold,
        max_round_seconds=args.max_round_seconds,
        append=args.append,
        random_order=not args.ordered,
        seed=args.seed,
        neuropy_dir=args.neuropy_dir,
        log_level=args.log_level,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.mindwave_port:
        args.mindwave_port = prompt_required_port("请输入脑环端口，例如 COM6: ")
    setup_logging(args.log_level)

    try:
        config = config_from_args(args)
        config.validate()
        run_collection(config)
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

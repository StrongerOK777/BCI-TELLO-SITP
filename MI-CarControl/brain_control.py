#!/usr/bin/env python3
"""Brain-signal control entrypoint for the HTTP-driven car."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from bin.eeg import BrainSignalReader, FeatureWindowResult, WindowResult
from bin.models import LABELS, ModelPredictor
from keyboard_control import CarSignalClient


MODE_FORWARD_BACKWARD = 0
MODE_TURNING = 1
MODE_NAMES = {
    MODE_FORWARD_BACKWARD: "前后",
    MODE_TURNING: "转向",
}
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "FinalModel.pth"


class SignalSender(Protocol):
    def send_signal(self, signal: str) -> object:
        ...


@dataclass
class CarBrainConfig:
    mindwave_port: Optional[str] = None
    mindwave_baud: Optional[int] = None
    model_path: str = str(DEFAULT_MODEL_PATH)
    neuropy_dir: str = str(DEVICE_DIR)
    window_size: int = 30
    sample_interval: float = 0.1
    attention_threshold: int = 30
    meditation_threshold: int = 50
    blink_threshold: int = 100
    poor_signal_threshold: int = 20
    mode_switch_blinks: int = 2
    min_decision_count: int = 20


class CarBrainController:
    def __init__(
        self,
        config: CarBrainConfig,
        reader: BrainSignalReader,
        predictor: ModelPredictor,
        sender: SignalSender,
    ) -> None:
        self.config = config
        self.reader = reader
        self.predictor = predictor
        self.sender = sender
        self.mode = MODE_FORWARD_BACKWARD
        self.running = False

    def run(self, max_windows: Optional[int] = None) -> None:
        self.running = True
        seen = 0
        try:
            self.reader.start()
            while self.running:
                self.step()
                seen += 1
                if max_windows is not None and seen >= max_windows:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.reader.stop()

    def step(self) -> None:
        mode_name = MODE_NAMES[self.mode]
        result = (
            self.reader.collect_feature_window(mode_name)
            if self.mode == MODE_TURNING
            else self.reader.collect_rule_window(mode_name)
        )
        if not result.valid:
            self.handle_invalid_window(result)
            return
        if self.mode == MODE_FORWARD_BACKWARD:
            self.handle_forward_backward_mode(result)
        else:
            self.handle_turning_mode(result)

    def handle_invalid_window(self, result: WindowResult) -> None:
        detail = result.reason or f"poorSignal={result.poor_signal}"
        print(f"脑环信号异常，仅报警不发送动作：{detail}")

    def handle_forward_backward_mode(self, result: WindowResult) -> None:
        if result.blink_count >= self.config.mode_switch_blinks:
            self.mode = MODE_TURNING
            print("双眨眼：切换到转向模式")
            return
        if (
            result.attention_count < self.config.min_decision_count
            or result.meditation_count < self.config.min_decision_count
        ):
            self._send("停止")
            return
        self._send("前进" if result.attention_count >= result.meditation_count else "后退")

    def handle_turning_mode(self, result: WindowResult) -> None:
        if result.blink_count >= self.config.mode_switch_blinks:
            self.mode = MODE_FORWARD_BACKWARD
            print("双眨眼：切换到前后模式")
            return
        if not isinstance(result, FeatureWindowResult):
            print("当前窗口缺少模型特征，仅报警不发送动作")
            return
        try:
            predicted = self.predictor.predict_window(result.feature_window)
        except Exception as exc:
            print(f"模型不可用，仅报警不发送动作：{exc}")
            return
        label = LABELS[predicted]
        print(f"模型预测：{label}")
        self._send({"left": "左转", "right": "右转", "rest": "停止"}[label])

    def _send(self, signal: str) -> None:
        try:
            print(f"{signal} -> {self.sender.send_signal(signal)}")
        except Exception as exc:
            print(f"发送失败：{exc}", file=sys.stderr)


def build_controller(
    config: Optional[CarBrainConfig] = None,
    *,
    reader: Optional[BrainSignalReader] = None,
    predictor: Optional[ModelPredictor] = None,
    sender: Optional[SignalSender] = None,
    host: str = "192.168.149.1",
    port: int = 5000,
    speed: int = 50,
) -> CarBrainController:
    config = config or CarBrainConfig()
    reader = reader or BrainSignalReader(
        port=config.mindwave_port,
        baud=config.mindwave_baud,
        neuropy_dir=config.neuropy_dir,
        window_size=config.window_size,
        sample_interval=config.sample_interval,
        attention_threshold=config.attention_threshold,
        meditation_threshold=config.meditation_threshold,
        blink_threshold=config.blink_threshold,
        poor_signal_threshold=config.poor_signal_threshold,
    )
    predictor = predictor or ModelPredictor(config.model_path, autoload=False)
    sender = sender or CarSignalClient(host=host, port=port, speed=speed)
    return CarBrainController(config, reader, predictor, sender)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain-signal control for the HTTP-driven car.")
    parser.add_argument("--mindwave-port", default=None)
    parser.add_argument("--mindwave-baud", type=int, default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--neuropy-dir", default=str(DEVICE_DIR))
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--attention-threshold", type=int, default=30)
    parser.add_argument("--meditation-threshold", type=int, default=50)
    parser.add_argument("--blink-threshold", type=int, default=100)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    return parser


def config_from_args(args: argparse.Namespace) -> CarBrainConfig:
    return CarBrainConfig(
        mindwave_port=args.mindwave_port,
        mindwave_baud=args.mindwave_baud,
        model_path=args.model_path,
        neuropy_dir=args.neuropy_dir,
        window_size=args.window_size,
        sample_interval=args.sample_interval,
        attention_threshold=args.attention_threshold,
        meditation_threshold=args.meditation_threshold,
        blink_threshold=args.blink_threshold,
        poor_signal_threshold=args.poor_signal_threshold,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        build_controller(
            config_from_args(args),
            host=args.host,
            port=args.port,
            speed=args.speed,
        ).run()
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

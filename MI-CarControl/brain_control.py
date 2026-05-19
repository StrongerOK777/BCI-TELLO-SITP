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
from bci_interface import get_mindwave_interface
from keyboard_control import CarSignalClient


MODE_FORWARD_BACKWARD = 0
MODE_TURNING = 1
MODE_NAMES = {
    MODE_FORWARD_BACKWARD: "前后",
    MODE_TURNING: "转向",
}
DEFAULT_MODEL_PATH = ROOT_DIR / "src" / "models" / "FinalModel.pth"
DEFAULT_INTERFACE = get_mindwave_interface(neuropy_dir=DEVICE_DIR)


class SignalSender(Protocol):
    def send_signal(self, signal: str) -> object:
        ...


@dataclass
class CarBrainConfig:
    mindwave_port: Optional[str] = DEFAULT_INTERFACE.port
    mindwave_baud: Optional[int] = DEFAULT_INTERFACE.baud
    model_path: str = str(DEFAULT_MODEL_PATH)
    neuropy_dir: str = DEFAULT_INTERFACE.neuropy_dir
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

        print(
            "前后模式 | "
            f"Attention: {result.attention_count}/{self.config.window_size}, "
            f"Meditation: {result.meditation_count}/{self.config.window_size}"
        )
        if (
            result.attention_count < self.config.min_decision_count
            and result.meditation_count < self.config.min_decision_count
        ):
            print(f"注意力和冥想都过低（各 < {self.config.min_decision_count}），停止")
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
    parser.add_argument("--mindwave-port", default=DEFAULT_INTERFACE.port)
    parser.add_argument("--mindwave-baud", type=int, default=DEFAULT_INTERFACE.baud)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--neuropy-dir", default=DEFAULT_INTERFACE.neuropy_dir)
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--attention-threshold", type=int, default=30)
    parser.add_argument("--meditation-threshold", type=int, default=50)
    parser.add_argument("--blink-threshold", type=int, default=100)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    parser.add_argument("--test-brain", action="store_true", help="仅读取脑环信号，不连接小车")
    parser.add_argument("--dry-run", action="store_true", help="模拟控制命令，不连接真实小车")
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
    if args.test_brain:
        return test_brain_signal_mode(args)
    if args.dry_run:
        return test_dry_run_mode(args)
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


def test_brain_signal_mode(args: argparse.Namespace) -> int:
    """只测试脑环读取，不连接小车。"""

    print("=" * 60)
    print("脑环信号测试模式（仅读取脑环数据）")
    print("=" * 60)
    print(f"端口: {args.mindwave_port}")
    print(f"波特率: {args.mindwave_baud}")
    print()

    reader = BrainSignalReader(
        port=args.mindwave_port,
        baud=args.mindwave_baud,
        neuropy_dir=args.neuropy_dir,
        window_size=args.window_size,
        sample_interval=args.sample_interval,
        attention_threshold=args.attention_threshold,
        meditation_threshold=args.meditation_threshold,
        blink_threshold=args.blink_threshold,
        poor_signal_threshold=args.poor_signal_threshold,
    )

    count = 0
    valid_count = 0
    try:
        reader.start()
        print("✓ 脑环已连接，开始采集数据...\n")
        while True:
            result = reader.collect_rule_window("测试")
            count += 1
            if result.valid:
                valid_count += 1
                print(
                    f"[{count}] ✓ 有效窗口 | "
                    f"Attention: {result.attention_count}, "
                    f"Meditation: {result.meditation_count}, "
                    f"Blinks: {result.blink_count}"
                )
            else:
                print(f"[{count}] ✗ 无效窗口 | 原因: {result.reason}")
            if result.samples:
                first = result.samples[0]
                print(
                    "      原始数据: "
                    f"A={first.attention}, M={first.meditation}, "
                    f"Signal={first.poorSignal}, Blink={first.blinkStrength}"
                )
            print()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(f"测试完成 | 总窗口: {count}, 有效: {valid_count} ({100 * valid_count / max(1, count):.1f}%)")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"✗ 错误: {exc}", file=sys.stderr)
        return 1
    finally:
        reader.stop()


def test_dry_run_mode(args: argparse.Namespace) -> int:
    """测试脑环 + 控制逻辑，但不连接真实小车。"""

    print("=" * 60)
    print("小车脑控 dry-run 模式（不连接真实小车）")
    print("=" * 60)

    class DummySender:
        def send_signal(self, signal: str) -> str:
            return f"[模拟] {signal}"

    try:
        controller = build_controller(
            config_from_args(args),
            predictor=ModelPredictor(args.model_path, autoload=False),
            sender=DummySender(),
            host=args.host,
            port=args.port,
            speed=args.speed,
        )
        controller.run()
        return 0
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

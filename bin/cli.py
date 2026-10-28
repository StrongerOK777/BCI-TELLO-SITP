"""Shared CLI / wiring helpers for every device ``brain_control.py``.

Each device config subclasses :class:`BrainSettings` (headset + decoding
knobs) and adds only its hardware fields; :func:`add_brain_arguments` and
:func:`settings_from_args` keep the command-line flags identical across
devices, and :func:`build_session` assembles the unified
:class:`bin.control.BrainControlSession`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from .control import BrainControlSession, ControlConfig
from .eeg import DEFAULT_MINDWAVE_BAUD, BrainSignalReader

S = TypeVar("S", bound="BrainSettings")


@dataclass
class BrainSettings:
    mindwave_port: Optional[str] = None
    mindwave_baud: Optional[int] = DEFAULT_MINDWAVE_BAUD
    model_path: Optional[str] = None
    neuropy_dir: Optional[str] = None
    window_size: int = 30
    sample_interval: float = 0.1
    attention_threshold: int = 30
    meditation_threshold: int = 50
    blink_threshold: int = 100
    poor_signal_threshold: int = 20
    mode_switch_blinks: int = 2
    min_decision_count: int = 20
    decision_margin: int = 3
    # 转弯模式信号来源："rule" = 专注执行、单眨眼换方向；"model" = FinalModel.pth
    turning_source: str = "rule"
    focus_threshold: int = 55
    confidence_threshold: float = 0.5
    vote_window: int = 1
    vote_min: int = 1


def add_brain_arguments(parser: argparse.ArgumentParser, defaults: BrainSettings) -> argparse.ArgumentParser:
    """Register the shared headset / decoding flags with device-specific defaults."""

    d = defaults
    parser.add_argument("--mindwave-port", default=d.mindwave_port)
    parser.add_argument("--mindwave-baud", type=int, default=d.mindwave_baud)
    parser.add_argument("--model-path", default=d.model_path)
    parser.add_argument("--neuropy-dir", default=d.neuropy_dir)
    parser.add_argument("--window-size", type=int, default=d.window_size)
    parser.add_argument("--sample-interval", type=float, default=d.sample_interval)
    parser.add_argument("--attention-threshold", type=int, default=d.attention_threshold)
    parser.add_argument("--meditation-threshold", type=int, default=d.meditation_threshold)
    parser.add_argument("--blink-threshold", type=int, default=d.blink_threshold)
    parser.add_argument("--poor-signal-threshold", type=int, default=d.poor_signal_threshold)
    parser.add_argument("--min-decision-count", type=int, default=d.min_decision_count,
                        help="升降/前后：两者都低于它则静止；转弯：需要的专注采样数")
    parser.add_argument("--decision-margin", type=int, default=d.decision_margin,
                        help="attention 与 meditation 计数至少相差此值才选边")
    parser.add_argument("--turning-source", choices=["rule", "model"], default=d.turning_source,
                        help="转弯模式信号来源：rule=专注执行、单眨眼换方向；model=FinalModel.pth")
    parser.add_argument("--focus-threshold", type=int, default=d.focus_threshold,
                        help="转弯模式中算作专注的 attention 值")
    parser.add_argument("--confidence-threshold", type=float, default=d.confidence_threshold,
                        help="model 转弯模式下 softmax 的静息带阈值")
    parser.add_argument("--vote-window", type=int, default=d.vote_window, help="多窗投票窗口长度（1 = 关闭）")
    parser.add_argument("--vote-min", type=int, default=d.vote_min, help="多窗投票所需票数")
    parser.add_argument("--test-brain", action="store_true", help="仅读取脑环信号，不连接设备")
    return parser


def settings_from_args(cls: Type[S], args: argparse.Namespace, **overrides: Any) -> S:
    """Build ``cls`` from every parsed flag whose dest matches a field name."""

    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if hasattr(args, f.name)}
    values.update(overrides)
    return cls(**values)


def control_config(settings: BrainSettings) -> ControlConfig:
    names = {f.name for f in dataclasses.fields(ControlConfig)}
    return ControlConfig(**{name: getattr(settings, name) for name in names})


def build_reader(settings: BrainSettings, **kwargs: Any) -> BrainSignalReader:
    return BrainSignalReader(
        port=settings.mindwave_port,
        baud=settings.mindwave_baud,
        neuropy_dir=settings.neuropy_dir,
        window_size=settings.window_size,
        sample_interval=settings.sample_interval,
        attention_threshold=settings.attention_threshold,
        meditation_threshold=settings.meditation_threshold,
        blink_threshold=settings.blink_threshold,
        poor_signal_threshold=settings.poor_signal_threshold,
        **kwargs,
    )


def build_predictor(settings: BrainSettings) -> Any:
    from .models import ModelPredictor  # torch is only needed for the model source

    return ModelPredictor(settings.model_path, autoload=False)


def build_session(settings: BrainSettings, device: Any, *, reader: Any = None, predictor: Any = None,
                  listeners=()) -> BrainControlSession:
    if predictor is None and settings.turning_source == "model":
        predictor = build_predictor(settings)
    return BrainControlSession(
        control_config(settings),
        reader if reader is not None else build_reader(settings),
        device,
        predictor,
        listeners=listeners,
    )


def run_brain_signal_test(settings: BrainSettings, *, reader: Any = None) -> int:
    """Read the headset only (no device): window quality + focus distribution."""

    print("=" * 60)
    print("脑环信号测试模式（仅读取脑环数据，不连接设备）")
    print("=" * 60)
    print(f"端口: {settings.mindwave_port}，波特率: {settings.mindwave_baud}\n")
    reader = reader if reader is not None else build_reader(settings)
    count = valid_count = 0
    try:
        reader.start()
        print("✓ 脑环已连接，开始采集数据...\n")
        while True:
            result = reader.collect_rule_window("测试")
            count += 1
            if result.valid:
                valid_count += 1
                attention = sorted(s.attention for s in result.samples)
                focused = sum(1 for value in attention if value >= settings.focus_threshold)
                print(
                    f"[{count}] ✓ 有效窗口 | Attention: {result.attention_count}, "
                    f"Meditation: {result.meditation_count}, Blinks: {result.blink_count}\n"
                    f"      专注采样 ≥ {settings.focus_threshold}: {focused}/{len(attention)}，"
                    f"attention 中位数: {attention[len(attention) // 2] if attention else 0}"
                )
            else:
                print(f"[{count}] ✗ 无效窗口 | 原因: {result.reason}")
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

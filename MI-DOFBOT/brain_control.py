#!/usr/bin/env python3
"""Motor-imagery BCI control state machine for the DOFBOT arm.

This is the arm counterpart of ``MI-DroneControl/brain_control.py``: it reuses
the shared core (``bin.eeg`` for windowed acquisition, ``bin.models`` for the
one trained model, ``bin.decoding`` for confidence gating + temporal
smoothing) and keeps only the arm-specific wiring here. Hardware is isolated
behind ``arm_hardware.ArmDevice`` so tests inject a recording
``SimulatedArmController``.

Decoding pipeline (EEG window -> arm command):
  1. blink events are discrete: 2 blinks cycle the mode, 1 blink toggles grip.
  2. otherwise a per-mode decoder emits a *raw* intent:
       - 升降 / 前后 modes: attention vs meditation with a dead-band -> rest
       - 转弯 mode: the shared model, gated by softmax confidence -> rest band
  3. the raw intent passes through a majority-vote window (hysteresis) so a
     single noisy window cannot move the servos; only a committed intent acts.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEVICE_DIR) not in sys.path:
    sys.path.insert(0, str(DEVICE_DIR))

from bin.decoding import VoteWindow, gate_by_confidence
from bin.eeg import BrainSignalReader, FeatureWindowResult, WindowResult
from bin.models import LABELS, ModelPredictor
from bci_interface import get_mindwave_interface
from arm_hardware import ArmController, ArmDevice, SimulatedArmController


DEFAULT_MODEL_PATH = ROOT_DIR / "src" / "models" / "FinalModel.pth"
DEFAULT_INTERFACE = get_mindwave_interface(neuropy_dir=DEVICE_DIR)

IDLE_INTENT = "rest"

MODE_VERTICAL = 0
MODE_ROTATION = 1
MODE_FORWARD_BACKWARD = 2
MODE_NAMES = {
    MODE_VERTICAL: "升降",
    MODE_ROTATION: "转弯",
    MODE_FORWARD_BACKWARD: "前后",
}
MODE_COUNT = len(MODE_NAMES)

# Maps the model's class labels to the arm intents used in 转弯 (rotation) mode.
ROTATION_INTENTS = {"left": "base_left", "right": "base_right", "rest": IDLE_INTENT}


@dataclass
class ArmBrainConfig:
    mindwave_port: Optional[str] = DEFAULT_INTERFACE.port
    mindwave_baud: Optional[int] = DEFAULT_INTERFACE.baud
    model_path: str = str(DEFAULT_MODEL_PATH)
    neuropy_dir: str = DEFAULT_INTERFACE.neuropy_dir
    simulated: bool = False
    dry_run: bool = False
    arm_port: str = "COM4"
    arm_lib_dir: Optional[str] = None
    angle_step: int = 5
    move_time_ms: int = 200
    window_size: int = 30
    sample_interval: float = 0.1
    attention_threshold: int = 45
    meditation_threshold: int = 55
    blink_threshold: int = 100
    poor_signal_threshold: int = 20
    mode_switch_blinks: int = 2
    # dead-band: both counts below this -> deliberate rest (do nothing).
    min_decision_count: int = 15
    # attention/meditation counts must differ by at least this to pick a side.
    decision_margin: int = 3
    # softmax gate for the model in 转弯 mode.
    confidence_threshold: float = 0.5
    # majority-vote hysteresis: commit an intent only after it wins vote_min of
    # the last vote_window windows. Set both to 1 to disable smoothing.
    vote_window: int = 3
    vote_min: int = 2
    action_pause_sec: float = 0.2


class ArmBrainController:
    def __init__(
        self,
        config: ArmBrainConfig,
        reader: BrainSignalReader,
        arm: ArmDevice,
        predictor: ModelPredictor,
    ) -> None:
        self.config = config
        self.reader = reader
        self.arm = arm
        self.predictor = predictor
        self.mode = MODE_VERTICAL
        self.gripper_open = False
        self.running = False
        self._vote = VoteWindow(
            size=config.vote_window,
            min_votes=config.vote_min,
            idle_intent=IDLE_INTENT,
        )

    def run(self, max_windows: Optional[int] = None) -> None:
        self.running = True
        seen = 0
        try:
            self.arm.connect()
            self.arm.home()
            self.reader.start()
            while self.running:
                self.step()
                seen += 1
                if max_windows is not None and seen >= max_windows:
                    break
                time.sleep(self.config.action_pause_sec)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def step(self) -> None:
        mode_name = MODE_NAMES[self.mode]
        print(f"{mode_name}：")
        if self.mode == MODE_ROTATION:
            result = self.reader.collect_feature_window(mode_name)
        else:
            result = self.reader.collect_rule_window(mode_name)

        if not result.valid:
            self.handle_invalid_window(result)
            return

        if result.blink_count >= self.config.mode_switch_blinks:
            self.switch_mode()
            return
        if result.blink_count == 1:
            self.toggle_gripper()
            return

        raw_intent = self.decode_intent(result)
        committed = self._vote.push(raw_intent)
        if committed == IDLE_INTENT:
            print(f"保持静止（原始意图 {raw_intent}）")
            return
        print(f"执行动作：{committed}")
        self.arm.execute_action(committed)

    def switch_mode(self) -> None:
        old_mode = self.mode
        self.mode = (self.mode + 1) % MODE_COUNT
        self._vote.reset()
        print(f"双眨眼：{MODE_NAMES[old_mode]} → {MODE_NAMES[self.mode]}")

    def toggle_gripper(self) -> None:
        if self.gripper_open:
            self.arm.execute_action("grip_close")
            self.gripper_open = False
            print("单眨眼：夹爪闭合")
        else:
            self.arm.execute_action("grip_open")
            self.gripper_open = True
            print("单眨眼：夹爪张开")

    def decode_intent(self, result: WindowResult) -> str:
        if self.mode == MODE_VERTICAL:
            return self.decode_rule(result, "arm_up", "arm_down")
        if self.mode == MODE_FORWARD_BACKWARD:
            return self.decode_rule(result, "arm_forward", "arm_backward")
        return self.decode_rotation(result)

    def decode_rule(self, result: WindowResult, attention_intent: str, meditation_intent: str) -> str:
        if (
            result.attention_count < self.config.min_decision_count
            and result.meditation_count < self.config.min_decision_count
        ):
            return IDLE_INTENT
        if result.attention_count >= result.meditation_count + self.config.decision_margin:
            return attention_intent
        if result.meditation_count >= result.attention_count + self.config.decision_margin:
            return meditation_intent
        return IDLE_INTENT

    def decode_rotation(self, result: WindowResult) -> str:
        if not isinstance(result, FeatureWindowResult) or not result.feature_window:
            return IDLE_INTENT
        try:
            probabilities = self.predictor.predict_proba(result.feature_window)
        except Exception as exc:
            print(f"模型不可用，保持静止：{exc}")
            return IDLE_INTENT
        decision = gate_by_confidence(
            probabilities,
            LABELS,
            self.config.confidence_threshold,
            reject_intent=IDLE_INTENT,
        )
        print(f"模型预测：{decision.intent}（{decision.reason}）")
        return ROTATION_INTENTS.get(decision.intent, IDLE_INTENT)

    def handle_invalid_window(self, result: WindowResult) -> None:
        detail = result.reason or f"poorSignal={result.poor_signal}"
        print(f"脑环信号异常，仅报警不发送动作：{detail}")
        if result.disconnected:
            print("poorSignal=200，机械臂复位并等待信号恢复")
            self._vote.reset()
            self.mode = MODE_VERTICAL
            try:
                self.arm.home()
            except Exception as exc:
                print(f"复位失败：{exc}")
        time.sleep(self.config.action_pause_sec)

    def shutdown(self) -> None:
        self.running = False
        try:
            self.reader.stop()
        finally:
            try:
                self.arm.home()
            except Exception as exc:
                print(f"复位失败：{exc}")
            self.arm.close()
        print("\n" + "=" * 60)
        print("所有进程已退出")
        print("=" * 60)


def build_controller(
    config: Optional[ArmBrainConfig] = None,
    *,
    reader: Optional[BrainSignalReader] = None,
    arm: Optional[ArmDevice] = None,
    predictor: Optional[ModelPredictor] = None,
) -> ArmBrainController:
    config = config or ArmBrainConfig()
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
    if arm is None:
        if config.simulated:
            arm = SimulatedArmController(angle_step=config.angle_step)
        else:
            arm = ArmController(
                com_port=config.arm_port,
                arm_lib_dir=config.arm_lib_dir,
                dry_run=config.dry_run,
                angle_step=config.angle_step,
                move_time_ms=config.move_time_ms,
            )
    predictor = predictor or ModelPredictor(config.model_path, autoload=False)
    return ArmBrainController(config, reader, arm, predictor)


def run_brain_signal_test(config: ArmBrainConfig) -> int:
    """Read the headset only (no arm), reporting window quality. Like the car's --test-brain."""

    print("=" * 60)
    print("脑环信号测试模式（仅读取脑环数据，不连接机械臂）")
    print("=" * 60)
    reader = BrainSignalReader(
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control the DOFBOT arm with MindWave motor-imagery signals.")
    parser.add_argument("--mindwave-port", default=DEFAULT_INTERFACE.port)
    parser.add_argument("--mindwave-baud", type=int, default=DEFAULT_INTERFACE.baud)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--neuropy-dir", default=DEFAULT_INTERFACE.neuropy_dir)
    parser.add_argument("--arm-port", default="COM4")
    parser.add_argument("--arm-lib-dir", default=None)
    parser.add_argument("--simulated", action="store_true", help="用内存中的仿真机械臂，不连接任何硬件库")
    parser.add_argument("--dry-run", action="store_true", help="连接真实适配器但不下发串口指令")
    parser.add_argument("--test-brain", action="store_true", help="仅读取脑环信号，不连接机械臂")
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--move-time-ms", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--attention-threshold", type=int, default=45)
    parser.add_argument("--meditation-threshold", type=int, default=55)
    parser.add_argument("--blink-threshold", type=int, default=100)
    parser.add_argument("--poor-signal-threshold", type=int, default=20)
    parser.add_argument("--decision-margin", type=int, default=3)
    parser.add_argument("--min-decision-count", type=int, default=15)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--vote-window", type=int, default=3)
    parser.add_argument("--vote-min", type=int, default=2)
    return parser


def config_from_args(args: argparse.Namespace) -> ArmBrainConfig:
    return ArmBrainConfig(
        mindwave_port=args.mindwave_port,
        mindwave_baud=args.mindwave_baud,
        model_path=args.model_path,
        neuropy_dir=args.neuropy_dir,
        simulated=args.simulated,
        dry_run=args.dry_run,
        arm_port=args.arm_port,
        arm_lib_dir=args.arm_lib_dir,
        angle_step=args.angle_step,
        move_time_ms=args.move_time_ms,
        window_size=args.window_size,
        sample_interval=args.sample_interval,
        attention_threshold=args.attention_threshold,
        meditation_threshold=args.meditation_threshold,
        blink_threshold=args.blink_threshold,
        poor_signal_threshold=args.poor_signal_threshold,
        decision_margin=args.decision_margin,
        min_decision_count=args.min_decision_count,
        confidence_threshold=args.confidence_threshold,
        vote_window=args.vote_window,
        vote_min=args.vote_min,
    )


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

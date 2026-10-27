"""Cued online test of the 转弯/turning pipeline: 左转 / 右转 / 静止.

The screen shows one cue at a time; the operator produces it with the headset
(rule mode: single blink to pick the direction, then focus). Each cue is timed
until the controller emits its first movement, and a Markdown/CSV/JSON report
is written for the defense.

What is measured is the full brain -> command pipeline of the chosen device
(the real ``brain_control.py`` controller, its FocusSteer / vote window /
model), with **simulated hardware** that only records commands - nothing
physical moves during the test.

Usage (from repo root):
    python experiments/online_cue_test/cue_test.py --device drone
    python experiments/online_cue_test/cue_test.py --device arm --trials-per-class 10 --focus-threshold 60
    python experiments/online_cue_test/cue_test.py --device car --mock-headset --no-gui   # 无脑环试跑

Any option the device's own brain_control.py accepts (``--focus-threshold``,
``--turning-source``, ``--mindwave-port``, ...) is passed through.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bin.eeg import EEGSnapshot, WindowResult  # noqa: E402
from bin.models import report_model  # noqa: E402
from bin.evaluation import (  # noqa: E402
    CUE_NAMES,
    OUTCOME_NAMES,
    REST_CUE,
    build_cue_schedule,
    format_summary_markdown,
    results_to_rows,
    run_cue_trial,
    summarize,
)

DEVICES = {
    # name: (brain_control.py path, turning-mode constant, hardware attribute, command -> direction)
    "car": ("MI-CarControl/brain_control.py", "MODE_TURNING", "sender", {"左转": "left", "右转": "right"}),
    "drone": ("MI-DroneControl/brain_control.py", "MODE_TURNING", "drone", {"left": "left", "right": "right"}),
    "arm": ("MI-DOFBOT/brain_control.py", "MODE_ROTATION", "arm", {"base_left": "left", "base_right": "right"}),
}
DIRECTION_NAMES = {"left": "左", "right": "右"}


def load_device_module(device: str):
    path = ROOT_DIR / DEVICES[device][0]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"{device}_brain_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


class RecordingSender:
    """Car stand-in for CarSignalClient: records signals, sends nothing."""

    def __init__(self) -> None:
        self.actions: List[str] = []

    def send_signal(self, signal: str) -> str:
        self.actions.append(signal)
        return f"[口令测试] {signal}"


class WindowTracker:
    """Reader proxy that remembers whether the last window was valid."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self.last_valid = True

    def _track(self, result):
        self.last_valid = bool(getattr(result, "valid", True))
        return result

    def collect_rule_window(self, mode_name):
        return self._track(self.reader.collect_rule_window(mode_name))

    def collect_feature_window(self, mode_name):
        return self._track(self.reader.collect_feature_window(mode_name))

    def start(self):
        self.reader.start()

    def stop(self):
        self.reader.stop()


class VirtualClock:
    """Advances by one EEG window per mock step so timeouts behave like real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class MockHeadsetReader:
    """Scripted headset for trying the test without hardware.

    With probability ``accuracy`` a window does the "right thing" for the cue
    (blink once if the steering points the wrong way, then focus; stay relaxed
    on 静止), otherwise it produces a random window.
    """

    def __init__(self, cue: Callable[[], str], direction: Callable[[], str], *, accuracy: float,
                 window_size: int, window_sec: float, clock: VirtualClock, seed: int) -> None:
        self.cue, self.direction = cue, direction
        self.accuracy = accuracy
        self.window_size = window_size
        self.window_sec = window_sec
        self.clock = clock
        self.rng = random.Random(seed)

    def _window(self, mode_name: str, attention: int, blink_count: int = 0) -> WindowResult:
        self.clock.now += self.window_sec
        samples = [EEGSnapshot(attention=attention, meditation=50) for _ in range(self.window_size)]
        return WindowResult(mode_name, blink_count=blink_count, samples=samples)

    def collect_rule_window(self, mode_name):
        cue = self.cue()
        if self.rng.random() >= self.accuracy:
            return self._window(mode_name, self.rng.choice([40, 70]), self.rng.choice([0, 0, 1]))
        if cue == REST_CUE:
            return self._window(mode_name, 40)
        if self.direction() != cue:
            return self._window(mode_name, 40, blink_count=1)
        return self._window(mode_name, 70)

    collect_feature_window = collect_rule_window

    def start(self):
        pass

    def stop(self):
        pass


class TerminalDisplay:
    def show(self, title: str, subtitle: str = "", footer: str = "", cue: Optional[str] = None) -> None:
        print(f"\n>>> {title}  {subtitle}  {footer}".rstrip())

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        pass


class PygameDisplay:
    """Large cue window: arrow for 左转/右转, ring for 静止, Chinese text."""

    FONT_CANDIDATES = "pingfangsc,hiraginosansgb,stheiti,microsoftyahei,simhei,notosanscjksc,arialunicodems"

    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.size = (720, 460)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("脑控口令测试")
        self.big = pygame.font.SysFont(self.FONT_CANDIDATES, 64)
        self.small = pygame.font.SysFont(self.FONT_CANDIDATES, 26)

    def _pump(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT or (
                event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt

    def show(self, title: str, subtitle: str = "", footer: str = "", cue: Optional[str] = None) -> None:
        pg = self.pygame
        w, h = self.size
        cx, cy = w // 2, h // 2 - 30
        self.screen.fill((250, 250, 248))
        color = (42, 120, 214)
        if cue == "left":
            pg.draw.polygon(self.screen, color, [(cx - 270, cy), (cx - 160, cy - 70), (cx - 160, cy + 70)])
        elif cue == "right":
            pg.draw.polygon(self.screen, color, [(cx + 270, cy), (cx + 160, cy - 70), (cx + 160, cy + 70)])
        elif cue == REST_CUE:
            pg.draw.circle(self.screen, (138, 137, 132), (cx, cy), 95, 8)
        for font, text, y, ink in (
            (self.big, title, cy, (11, 11, 11)),
            (self.small, subtitle, cy + 120, (82, 81, 78)),
            (self.small, footer, h - 30, (138, 137, 132)),
        ):
            if text:
                surface = font.render(text, True, ink)
                self.screen.blit(surface, surface.get_rect(center=(cx, y)))
        pg.display.flip()
        self._pump()

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._pump()
            time.sleep(0.05)

    def close(self) -> None:
        self.pygame.quit()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="口令测试：依次提示左转/右转/静止，记录命中率与耗时")
    parser.add_argument("--device", choices=sorted(DEVICES), required=True)
    parser.add_argument("--trials-per-class", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=15.0, help="左转/右转口令的最长等待秒数")
    parser.add_argument("--rest-hold", type=float, default=9.0, help="静止口令需要保持不动的秒数")
    parser.add_argument("--gap-sec", type=float, default=3.0, help="两条口令之间的准备时间")
    parser.add_argument("--seed", type=int, default=None, help="口令顺序随机种子（默认按时间）")
    parser.add_argument("--subject", default="anon", help="被试代号，写入结果文件名（不要用真实姓名）")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--no-gui", action="store_true", help="不弹出 pygame 窗口，只在终端提示")
    parser.add_argument("--mock-headset", action="store_true", help="用脚本模拟脑环信号试跑流程（结果不代表真实性能）")
    parser.add_argument("--mock-accuracy", type=float, default=0.85)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args, device_argv = build_arg_parser().parse_known_args(argv)
    module = load_device_module(args.device)
    _, mode_attr, hardware_attr, command_map = DEVICES[args.device]
    config = module.config_from_args(module.build_arg_parser().parse_args(device_argv))
    if hasattr(config, "simulated"):
        config = dataclasses.replace(config, simulated=True)

    clock: Callable[[], float] = time.monotonic
    current_cue = {"cue": REST_CUE}
    builder_kwargs = {"sender": RecordingSender()} if args.device == "car" else {}
    if args.mock_headset:
        virtual = VirtualClock()
        clock = virtual
        controller_ref = {}
        builder_kwargs["reader"] = MockHeadsetReader(
            lambda: current_cue["cue"],
            lambda: controller_ref["c"].steer.direction,
            accuracy=args.mock_accuracy,
            window_size=config.window_size,
            window_sec=config.window_size * config.sample_interval,
            clock=virtual,
            seed=args.seed or 0,
        )
    controller = module.build_controller(config, **builder_kwargs)
    if args.mock_headset:
        controller_ref["c"] = controller
    turning_mode = getattr(module, mode_attr)
    controller.reader = WindowTracker(controller.reader)
    hardware = getattr(controller, hardware_attr)
    rule_mode = getattr(config, "turning_source", "rule") != "model"
    if hasattr(config, "action_pause_sec"):
        config.action_pause_sec = 0.0  # no extra sleeps between windows during timing

    cursor = {"n": len(hardware.actions)}

    def poll_move() -> Optional[str]:
        new = hardware.actions[cursor["n"]:]
        cursor["n"] = len(hardware.actions)
        for action in new:
            if action in command_map:
                return command_map[action]
        return None

    def step() -> bool:
        controller.mode = turning_mode
        controller.step()
        if controller.mode != turning_mode:
            print("检测到双眨眼切换模式，口令测试中已自动切回转弯模式")
        return controller.reader.last_valid

    seed = args.seed if args.seed is not None else int(time.time())
    schedule = build_cue_schedule(args.trials_per_class, seed=seed)
    display = TerminalDisplay() if args.no_gui or args.mock_headset else PygameDisplay()
    wait = (lambda s: None) if args.mock_headset else display.wait
    results = []
    print(f"口令测试：设备 {args.device}，共 {len(schedule)} 条口令，转弯信号来源 "
          f"{'规则（专注执行、单眨眼换方向）' if rule_mode else '模型'}")
    try:
        if not rule_mode:
            report_model(controller.predictor)
        controller.reader.start()
        if hasattr(hardware, "connect"):
            hardware.connect()
        display.show("准备开始", "按 ESC 可随时退出", f"共 {len(schedule)} 条口令")
        wait(args.gap_sec)
        for index, cue in enumerate(schedule, start=1):
            current_cue["cue"] = REST_CUE
            display.show("放松，准备下一条", "", f"{index}/{len(schedule)}")
            wait(args.gap_sec)
            current_cue["cue"] = cue
            vote = getattr(controller, "_vote", None)  # arm: the gap was spent relaxed -> drop stale votes
            if vote is not None:
                vote.reset()
            direction = controller.steer.direction if rule_mode else ""
            hint = ""
            if rule_mode and cue != REST_CUE:
                hint = f"当前方向：{DIRECTION_NAMES[direction]}" + ("（先单眨眼切换）" if direction != cue else "，保持专注")
            elif cue == REST_CUE:
                hint = f"保持放松 {args.rest_hold:.0f} 秒，不要眨眼"
            display.show(CUE_NAMES[cue], hint, f"{index}/{len(schedule)}", cue=cue)
            result = run_cue_trial(
                index, cue, step=step, poll_move=poll_move,
                timeout_sec=args.rest_hold if cue == REST_CUE else args.timeout,
                clock=clock, direction_at_cue=direction,
            )
            results.append(result)
            latency = f"，耗时 {result.latency_sec:.1f}s" if result.latency_sec is not None else ""
            print(f"[{index}/{len(schedule)}] 口令 {CUE_NAMES[cue]} -> {OUTCOME_NAMES[result.outcome]}{latency}")
            display.show(OUTCOME_NAMES[result.outcome], latency.lstrip("，"), f"{index}/{len(schedule)}")
            wait(1.0)
    except KeyboardInterrupt:
        print("\n口令测试被中断，保存已完成的部分")
    finally:
        controller.reader.stop()
        display.close()

    if not results:
        print("没有完成任何口令，不生成报告")
        return 1
    summary = summarize(results)
    summary["config"] = {
        "device": args.device, "subject": args.subject, "seed": seed, "mock_headset": args.mock_headset,
        "turning_source": getattr(config, "turning_source", "rule"),
        "focus_threshold": getattr(config, "focus_threshold", None),
        "min_decision_count": getattr(config, "min_decision_count", None),
        "window_size": config.window_size, "sample_interval": config.sample_interval,
        "timeout_sec": args.timeout, "rest_hold_sec": args.rest_hold,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output_dir / f"{args.device}_{args.subject}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    rows = results_to_rows(results)
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    title = f"口令测试结果（{args.device}，被试 {args.subject}{'，模拟脑环' if args.mock_headset else ''}）"
    markdown = format_summary_markdown(summary, title=title)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"结果已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

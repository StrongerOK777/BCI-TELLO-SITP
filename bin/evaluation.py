"""Cued online evaluation: schedule cues, time each response, summarize.

A cue test shows the operator one command at a time (``left`` / ``right`` /
``rest``) and watches which movement the controller actually emits:

* ``left`` / ``right`` cue - correct when the first movement matches the cue,
  ``wrong`` when it is the opposite direction, ``timeout`` when nothing moves
  within the time limit. Latency is cue onset -> first movement.
* ``rest`` cue - correct when nothing moves for the whole hold period,
  ``false_move`` otherwise.

Stdlib-only (``summarize`` borrows ``bin.training.significance_threshold``);
clock/step/poll are injected, so the trial loop is unit-tested without a
headset or hardware.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

MOVE_CUES = ("left", "right")
REST_CUE = "rest"
CUE_NAMES = {"left": "左转", "right": "右转", "rest": "静止"}
OUTCOME_NAMES = {
    "correct": "命中",
    "wrong": "方向错误",
    "timeout": "超时未动作",
    "false_move": "误动作",
}


@dataclass
class CueTrialResult:
    index: int
    cue: str
    outcome: str  # correct | wrong | timeout | false_move
    response: Optional[str]  # first movement ("left"/"right") or None
    latency_sec: Optional[float]  # cue -> first movement; None when nothing moved
    duration_sec: float  # time spent on this trial
    windows: int  # controller steps (EEG windows) used
    invalid_windows: int = 0
    direction_at_cue: str = ""  # steering direction selected when the cue appeared

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


def build_cue_schedule(
    per_class: int,
    *,
    cues: Sequence[str] = ("left", "right", "rest"),
    seed: int = 0,
    max_run: int = 2,
) -> List[str]:
    """Balanced random order with no more than ``max_run`` identical cues in a row."""

    rng = random.Random(seed)
    pool = [cue for cue in cues for _ in range(per_class)]
    for _ in range(1000):
        rng.shuffle(pool)
        if all(len(set(pool[i : i + max_run + 1])) > 1 for i in range(len(pool) - max_run)):
            return list(pool)
    return list(pool)  # tiny or degenerate pools: best effort


def run_cue_trial(
    index: int,
    cue: str,
    *,
    step: Callable[[], Optional[bool]],
    poll_move: Callable[[], Optional[str]],
    timeout_sec: float,
    clock: Callable[[], float] = time.monotonic,
    direction_at_cue: str = "",
) -> CueTrialResult:
    """Drive one cue: call ``step()`` (one EEG window) until a verdict.

    ``step`` returns ``False`` for an invalid window (counted, not fatal).
    ``poll_move`` returns the first new movement since the last call, or None.
    For a ``rest`` cue ``timeout_sec`` is the hold period that must stay still.
    """

    poll_move()  # discard anything emitted before the cue appeared
    start = clock()
    windows = invalid = 0
    response: Optional[str] = None
    while clock() - start < timeout_sec:
        valid = step()
        windows += 1
        if valid is False:
            invalid += 1
        response = poll_move()
        if response is not None:
            break
    elapsed = clock() - start
    latency = elapsed if response is not None else None
    if cue == REST_CUE:
        outcome = "false_move" if response is not None else "correct"
    elif response is None:
        outcome = "timeout"
    else:
        outcome = "correct" if response == cue else "wrong"
    return CueTrialResult(index, cue, outcome, response, latency, elapsed, windows, invalid, direction_at_cue)


def wolpaw_itr(n_classes: int, accuracy: float, seconds_per_selection: float) -> float:
    """Wolpaw information transfer rate in bits/min (0 when at/below chance)."""

    if n_classes < 2 or seconds_per_selection <= 0:
        return 0.0
    p = min(max(accuracy, 0.0), 1.0)
    if p <= 1.0 / n_classes:
        return 0.0
    bits = math.log2(n_classes) + p * math.log2(p)
    if p < 1.0:
        bits += (1 - p) * math.log2((1 - p) / (n_classes - 1))
    return bits * 60.0 / seconds_per_selection


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def summarize(results: Sequence[CueTrialResult], *, alpha: float = 0.05) -> Dict:
    """Per-cue and overall statistics, confusion matrix, significance and ITR."""

    from .training import significance_threshold

    cues = [c for c in ("left", "right", "rest") if any(r.cue == c for r in results)]
    per_cue: Dict[str, Dict] = {}
    for cue in cues:
        items = [r for r in results if r.cue == cue]
        latencies = [r.latency_sec for r in items if r.correct and r.latency_sec is not None]
        per_cue[cue] = {
            "n": len(items),
            "correct": sum(r.correct for r in items),
            "accuracy": sum(r.correct for r in items) / len(items),
            "mean_latency_sec": _mean(latencies),
            "median_latency_sec": _median(latencies),
            "outcomes": {o: sum(r.outcome == o for r in items) for o in OUTCOME_NAMES if any(r.outcome == o for r in items)},
        }
    n = len(results)
    correct = sum(r.correct for r in results)
    accuracy = correct / n if n else 0.0
    mean_trial = _mean([r.duration_sec for r in results]) or 0.0
    move_hits = [r.latency_sec for r in results if r.cue in MOVE_CUES and r.correct and r.latency_sec is not None]
    confusion = {cue: {resp: 0 for resp in ("left", "right", "none")} for cue in cues}
    for r in results:
        confusion[r.cue][r.response or "none"] += 1
    n_classes = max(2, len(cues))
    threshold = significance_threshold(n, n_classes, alpha)
    return {
        "n_trials": n,
        "correct": correct,
        "accuracy": accuracy,
        "chance": 1.0 / n_classes,
        "significance_threshold": threshold,
        "significant": accuracy >= threshold,
        "mean_trial_sec": mean_trial,
        "mean_move_latency_sec": _mean(move_hits),
        "itr_bits_per_min": wolpaw_itr(n_classes, accuracy, mean_trial),
        "invalid_windows": sum(r.invalid_windows for r in results),
        "total_windows": sum(r.windows for r in results),
        "per_cue": per_cue,
        "confusion": confusion,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _fmt_sec(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f} s"


def format_summary_markdown(summary: Dict, *, title: str = "口令测试结果") -> str:
    """Chinese Markdown tables ready to paste into the defense slides/report."""

    lines = [f"## {title}", ""]
    lines += ["| 口令 | 次数 | 命中 | 命中率 | 平均耗时（命中） | 中位耗时 | 失败情况 |", "|---|---|---|---|---|---|---|"]
    for cue, stats in summary["per_cue"].items():
        failures = "，".join(
            f"{OUTCOME_NAMES[o]} {k}" for o, k in stats["outcomes"].items() if o != "correct"
        ) or "—"
        lines.append(
            f"| {CUE_NAMES[cue]} | {stats['n']} | {stats['correct']} | {_fmt_pct(stats['accuracy'])} | "
            f"{_fmt_sec(stats['mean_latency_sec'])} | {_fmt_sec(stats['median_latency_sec'])} | {failures} |"
        )
    sig = "是" if summary["significant"] else "否"
    lines += [
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 总体命中率 | **{_fmt_pct(summary['accuracy'])}**（{summary['correct']}/{summary['n_trials']}） |",
        f"| 随机猜测 / 显著性阈值（p<0.05） | {_fmt_pct(summary['chance'])} / {_fmt_pct(summary['significance_threshold'])}，显著：{sig} |",
        f"| 方向指令平均响应时间 | {_fmt_sec(summary['mean_move_latency_sec'])} |",
        f"| 每条口令平均用时 | {_fmt_sec(summary['mean_trial_sec'])} |",
        f"| 信息传输率 ITR（Wolpaw） | {summary['itr_bits_per_min']:.2f} bit/min |",
        f"| 信号无效窗口 | {summary['invalid_windows']}/{summary['total_windows']} |",
        "",
        "| 口令 \\ 实际动作 | 左转 | 右转 | 无动作 |",
        "|---|---|---|---|",
    ]
    for cue, row in summary["confusion"].items():
        lines.append(f"| {CUE_NAMES[cue]} | {row['left']} | {row['right']} | {row['none']} |")
    return "\n".join(lines) + "\n"


def results_to_rows(results: Sequence[CueTrialResult]) -> List[Dict]:
    return [asdict(r) for r in results]

"""One brain-control state machine shared by every device.

Every device (car, drone, arm) is driven by the same :class:`BrainControlSession`.
A device only declares which intents it supports and how to perform them
(:class:`Device`); the session turns EEG windows into those intents:

* modes cycle 升降 -> 前后 -> 转弯 (only the ones the device supports);
* a double blink switches to the next mode;
* a single blink flips the steering direction in 转弯 (rule source) and runs
  the device's ``aux`` action elsewhere (e.g. the arm gripper);
* 升降 / 前后 compare attention vs meditation counts with a dead-band and a
  margin; 转弯 uses :class:`bin.decoding.FocusSteer` or, opt-in, the shared
  model behind a softmax confidence gate;
* an optional :class:`bin.decoding.VoteWindow` debounces every decision;
* a disconnected headset triggers the device's ``emergency_stop`` and resets
  the mode; any decoding failure falls back to ``rest``.

Stdlib-only apart from the sibling ``bin`` modules; all inputs are injected so
the whole machine is unit-tested with fakes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .decoding import FocusSteer, VoteWindow, gate_by_confidence
from .eeg import FeatureWindowResult, WindowResult
from .models import LABELS

REST = "rest"
AUX = "aux"
MOVE_INTENTS = ("forward", "backward", "left", "right", "up", "down")
INTENT_NAMES = {
    "forward": "前进", "backward": "后退", "left": "左转", "right": "右转",
    "up": "上升", "down": "下降", REST: "静止", AUX: "附加动作",
}

MODE_VERTICAL = "vertical"
MODE_FORWARD_BACKWARD = "forward_backward"
MODE_TURNING = "turning"
MODE_ORDER = (MODE_VERTICAL, MODE_FORWARD_BACKWARD, MODE_TURNING)
MODE_NAMES = {MODE_VERTICAL: "升降", MODE_FORWARD_BACKWARD: "前后", MODE_TURNING: "转弯"}
# (attention-high intent, meditation-high intent); 转弯 is steered instead.
MODE_INTENTS = {
    MODE_VERTICAL: ("up", "down"),
    MODE_FORWARD_BACKWARD: ("forward", "backward"),
    MODE_TURNING: ("left", "right"),
}
DIRECTION_NAMES = {"left": "左", "right": "右"}


@dataclass
class ActionResult:
    intent: str
    executed: bool
    message: str = ""


class Device(Protocol):
    """What a device adapter must offer to be brain-controlled."""

    name: str
    capabilities: frozenset  # subset of MOVE_INTENTS, plus AUX if supported
    actions: List[str]  # executed intents, in order (for tests / UI)

    def connect(self) -> None: ...

    def close(self) -> None: ...  # must leave the hardware safe (land / home / stop)

    def execute(self, intent: str) -> ActionResult: ...

    def rest(self) -> ActionResult: ...

    def emergency_stop(self) -> ActionResult: ...

    def aux(self) -> ActionResult: ...


class DeviceBase:
    """Shared plumbing for adapters: capability check + executed-intent log.

    Subclasses set ``name`` / ``capabilities`` and implement ``_perform``
    (and optionally ``_rest`` / ``_emergency`` / ``_aux``), each returning a
    short Chinese message.
    """

    name = "device"
    capabilities: frozenset = frozenset()

    def __init__(self) -> None:
        self.actions: List[str] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def execute(self, intent: str) -> ActionResult:
        if intent not in self.capabilities or intent == AUX:
            return ActionResult(intent, False, f"不支持 {INTENT_NAMES.get(intent, intent)}")
        refusal = self._refuse(intent)
        if refusal:
            return ActionResult(intent, False, refusal)
        message = self._perform(intent)
        self.actions.append(intent)
        return ActionResult(intent, True, message)

    def rest(self) -> ActionResult:
        return ActionResult(REST, True, self._rest())

    def emergency_stop(self) -> ActionResult:
        message = self._emergency()
        self.actions.append("emergency_stop")
        return ActionResult("emergency_stop", True, message)

    def aux(self) -> ActionResult:
        if AUX not in self.capabilities:
            return ActionResult(AUX, False, "没有附加动作")
        message = self._aux()
        self.actions.append(AUX)
        return ActionResult(AUX, True, message)

    # hooks -------------------------------------------------------------------
    def _refuse(self, intent: str) -> str:
        return ""

    def _perform(self, intent: str) -> str:
        raise NotImplementedError

    def _rest(self) -> str:
        return "保持静止"

    def _emergency(self) -> str:
        return self._rest()

    def _aux(self) -> str:
        return ""


@dataclass
class ControlConfig:
    window_size: int = 30
    mode_switch_blinks: int = 2
    # dead-band: both counts below this -> rest; also the FocusSteer focus count
    min_decision_count: int = 20
    # attention/meditation counts must differ by at least this to pick a side
    decision_margin: int = 3
    turning_source: str = "rule"  # "rule" = FocusSteer, "model" = shared model
    focus_threshold: int = 55
    confidence_threshold: float = 0.5
    # majority vote over the last vote_window decisions; 1/1 = off
    vote_window: int = 1
    vote_min: int = 1


@dataclass
class StepReport:
    """Everything one EEG window produced; pushed to listeners (UI, logs)."""

    mode: str
    mode_name: str
    valid: bool = True
    attention_count: int = 0
    meditation_count: int = 0
    blink_count: int = 0
    attention_values: List[int] = field(default_factory=list)
    raw_intent: str = REST
    intent: str = REST
    executed: bool = False
    message: str = ""
    direction: str = "left"
    event: str = "decision"  # decision | invalid | disconnect | mode_switch | flip | aux | halted

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data.pop("attention_values", None)
        return data


def available_modes(capabilities: Sequence[str]) -> Tuple[str, ...]:
    caps = set(capabilities)
    return tuple(mode for mode in MODE_ORDER if set(MODE_INTENTS[mode]) <= caps)


class BrainControlSession:
    def __init__(
        self,
        config: ControlConfig,
        reader: Any,
        device: Device,
        predictor: Any = None,
        *,
        listeners: Sequence[Callable[[StepReport], None]] = (),
    ) -> None:
        self.config = config
        self.reader = reader
        self.device = device
        self.predictor = predictor
        self.listeners: List[Callable[[StepReport], None]] = list(listeners)
        self.modes = available_modes(device.capabilities)
        if not self.modes:
            raise ValueError(f"设备 {device.name} 不支持任何控制模式")
        self.mode = self.modes[0]
        self.steer = FocusSteer(
            ("left", "right"),
            focus_threshold=config.focus_threshold,
            min_focus_count=config.min_decision_count,
            idle_intent=REST,
        )
        self._vote = VoteWindow(config.vote_window, config.vote_min, idle_intent=REST)
        self._device_lock = threading.RLock()
        self._halted = False

    # -- configuration -------------------------------------------------------
    def apply_config(self, **changes: Any) -> None:
        """Update tunables live (focus threshold, votes, ...) between windows."""

        for key, value in changes.items():
            if not hasattr(self.config, key):
                raise KeyError(key)
            setattr(self.config, key, type(getattr(self.config, key))(value))
        self.steer.focus_threshold = int(self.config.focus_threshold)
        self.steer.min_focus_count = max(1, int(self.config.min_decision_count))
        self._vote = VoteWindow(self.config.vote_window, self.config.vote_min, idle_intent=REST)

    def reset_votes(self) -> None:
        self._vote.reset()

    # -- main loop -------------------------------------------------------------
    def run(self, stop_event: Optional[threading.Event] = None, *, max_windows: Optional[int] = None,
            manage_device: bool = True) -> None:
        stop_event = stop_event or threading.Event()
        self._halted = False
        if self.config.turning_source == "model" and self.predictor is not None:
            from .models import report_model

            report_model(self.predictor)
        seen = 0
        try:
            if manage_device:
                self.device.connect()
            self.reader.start()
            while not stop_event.is_set():
                self.step()
                seen += 1
                if max_windows is not None and seen >= max_windows:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.reader.stop()
            finally:
                if manage_device:
                    with self._device_lock:
                        self.device.close()

    def step(self) -> StepReport:
        mode = self.mode
        use_model = mode == MODE_TURNING and self.config.turning_source == "model"
        name = MODE_NAMES[mode]
        result = self.reader.collect_feature_window(name) if use_model else self.reader.collect_rule_window(name)
        report = StepReport(
            mode=mode,
            mode_name=name,
            valid=result.valid,
            attention_count=result.attention_count,
            meditation_count=result.meditation_count,
            blink_count=result.blink_count,
            attention_values=[s.attention for s in result.samples],
            direction=self.steer.direction,
        )
        if self._halted:
            report.event, report.message = "halted", "已急停，本窗口不执行"
        elif not result.valid:
            self._handle_invalid(result, report)
        elif result.blink_count >= self.config.mode_switch_blinks:
            self._next_mode(report)
        elif result.blink_count == 1:
            self._single_blink(report)
        else:
            self._decide(result, report, use_model)
        print(f"{name}模式 | {report.message}")
        self._notify(report)
        return report

    # -- manual / safety -----------------------------------------------------
    def execute_manual(self, intent: str) -> ActionResult:
        if intent == REST:
            return self._call(self.device.rest)
        if intent == AUX:
            if AUX not in self.device.capabilities:
                return ActionResult(AUX, False, f"{self.device.name} 没有附加动作")
            return self._call(self.device.aux)
        if intent not in self.device.capabilities:
            return ActionResult(intent, False, f"{self.device.name} 不支持 {INTENT_NAMES.get(intent, intent)}")
        return self._call(self.device.execute, intent)

    def emergency_stop(self) -> ActionResult:
        self._halted = True
        self._reset_to_start()
        return self._call(self.device.emergency_stop)

    def resume(self) -> None:
        self._halted = False

    @property
    def halted(self) -> bool:
        return self._halted

    # -- internals -------------------------------------------------------------
    def _call(self, fn: Callable, *args: Any) -> ActionResult:
        with self._device_lock:
            try:
                return fn(*args)
            except Exception as exc:  # hardware errors must not kill the loop
                return ActionResult(args[0] if args else REST, False, f"设备执行失败：{exc}")

    def _notify(self, report: StepReport) -> None:
        for listener in list(self.listeners):
            try:
                listener(report)
            except Exception as exc:
                print(f"状态推送失败：{exc}")

    def _reset_to_start(self) -> None:
        self.mode = self.modes[0]
        self._vote.reset()

    def _handle_invalid(self, result: WindowResult, report: StepReport) -> None:
        detail = result.reason or f"poorSignal={result.poor_signal}"
        if result.disconnected:
            outcome = self._call(self.device.emergency_stop)
            self._reset_to_start()
            report.event = "disconnect"
            report.executed = outcome.executed
            report.message = f"脑环断开（{detail}），执行安全动作：{outcome.message}"
        else:
            report.event = "invalid"
            report.message = f"脑环信号异常，仅报警不动作：{detail}"

    def _next_mode(self, report: StepReport) -> None:
        old = self.mode
        self.mode = self.modes[(self.modes.index(old) + 1) % len(self.modes)]
        self._vote.reset()
        self._call(self.device.rest)
        report.event = "mode_switch"
        report.message = f"双眨眼：{MODE_NAMES[old]} → {MODE_NAMES[self.mode]}"

    def _single_blink(self, report: StepReport) -> None:
        if self.mode == MODE_TURNING and self.config.turning_source != "model":
            direction = self.steer.flip()
            self._vote.reset()
            report.event, report.direction = "flip", direction
            report.message = f"单眨眼：方向切换为{DIRECTION_NAMES[direction]}，本窗口不动作"
        elif AUX in self.device.capabilities:
            outcome = self._call(self.device.aux)
            self._vote.reset()
            report.event, report.intent, report.executed = "aux", AUX, outcome.executed
            report.message = f"单眨眼：{outcome.message}"
        else:
            report.event = "aux"
            report.message = "检测到单眨眼，重新测量"

    def _decide(self, result: WindowResult, report: StepReport, use_model: bool) -> None:
        raw, reason = self._decode(result, use_model)
        committed = self._vote.push(raw)
        report.raw_intent, report.intent = raw, committed
        if committed == REST:
            outcome = self._call(self.device.rest)
            waiting = "，等待投票" if raw != REST else ""
            report.message = f"{reason} → 静止{waiting}"
        else:
            outcome = self._call(self.device.execute, committed)
            report.message = f"{reason} → {INTENT_NAMES[committed]}：{outcome.message}"
        report.executed = outcome.executed and committed != REST

    def _decode(self, result: WindowResult, use_model: bool) -> Tuple[str, str]:
        if self.mode == MODE_TURNING:
            if use_model:
                return self._decode_model(result)
            decision = self.steer.decide([s.attention for s in result.samples])
            return decision.intent, f"当前方向{DIRECTION_NAMES[self.steer.direction]}，{decision.reason}"
        attention_intent, meditation_intent = MODE_INTENTS[self.mode]
        a, m = result.attention_count, result.meditation_count
        counts = f"Attention {a} / Meditation {m}"
        if a < self.config.min_decision_count and m < self.config.min_decision_count:
            return REST, f"{counts}，都低于 {self.config.min_decision_count}"
        if a >= m + self.config.decision_margin:
            return attention_intent, f"{counts}，专注更高"
        if m >= a + self.config.decision_margin:
            return meditation_intent, f"{counts}，放松更高"
        return REST, f"{counts}，差值小于 {self.config.decision_margin}"

    def _decode_model(self, result: WindowResult) -> Tuple[str, str]:
        if self.predictor is None or not isinstance(result, FeatureWindowResult) or not result.feature_window:
            return REST, "缺少模型特征"
        try:
            probabilities = self.predictor.predict_proba(result.feature_window)
        except Exception as exc:
            return REST, f"模型不可用：{exc}"
        decision = gate_by_confidence(probabilities, LABELS, self.config.confidence_threshold, reject_intent=REST)
        return decision.intent, f"模型预测 {decision.intent}（{decision.reason}）"

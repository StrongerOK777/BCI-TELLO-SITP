"""ControlHub: one place that owns the device, the headset and the brain loop.

The web panel (``control_panel/app.py``) is a thin HTTP layer over this class;
keeping the lifecycle here (stdlib only) means it is unit-tested without Flask
or hardware. Devices are built by injected factories, so ``bin`` never imports
a device folder.

Rules:
* only one device is connected at a time; connecting another disconnects first;
* real hardware is refused unless the hub was created with ``allow_real=True``;
* manual commands are refused while the brain loop runs (pause it first);
* the emergency stop always works: it halts the loop and runs the device's
  safety action (stop / land / home) immediately.

Every change is published to subscribers as ``(event, payload)`` through
bounded queues (the SSE stream reads them); a slow subscriber drops events
instead of blocking the control loop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cli import BrainSettings, build_reader, build_session
from .control import INTENT_NAMES, MODE_NAMES, BrainControlSession, StepReport
from .eeg import EEGSnapshot, SimulatedBrainSignalReader, SimulatedHeadset

# factory(real_hardware) -> (settings, device)
DeviceFactory = Callable[[bool], Tuple[BrainSettings, Any]]

TUNABLES = ("focus_threshold", "min_decision_count", "decision_margin", "vote_window", "vote_min",
            "confidence_threshold", "turning_source")


@dataclass
class DeviceOption:
    key: str
    label: str
    factory: DeviceFactory


class ControlHub:
    def __init__(self, devices: List[DeviceOption], *, allow_real: bool = False,
                 sim_sample_interval: Optional[float] = None) -> None:
        self.devices = {option.key: option for option in devices}
        self.allow_real = allow_real
        self.sim_sample_interval = sim_sample_interval
        self.session: Optional[BrainControlSession] = None
        self.settings: Optional[BrainSettings] = None
        self.device_key: Optional[str] = None
        self.real_hardware = False
        self.real_headset = False
        self.headset: Optional[SimulatedHeadset] = None
        self.last_report: Optional[Dict[str, Any]] = None
        self.overrides: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._subscribers: List[queue.Queue] = []
        self._sub_lock = threading.Lock()

    # -- pub/sub ---------------------------------------------------------------
    def subscribe(self, maxsize: int = 500) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._sub_lock:
            self._subscribers.append(q)
        q.put_nowait(("state", self.state()))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: str, payload: Dict[str, Any]) -> None:
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait((event, payload))
            except queue.Full:
                pass

    def log(self, message: str, level: str = "info") -> None:
        print(message)
        self.publish("log", {"message": message, "level": level, "t": time.time()})

    # -- state -----------------------------------------------------------------
    @property
    def running(self) -> bool:
        # stopping counts as not running: the loop executes nothing after stop/emergency
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def state(self) -> Dict[str, Any]:
        session = self.session
        device = session.device if session else None
        config = session.config if session else None
        return {
            "devices": [{"key": o.key, "label": o.label} for o in self.devices.values()],
            "allow_real": self.allow_real,
            "connected": session is not None,
            "device": self.device_key,
            "real_hardware": self.real_hardware,
            "real_headset": self.real_headset,
            "running": self.running,
            "halted": bool(session and session.halted),
            "mode": session.mode if session else None,
            "mode_name": MODE_NAMES.get(session.mode) if session else None,
            "modes": [{"key": m, "name": MODE_NAMES[m]} for m in session.modes] if session else [],
            "direction": session.steer.direction if session else None,
            "capabilities": sorted(device.capabilities) if device else [],
            "intent_names": INTENT_NAMES,
            "config": {name: getattr(config, name) for name in TUNABLES} if config else {},
            "headset": None if self.headset is None else {
                "attention": self.headset.target_attention,
                "meditation": self.headset.target_meditation,
                "poor_signal": self.headset.poorSignal,
            },
            "last_report": self.last_report,
        }

    def _publish_state(self) -> None:
        self.publish("state", self.state())

    # -- lifecycle -------------------------------------------------------------
    def connect(self, device_key: str, *, real_hardware: bool = False, real_headset: bool = False) -> Dict[str, Any]:
        if device_key not in self.devices:
            raise KeyError(f"未知设备：{device_key}")
        if real_hardware and not self.allow_real:
            raise PermissionError("服务未以 --real 启动，不允许连接真实硬件")
        with self._lock:
            if self.session is not None:
                self.disconnect()
            settings, device = self.devices[device_key].factory(real_hardware)
            for key, value in self.overrides.items():
                if hasattr(settings, key):
                    setattr(settings, key, type(getattr(settings, key))(value))
            if real_headset:
                self.headset = None
                reader = build_reader(settings, sample_listener=self._on_sample)
            else:
                self.headset = SimulatedHeadset()
                interval = self.sim_sample_interval or settings.sample_interval
                reader = SimulatedBrainSignalReader(
                    self.headset,
                    window_size=settings.window_size,
                    sample_interval=interval,
                    attention_threshold=settings.attention_threshold,
                    meditation_threshold=settings.meditation_threshold,
                    blink_threshold=settings.blink_threshold,
                    poor_signal_threshold=settings.poor_signal_threshold,
                    sample_listener=self._on_sample,
                )
            session = build_session(settings, device, reader=reader, listeners=[self._on_report])
            device.connect()
            self.session, self.settings, self.device_key = session, settings, device_key
            self.real_hardware, self.real_headset = real_hardware, real_headset
            self.last_report = None
        self.log(f"已连接 {self.devices[device_key].label}（{'真实硬件' if real_hardware else '模拟设备'}，"
                 f"{'真实脑环' if real_headset else '模拟脑环'}）")
        self._publish_state()
        return self.state()

    def disconnect(self) -> Dict[str, Any]:
        with self._lock:
            if self.session is None:
                return self.state()
            self.stop_brain()
            try:
                self.session.device.close()
            except Exception as exc:
                self.log(f"断开设备时出错：{exc}", "error")
            label = self.devices[self.device_key].label if self.device_key else ""
            self.session, self.settings, self.device_key, self.headset = None, None, None, None
            self.real_hardware = self.real_headset = False
        self.log(f"已断开 {label}")
        self._publish_state()
        return self.state()

    def start_brain(self) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session()
            if self.running:
                return self.state()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(self._window_timeout())  # let a stopping loop finish its window
            session.resume()
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run_loop, args=(session, self._stop), name="brain-control", daemon=True
            )
            self._thread.start()
        self.log("脑控已启动")
        self._publish_state()
        return self.state()

    def _run_loop(self, session: BrainControlSession, stop: threading.Event) -> None:
        try:
            session.run(stop, manage_device=False)
        except Exception as exc:  # e.g. headset serial port missing
            self.log(f"脑控停止：{exc}", "error")
            try:
                session.execute_manual("rest")
            except Exception:
                pass
        finally:
            self._publish_state()

    def stop_brain(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        thread = self._thread
        if thread is None:
            return self.state()
        self._stop.set()
        thread.join(self._window_timeout() if timeout is None else timeout)
        self._thread = None
        if self.session is not None:
            self.session.execute_manual("rest")
        self.log("脑控已暂停")
        self._publish_state()
        return self.state()

    def command(self, intent: str) -> Dict[str, Any]:
        session = self._require_session()
        if self.running:
            raise RuntimeError("脑控运行中，请先暂停脑控再手动操作")
        result = session.execute_manual(intent)
        payload = {"source": "manual", "intent": result.intent, "executed": result.executed,
                   "message": result.message, "t": time.time()}
        self.publish("action", payload)
        self.log(f"手动 {INTENT_NAMES.get(intent, intent)}：{result.message}",
                 "info" if result.executed else "warn")
        self._publish_state()
        return payload

    def emergency(self) -> Dict[str, Any]:
        session = self.session
        if session is None:
            return {"executed": False, "message": "未连接设备"}
        self._stop.set()  # the loop exits after the current window; nothing more executes
        result = session.emergency_stop()
        payload = {"source": "emergency", "intent": result.intent, "executed": result.executed,
                   "message": result.message, "t": time.time()}
        self.publish("action", payload)
        self.log(f"急停：{result.message}", "error")
        self._publish_state()
        return payload

    def update_config(self, **changes: Any) -> Dict[str, Any]:
        unknown = set(changes) - set(TUNABLES)
        if unknown:
            raise KeyError(f"不可修改的参数：{', '.join(sorted(unknown))}")
        if "turning_source" in changes and changes["turning_source"] not in ("rule", "model"):
            raise ValueError("turning_source 只能是 rule 或 model")
        self.overrides.update(changes)
        if self.session is not None:
            self.session.apply_config(**changes)
            if changes.get("turning_source") == "model" and self.session.predictor is None and self.settings:
                from .cli import build_predictor

                self.session.predictor = build_predictor(self.settings)
        self._publish_state()
        return self.state()

    def set_sim_headset(self, *, attention: Optional[int] = None, meditation: Optional[int] = None,
                        poor_signal: Optional[int] = None, blinks: int = 0) -> Dict[str, Any]:
        if self.headset is None:
            raise RuntimeError("当前未使用模拟脑环")
        self.headset.set(attention=attention, meditation=meditation, poor_signal=poor_signal)
        if blinks:
            self.headset.queue_blinks(blinks)
        return self.state()["headset"]

    # -- listeners -------------------------------------------------------------
    def _on_sample(self, snapshot: EEGSnapshot) -> None:
        self.publish("sample", {
            "t": snapshot.timestamp, "attention": snapshot.attention, "meditation": snapshot.meditation,
            "poor_signal": snapshot.poorSignal, "blink": snapshot.blinkStrength,
        })

    def _on_report(self, report: StepReport) -> None:
        payload = report.to_dict()
        payload["t"] = time.time()
        self.last_report = payload
        self.publish("report", payload)
        self._publish_state()

    def _window_timeout(self) -> float:
        settings = self.settings
        return (settings.window_size * settings.sample_interval + 2.0) if settings else 5.0

    def _require_session(self) -> BrainControlSession:
        if self.session is None:
            raise RuntimeError("请先连接设备")
        return self.session

    def shutdown(self) -> None:
        self.disconnect()

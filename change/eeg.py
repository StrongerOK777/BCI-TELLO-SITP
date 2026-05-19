"""EEG acquisition and feature utilities shared by hardware controllers."""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence


DEFAULT_MINDWAVE_BAUD = 57600


@dataclass
class EEGSnapshot:
    attention: int = 0
    meditation: int = 0
    delta: int = 0
    theta: int = 0
    lowAlpha: int = 0
    highAlpha: int = 0
    lowBeta: int = 0
    highBeta: int = 0
    lowGamma: int = 0
    midGamma: int = 0
    rawValue: int = 0
    poorSignal: int = 0
    blinkStrength: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WindowResult:
    mode_name: str
    attention_count: int = 0
    meditation_count: int = 0
    blink_count: int = 0
    poor_signal: int = 0
    valid: bool = True
    disconnected: bool = False
    reason: str = ""
    samples: List[EEGSnapshot] = field(default_factory=list)


@dataclass
class FeatureWindowResult(WindowResult):
    feature_window: List[List[float]] = field(default_factory=list)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def add_sys_path(path: Optional[str]) -> None:
    if not path:
        return
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)


def import_neurosky(neuropy_dir: Optional[str] = None) -> Any:
    """Import ``NeuroSkyPy`` from an explicitly supplied dependency location."""

    # Try direct module import first
    try:
        import neuropy
        return neuropy.NeuroSkyPy
    except ImportError:
        pass
    
    # Try from specified directory
    if neuropy_dir:
        add_sys_path(neuropy_dir)
        try:
            import neuropy
            return neuropy.NeuroSkyPy
        except ImportError:
            pass
    
    # Try from environment variable
    neuropy_env = os.getenv("NEUROPY_DIR")
    if neuropy_env:
        add_sys_path(neuropy_env)
        try:
            import neuropy
            return neuropy.NeuroSkyPy
        except ImportError:
            pass
    
    raise RuntimeError(
        "Could not import NeuroSkyPy. Ensure neuropy.py is in the Python path or "
        "set --neuropy-dir or NEUROPY_DIR environment variable to the folder containing neuropy.py."
    )


def build_feature_vector(
    attention: int,
    meditation: int,
    delta: int,
    theta: int,
    low_alpha: int,
    high_alpha: int,
    low_beta: int,
    high_beta: int,
    low_gamma: int,
    mid_gamma: int,
    blink_strength: int,
) -> List[float]:
    beta = low_beta + high_beta
    alpha = low_alpha + high_alpha
    theta_safe = theta if theta != 0 else 1e-6
    beta_theta_ratio = beta / theta_safe
    alpha_theta_ratio = alpha / theta_safe
    engagement = beta / (alpha + 1e-6)
    return [
        float(attention),
        float(meditation),
        float(delta),
        float(theta),
        float(low_alpha),
        float(high_alpha),
        float(low_beta),
        float(high_beta),
        float(low_gamma),
        float(mid_gamma),
        float(blink_strength),
        float(beta_theta_ratio),
        float(alpha_theta_ratio),
        float(engagement),
    ]


def build_feature_vector_from_snapshot(snapshot: EEGSnapshot) -> List[float]:
    return build_feature_vector(
        snapshot.attention,
        snapshot.meditation,
        snapshot.delta,
        snapshot.theta,
        snapshot.lowAlpha,
        snapshot.highAlpha,
        snapshot.lowBeta,
        snapshot.highBeta,
        snapshot.lowGamma,
        snapshot.midGamma,
        snapshot.blinkStrength,
    )


class BrainSignalReader:
    """Windowed NeuroSky reader with signal-quality and blink handling."""

    def __init__(
        self,
        port: Optional[str] = None,
        baud: Optional[int] = None,
        *,
        neuropy_dir: Optional[str] = None,
        window_size: int = 30,
        sample_interval: float = 0.1,
        attention_threshold: int = 30,
        meditation_threshold: int = 50,
        blink_threshold: int = 100,
        poor_signal_threshold: int = 20,
        blink_debounce_sec: float = 0.5,
        device_factory: Optional[Callable[[str, int], Any]] = None,
    ) -> None:
        self.port = port or os.getenv("MINDWAVE_PORT")
        self.baud = int(baud or os.getenv("MINDWAVE_BAUD", str(DEFAULT_MINDWAVE_BAUD)))
        self.neuropy_dir = neuropy_dir
        self.window_size = max(1, int(window_size))
        self.sample_interval = max(0.001, float(sample_interval))
        self.attention_threshold = attention_threshold
        self.meditation_threshold = meditation_threshold
        self.blink_threshold = blink_threshold
        self.poor_signal_threshold = poor_signal_threshold
        self.blink_debounce_sec = max(0.0, blink_debounce_sec)
        self.device_factory = device_factory
        self.device: Any = None
        self.running = False
        self.last_blink_time = 0.0
        self.blink_active = False

    def start(self) -> None:
        if self.port is None:
            raise RuntimeError("MindWave port is required. Pass --mindwave-port or set MINDWAVE_PORT.")
        print(f"[BrainSignalReader] 初始化脑环设备...")
        print(f"[BrainSignalReader] 端口: {self.port}, 波特率: {self.baud}")
        factory = self.device_factory or import_neurosky(self.neuropy_dir)
        print(f"[BrainSignalReader] NeuroSkyPy 加载成功")
        self.device = factory(self.port, self.baud)
        print(f"[BrainSignalReader] 启动脑环设备...")
        self.device.start()
        print(f"[BrainSignalReader] 脑环设备启动成功")
        self.running = True

    def stop(self) -> None:
        self.running = False
        if self.device is not None:
            self.device.stop()

    def read_snapshot(self) -> EEGSnapshot:
        if self.device is None:
            raise RuntimeError("BrainSignalReader.start() must be called before reading")
        return EEGSnapshot(
            attention=safe_int(getattr(self.device, "attention", 0)),
            meditation=safe_int(getattr(self.device, "meditation", 0)),
            delta=safe_int(getattr(self.device, "delta", 0)),
            theta=safe_int(getattr(self.device, "theta", 0)),
            lowAlpha=safe_int(getattr(self.device, "lowAlpha", 0)),
            highAlpha=safe_int(getattr(self.device, "highAlpha", 0)),
            lowBeta=safe_int(getattr(self.device, "lowBeta", 0)),
            highBeta=safe_int(getattr(self.device, "highBeta", 0)),
            lowGamma=safe_int(getattr(self.device, "lowGamma", 0)),
            midGamma=safe_int(getattr(self.device, "midGamma", 0)),
            rawValue=safe_int(getattr(self.device, "rawValue", 0)),
            poorSignal=safe_int(getattr(self.device, "poorSignal", 0)),
            blinkStrength=safe_int(getattr(self.device, "blinkStrength", 0)),
            timestamp=time.time(),
        )

    def collect_rule_window(self, mode_name: str) -> WindowResult:
        result = self._collect_window(mode_name, include_features=False)
        return WindowResult(
            mode_name=result.mode_name,
            attention_count=result.attention_count,
            meditation_count=result.meditation_count,
            blink_count=result.blink_count,
            poor_signal=result.poor_signal,
            valid=result.valid,
            disconnected=result.disconnected,
            reason=result.reason,
            samples=result.samples,
        )

    def collect_feature_window(self, mode_name: str) -> FeatureWindowResult:
        return self._collect_window(mode_name, include_features=True)

    def _collect_window(self, mode_name: str, *, include_features: bool) -> FeatureWindowResult:
        result = FeatureWindowResult(mode_name=mode_name)
        start_time = time.time()
        index = 0
        while index < self.window_size:
            snapshot = self.read_snapshot()
            result.samples.append(snapshot)
            result.poor_signal = snapshot.poorSignal

            if snapshot.poorSignal == 200:
                result.valid = False
                result.disconnected = True
                result.reason = "poorSignal=200"
                break
            if (
                snapshot.poorSignal >= self.poor_signal_threshold
                or snapshot.attention == 0
                or snapshot.meditation == 0
            ):
                result.valid = False
                if snapshot.attention == 0 or snapshot.meditation == 0:
                    result.reason = f"脑电波未读取：attention={snapshot.attention} meditation={snapshot.meditation}"
                else:
                    result.reason = f"poorSignal={snapshot.poorSignal}"
                break

            if snapshot.attention > self.attention_threshold:
                result.attention_count += 1
            if snapshot.meditation > self.meditation_threshold:
                result.meditation_count += 1
            if self._count_blink(snapshot):
                result.blink_count += 1
            if include_features:
                result.feature_window.append(build_feature_vector_from_snapshot(snapshot))

            index += 1
            sleep_time = start_time + index * self.sample_interval - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
        return result

    def _count_blink(self, snapshot: EEGSnapshot) -> bool:
        if snapshot.blinkStrength <= self.blink_threshold:
            self.blink_active = False
            return False
        if self.blink_active:
            return False
        now = snapshot.timestamp
        self.blink_active = True
        if now - self.last_blink_time < self.blink_debounce_sec:
            return False
        self.last_blink_time = now
        return True

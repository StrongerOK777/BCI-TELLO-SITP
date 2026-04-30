#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain EEG controlled robotic arm.

This program reads NeuroSky / MindWave data through neuropy.py and maps
windowed EEG decisions to the servo actions used by keyboard_control.py.
The first version intentionally favors safety, dry-run testing, and simple
rule-based control. Optional model prediction is only used for rotation mode
when --model-path is provided and successfully loaded.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("brain_arm_control")

HOME_POSE = [90, 90, 90, 90, 135, 90]
SERVO_RANGES = {
    1: (0, 180),
    2: (0, 180),
    3: (0, 180),
    4: (0, 180),
    5: (0, 270),
    6: (0, 180),
}

MODE_VERTICAL = 0
MODE_ROTATION = 1
MODE_FORWARD_BACKWARD = 2
MODE_NAMES = {
    MODE_VERTICAL: "vertical",
    MODE_ROTATION: "rotation",
    MODE_FORWARD_BACKWARD: "forward_backward",
}

MODEL_LABELS = {
    0: "left",
    1: "right",
    2: "rest",
}

# Helpful defaults for the paths in the user's current project. The command
# line --neuropy-dir and --arm-lib-dir options can override these on any setup.
KNOWN_NEUROPY_DIR = Path(r"D:\XX\SITP脑机\总体文件\drone")
KNOWN_ARM_LIB_DIR = Path(r"D:\XX\SITP脑机\机械臂控制\0.py_install\Arm_Lib (Windows)")


@dataclass
class BrainArmConfig:
    mindwave_port: str
    mindwave_baud: int = 57600
    arm_port: str = "COM4"
    model_path: Optional[str] = None
    dry_run: bool = False
    angle_step: int = 5
    move_time_ms: int = 200
    attention_threshold: int = 45
    meditation_threshold: int = 55
    blink_threshold: int = 100
    poor_signal_threshold: int = 100
    decision_margin: int = 3
    window_size: int = 30
    sample_interval: float = 0.1
    blink_debounce_sec: float = 0.5
    gripper_servo_id: int = 6
    gripper_open_angle: int = 120
    gripper_close_angle: int = 60
    neuropy_dir: Optional[str] = None
    arm_lib_dir: Optional[str] = None
    log_level: str = "INFO"
    home_time_ms: int = 1000
    action_pause_sec: float = 0.2

    def validate(self) -> None:
        if self.angle_step < 1:
            LOGGER.warning("angle_step=%s is too small; clamped to 1", self.angle_step)
            self.angle_step = 1
        if self.angle_step > 10:
            LOGGER.warning("angle_step=%s is too large; clamped to 10", self.angle_step)
            self.angle_step = 10
        if self.move_time_ms < 1:
            LOGGER.warning("move_time_ms=%s is invalid; using 200", self.move_time_ms)
            self.move_time_ms = 200
        if self.window_size < 1:
            LOGGER.warning("window_size=%s is invalid; using 30", self.window_size)
            self.window_size = 30
        if self.sample_interval <= 0:
            LOGGER.warning("sample_interval=%s is invalid; using 0.1", self.sample_interval)
            self.sample_interval = 0.1
        if self.blink_debounce_sec < 0:
            LOGGER.warning("blink_debounce_sec=%s is invalid; using 0.5", self.blink_debounce_sec)
            self.blink_debounce_sec = 0.5
        if self.gripper_servo_id not in SERVO_RANGES:
            raise ValueError("gripper_servo_id must be one of 1, 2, 3, 4, 5, 6")
        self.gripper_open_angle = clamp_servo_angle(
            self.gripper_servo_id, self.gripper_open_angle
        )
        self.gripper_close_angle = clamp_servo_angle(
            self.gripper_servo_id, self.gripper_close_angle
        )


@dataclass
class EEGSnapshot:
    attention: int
    meditation: int
    delta: int
    theta: int
    lowAlpha: int
    highAlpha: int
    lowBeta: int
    highBeta: int
    lowGamma: int
    midGamma: int
    rawValue: int
    poorSignal: int
    blinkStrength: int
    timestamp: float


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


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def clamp_servo_angle(servo_id: int, angle: int) -> int:
    min_angle, max_angle = SERVO_RANGES[servo_id]
    return clamp(int(angle), min_angle, max_angle)


def add_sys_path(path: Optional[str]) -> None:
    if not path:
        return
    resolved = Path(path).expanduser()
    if resolved.exists():
        value = str(resolved)
        if value not in sys.path:
            sys.path.insert(0, value)


def configure_dependency_paths(config: BrainArmConfig) -> None:
    script_dir = Path(__file__).resolve().parent
    add_sys_path(str(script_dir))
    add_sys_path(config.neuropy_dir)
    add_sys_path(config.arm_lib_dir)
    add_sys_path(os.getenv("NEUROPY_DIR"))
    add_sys_path(os.getenv("ARM_LIB_DIR"))
    add_sys_path(str(KNOWN_NEUROPY_DIR))
    add_sys_path(str(KNOWN_ARM_LIB_DIR))


def import_attr(module_name: str, attr_name: str, path_option: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import {module_name}. Add its folder with {path_option} "
            "or set the corresponding environment variable."
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(f"{module_name}.py does not provide {attr_name}") from exc


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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


def combine_feature_window(feature_window: Sequence[Sequence[float]]) -> List[float]:
    if not feature_window:
        return []
    rows = [list(row) for row in feature_window]
    row_count = len(rows)
    column_count = len(rows[0])
    means: List[float] = []
    stds: List[float] = []
    for column_index in range(column_count):
        values = [row[column_index] for row in rows]
        mean_value = sum(values) / row_count
        variance = sum((value - mean_value) ** 2 for value in values) / row_count
        means.append(mean_value)
        stds.append(math.sqrt(variance))
    return means + stds


class BrainSignalReader:
    def __init__(self, config: BrainArmConfig) -> None:
        self.config = config
        self.device: Any = None
        self.running = False
        self.last_blink_time = 0.0
        self.blink_active = False

    def start(self) -> None:
        NeuroSkyPy = import_attr("neuropy", "NeuroSkyPy", "--neuropy-dir")
        LOGGER.info(
            "Connecting MindWave: port=%s baud=%s",
            self.config.mindwave_port,
            self.config.mindwave_baud,
        )
        self.device = NeuroSkyPy(self.config.mindwave_port, self.config.mindwave_baud)
        self.device.start()
        self.running = True
        LOGGER.info("MindWave reader started")

    def stop(self) -> None:
        self.running = False
        if self.device is None:
            return
        try:
            self.device.stop()
            LOGGER.info("MindWave reader stopped")
        except Exception as exc:
            LOGGER.warning("Failed to stop MindWave reader cleanly: %s", exc)

    def read_snapshot(self) -> EEGSnapshot:
        if self.device is None:
            raise RuntimeError("MindWave reader has not been started")
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

    def _collect_window(self, mode_name: str, include_features: bool) -> FeatureWindowResult:
        result = FeatureWindowResult(mode_name=mode_name)
        start_time = time.time()

        for index in range(self.config.window_size):
            snapshot = self.read_snapshot()
            result.samples.append(snapshot)
            result.poor_signal = snapshot.poorSignal

            if snapshot.poorSignal == 200:
                result.valid = False
                result.disconnected = True
                result.reason = "poorSignal=200"
                break
            if snapshot.poorSignal >= self.config.poor_signal_threshold:
                result.valid = False
                result.reason = f"poorSignal={snapshot.poorSignal}"
                break

            if snapshot.attention > self.config.attention_threshold:
                result.attention_count += 1
            if snapshot.meditation > self.config.meditation_threshold:
                result.meditation_count += 1
            if self._count_blink(snapshot):
                result.blink_count += 1
            if include_features:
                result.feature_window.append(build_feature_vector_from_snapshot(snapshot))

            target_time = start_time + (index + 1) * self.config.sample_interval
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        LOGGER.info(
            "window mode=%s attention_count=%d meditation_count=%d blink_count=%d "
            "poorSignal=%d valid=%s reason=%s",
            result.mode_name,
            result.attention_count,
            result.meditation_count,
            result.blink_count,
            result.poor_signal,
            result.valid,
            result.reason or "-",
        )
        return result

    def _count_blink(self, snapshot: EEGSnapshot) -> bool:
        now = snapshot.timestamp
        if snapshot.blinkStrength > self.config.blink_threshold:
            if self.blink_active:
                return False
            if now - self.last_blink_time < self.config.blink_debounce_sec:
                self.blink_active = True
                return False
            self.last_blink_time = now
            self.blink_active = True
            return True

        self.blink_active = False
        return False


class ArmController:
    def __init__(self, config: BrainArmConfig) -> None:
        self.config = config
        self.arm: Any = None
        self.current_angles: Dict[int, int] = {
            1: 90,
            2: 90,
            3: 90,
            4: 90,
            5: 135,
            6: 90,
        }
        self.gripper_is_open = False

        if self.config.dry_run:
            LOGGER.info("Arm controller initialized in dry-run mode")
            return

        Arm_Device = import_attr("Arm_Lib", "Arm_Device", "--arm-lib-dir")
        LOGGER.info("Connecting arm: port=%s", self.config.arm_port)
        try:
            self.arm = Arm_Device(self.config.arm_port)
            LOGGER.info("Arm connected")
        except Exception as exc:
            raise RuntimeError(f"Failed to connect arm on {self.config.arm_port}: {exc}") from exc

    def home(self) -> None:
        pose = HOME_POSE[:]
        if self.config.dry_run:
            LOGGER.info("[DRY-RUN] home -> %s", pose)
        else:
            self._require_arm()
            self.arm.Arm_serial_servo_write6(*pose, self.config.home_time_ms)
            LOGGER.info("home -> %s", pose)
        self._sync_pose(pose)

    def safe_stop(self) -> None:
        LOGGER.info("safe_stop -> hold current position %s", self.angles_as_list())

    def base_left(self) -> None:
        self._step_servo(1, -1, "base_left")

    def base_right(self) -> None:
        self._step_servo(1, 1, "base_right")

    def arm_up(self) -> None:
        self._step_servo(3, 1, "arm_up")

    def arm_down(self) -> None:
        self._step_servo(3, -1, "arm_down")

    def arm_forward(self) -> None:
        self._step_servo(2, 1, "arm_forward")

    def arm_backward(self) -> None:
        self._step_servo(2, -1, "arm_backward")

    def joint4_decrease(self) -> None:
        self._step_servo(4, -1, "joint4_decrease")

    def joint4_increase(self) -> None:
        self._step_servo(4, 1, "joint4_increase")

    def grip_open(self) -> None:
        angle = clamp_servo_angle(self.config.gripper_servo_id, self.config.gripper_open_angle)
        self.write_servo_angle(
            self.config.gripper_servo_id,
            angle,
            self.config.move_time_ms,
            "grip_open",
        )
        self.gripper_is_open = True

    def grip_close(self) -> None:
        angle = clamp_servo_angle(self.config.gripper_servo_id, self.config.gripper_close_angle)
        self.write_servo_angle(
            self.config.gripper_servo_id,
            angle,
            self.config.move_time_ms,
            "grip_close",
        )
        self.gripper_is_open = False

    def toggle_gripper(self) -> None:
        if self.gripper_is_open:
            self.grip_close()
        else:
            self.grip_open()

    def execute_action(self, action_name: str) -> None:
        actions = {
            "home": self.home,
            "safe_stop": self.safe_stop,
            "base_left": self.base_left,
            "base_right": self.base_right,
            "arm_up": self.arm_up,
            "arm_down": self.arm_down,
            "arm_forward": self.arm_forward,
            "arm_backward": self.arm_backward,
            "joint4_decrease": self.joint4_decrease,
            "joint4_increase": self.joint4_increase,
            "grip_open": self.grip_open,
            "grip_close": self.grip_close,
            "toggle_gripper": self.toggle_gripper,
        }
        action = actions.get(action_name)
        if action is None:
            LOGGER.warning("Unknown arm action: %s", action_name)
            return
        action()

    def close(self) -> None:
        if self.arm is None:
            return
        serial_obj = getattr(self.arm, "ser", None)
        if serial_obj is None:
            return
        try:
            serial_obj.close()
            LOGGER.info("Arm serial port closed")
        except Exception as exc:
            LOGGER.warning("Failed to close arm serial port: %s", exc)

    def write_servo_angle(
        self,
        servo_id: int,
        angle: int,
        move_time_ms: int,
        action_name: str,
    ) -> None:
        clamped_angle = clamp_servo_angle(servo_id, angle)
        old_angle = self.current_angles.get(servo_id, HOME_POSE[servo_id - 1])
        if self.config.dry_run:
            LOGGER.info(
                "[DRY-RUN] %s -> servo=%d angle=%d old=%d",
                action_name,
                servo_id,
                clamped_angle,
                old_angle,
            )
        else:
            self._require_arm()
            self.arm.Arm_serial_servo_write(servo_id, clamped_angle, move_time_ms)
            LOGGER.info(
                "%s -> servo=%d angle=%d old=%d",
                action_name,
                servo_id,
                clamped_angle,
                old_angle,
            )
        self.current_angles[servo_id] = clamped_angle

    def angles_as_list(self) -> List[int]:
        return [self.current_angles[index] for index in range(1, 7)]

    def _step_servo(self, servo_id: int, direction: int, action_name: str) -> None:
        current = self.current_angles[servo_id]
        next_angle = current + direction * self.config.angle_step
        self.write_servo_angle(servo_id, next_angle, self.config.move_time_ms, action_name)

    def _sync_pose(self, pose: Sequence[int]) -> None:
        for servo_id, angle in enumerate(pose, start=1):
            self.current_angles[servo_id] = clamp_servo_angle(servo_id, int(angle))
        self.gripper_is_open = (
            self.current_angles[self.config.gripper_servo_id]
            >= self.config.gripper_open_angle
        )

    def _require_arm(self) -> None:
        if self.arm is None:
            raise RuntimeError("Arm is not connected")


class OptionalEEGModel:
    def __init__(self, model_path: Optional[str], autoload: bool = True) -> None:
        self.model_path = model_path
        self.model: Any = None
        self.torch: Any = None
        self.feature_mean: Any = None
        self.feature_std: Any = None
        self.enabled = False
        self.load_attempted = False

        if autoload:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self.load_attempted:
            return
        self.load_attempted = True
        if not self.model_path:
            LOGGER.info("No --model-path provided; using rule control")
            return
        self.load(self.model_path)

    def load(self, model_path: str) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            LOGGER.warning("Model file does not exist: %s. Falling back to rules.", path)
            return

        try:
            import torch
        except Exception as exc:
            LOGGER.warning("PyTorch is not available: %s. Falling back to rules.", exc)
            return

        try:
            checkpoint = torch.load(str(path), map_location="cpu")
            state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
            if state_dict is None:
                state_dict = checkpoint
            if not isinstance(state_dict, dict):
                raise RuntimeError("checkpoint does not contain a state dict")
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {
                    key.replace("module.", "", 1): value
                    for key, value in state_dict.items()
                }

            input_dim = None
            if isinstance(checkpoint, dict):
                input_dim = checkpoint.get("input_dim")
            if input_dim is None and "fc1.weight" in state_dict:
                input_dim = int(state_dict["fc1.weight"].shape[1])
            if input_dim is None:
                input_dim = 28

            model = self._build_model(torch, int(input_dim))
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.torch = torch
            if isinstance(checkpoint, dict):
                self.feature_mean = checkpoint.get("feature_mean")
                self.feature_std = checkpoint.get("feature_std")
            self.enabled = True
            LOGGER.info("Model loaded: %s input_dim=%s", path, input_dim)
        except Exception as exc:
            LOGGER.warning("Model load failed: %s. Falling back to rules.", exc)
            self.enabled = False
            self.model = None
            self.torch = None

    def predict(self, feature_window: Sequence[Sequence[float]]) -> Optional[str]:
        if not self.enabled or self.model is None or self.torch is None:
            return None
        combined = combine_feature_window(feature_window)
        if not combined:
            LOGGER.warning("Model window is empty; falling back to rules")
            return None

        try:
            torch = self.torch
            input_tensor = torch.tensor(combined, dtype=torch.float32)
            if self.feature_mean is not None and self.feature_std is not None:
                feature_mean = torch.as_tensor(self.feature_mean, dtype=torch.float32)
                feature_std = torch.as_tensor(self.feature_std, dtype=torch.float32)
                if feature_mean.numel() == input_tensor.numel() and feature_std.numel() == input_tensor.numel():
                    input_tensor = (input_tensor - feature_mean) / (feature_std + 1e-6)
                else:
                    LOGGER.warning(
                        "Model normalization shape mismatch; using raw combined features"
                    )
            with torch.no_grad():
                output = self.model(input_tensor.unsqueeze(0))
                predicted_class = int(torch.argmax(output, dim=1).item())
            label = MODEL_LABELS.get(predicted_class, "rest")
            LOGGER.info("model prediction -> class=%d label=%s", predicted_class, label)
            return label
        except Exception as exc:
            LOGGER.warning("Model prediction failed: %s. Falling back to rules.", exc)
            return None

    @staticmethod
    def _build_model(torch: Any, input_dim: int) -> Any:
        nn = torch.nn

        class FinalUnifiedModel(nn.Module):
            def __init__(self, model_input_dim: int) -> None:
                super().__init__()
                self.fc1 = nn.Linear(model_input_dim, 384)
                self.bn1 = nn.BatchNorm1d(384)
                self.dropout1 = nn.Dropout(0.12)
                self.fc2 = nn.Linear(384, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.dropout2 = nn.Dropout(0.15)
                self.fc3 = nn.Linear(256, 256)
                self.bn3 = nn.BatchNorm1d(256)
                self.dropout3 = nn.Dropout(0.15)
                self.fc4 = nn.Linear(256, 128)
                self.bn4 = nn.BatchNorm1d(128)
                self.dropout4 = nn.Dropout(0.18)
                self.fc5 = nn.Linear(128, 3)

            def forward(self, x: Any) -> Any:
                x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
                x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
                identity = x
                x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
                x = x + identity
                x = self.dropout4(torch.relu(self.bn4(self.fc4(x))))
                return self.fc5(x)

        return FinalUnifiedModel(input_dim)


class BrainArmController:
    def __init__(
        self,
        config: BrainArmConfig,
        reader: BrainSignalReader,
        arm: ArmController,
        eeg_model: OptionalEEGModel,
    ) -> None:
        self.config = config
        self.reader = reader
        self.arm = arm
        self.eeg_model = eeg_model
        self.mode = MODE_VERTICAL
        self.running = False

    def run(self) -> None:
        self.running = True
        LOGGER.info("Starting brain-arm control with config=%s", asdict(self.config))
        try:
            self.arm.home()
            self.eeg_model.ensure_loaded()
            self.reader.start()
            while self.running:
                mode_name = MODE_NAMES[self.mode]
                LOGGER.info("current mode=%d %s", self.mode, mode_name)
                if self.mode == MODE_ROTATION and self.eeg_model.enabled:
                    result = self.reader.collect_feature_window(mode_name)
                else:
                    result = self.reader.collect_rule_window(mode_name)
                self.handle_window(result)
                time.sleep(self.config.action_pause_sec)
        except KeyboardInterrupt:
            LOGGER.info("Ctrl+C received")
        except Exception:
            LOGGER.exception("Fatal error in main loop")
            raise
        finally:
            self.shutdown()

    def handle_window(self, result: WindowResult) -> None:
        if not result.valid:
            self.handle_invalid_window(result)
            return

        if result.blink_count >= 2:
            old_mode = self.mode
            self.mode = (self.mode + 1) % 3
            LOGGER.info(
                "decision=mode switch %s -> %s",
                MODE_NAMES[old_mode],
                MODE_NAMES[self.mode],
            )
            return

        if result.blink_count == 1:
            LOGGER.info("decision=toggle_gripper")
            self.arm.toggle_gripper()
            return

        if self.mode == MODE_VERTICAL:
            self.handle_vertical_mode(result)
        elif self.mode == MODE_ROTATION:
            self.handle_rotation_mode(result)
        elif self.mode == MODE_FORWARD_BACKWARD:
            self.handle_forward_backward_mode(result)

    def handle_invalid_window(self, result: WindowResult) -> None:
        if result.disconnected:
            LOGGER.warning("%s; returning arm home and waiting for recovery", result.reason)
            try:
                self.arm.home()
            except Exception as exc:
                LOGGER.warning("Failed to home arm after poorSignal=200: %s", exc)
            self.wait_for_signal_recovery()
            return

        LOGGER.warning("%s; no action for this window", result.reason)
        self.arm.safe_stop()

    def handle_vertical_mode(self, result: WindowResult) -> None:
        decision = self.compare_attention_meditation(result)
        if decision == "attention":
            LOGGER.info("decision=arm_up")
            self.arm.arm_up()
        elif decision == "meditation":
            LOGGER.info("decision=arm_down")
            self.arm.arm_down()
        else:
            LOGGER.info("decision=rest")

    def handle_rotation_mode(self, result: WindowResult) -> None:
        prediction: Optional[str] = None
        if self.eeg_model.enabled and isinstance(result, FeatureWindowResult):
            prediction = self.eeg_model.predict(result.feature_window)

        if prediction is not None:
            if prediction == "left":
                LOGGER.info("decision=base_left model=left")
                self.arm.base_left()
            elif prediction == "right":
                LOGGER.info("decision=base_right model=right")
                self.arm.base_right()
            else:
                LOGGER.info("decision=rest model=rest")
            return

        decision = self.compare_attention_meditation(result)
        if decision == "attention":
            LOGGER.info("decision=base_right rule=attention")
            self.arm.base_right()
        elif decision == "meditation":
            LOGGER.info("decision=base_left rule=meditation")
            self.arm.base_left()
        else:
            LOGGER.info("decision=rest")

    def handle_forward_backward_mode(self, result: WindowResult) -> None:
        decision = self.compare_attention_meditation(result)
        if decision == "attention":
            LOGGER.info("decision=arm_forward")
            self.arm.arm_forward()
        elif decision == "meditation":
            LOGGER.info("decision=arm_backward")
            self.arm.arm_backward()
        else:
            LOGGER.info("decision=rest")

    def compare_attention_meditation(self, result: WindowResult) -> str:
        if result.attention_count >= result.meditation_count + self.config.decision_margin:
            return "attention"
        if result.meditation_count >= result.attention_count + self.config.decision_margin:
            return "meditation"
        return "rest"

    def wait_for_signal_recovery(self) -> None:
        LOGGER.info(
            "Waiting for poorSignal to fall below %d",
            self.config.poor_signal_threshold,
        )
        while self.running:
            try:
                snapshot = self.reader.read_snapshot()
            except Exception as exc:
                LOGGER.warning("Could not read signal while waiting: %s", exc)
                time.sleep(0.5)
                continue

            if snapshot.poorSignal < self.config.poor_signal_threshold:
                LOGGER.info("Signal recovered: poorSignal=%d", snapshot.poorSignal)
                return
            self.arm.safe_stop()
            time.sleep(0.5)

    def shutdown(self) -> None:
        self.running = False
        LOGGER.info("Shutting down")
        try:
            self.reader.stop()
        except Exception as exc:
            LOGGER.warning("Reader cleanup failed: %s", exc)
        try:
            self.arm.home()
        except Exception as exc:
            LOGGER.warning("Arm home during shutdown failed: %s", exc)
        try:
            self.arm.close()
        except Exception as exc:
            LOGGER.warning("Arm cleanup failed: %s", exc)
        LOGGER.info("Shutdown complete")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a robotic arm with NeuroSky / MindWave EEG signals."
    )
    parser.add_argument("--mindwave-port", default=os.getenv("MINDWAVE_PORT", "COM6"))
    parser.add_argument("--mindwave-baud", type=int, default=57600)
    parser.add_argument("--arm-port", default=os.getenv("ARM_PORT", "COM4"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--move-time-ms", type=int, default=200)
    parser.add_argument("--attention-threshold", type=int, default=45)
    parser.add_argument("--meditation-threshold", type=int, default=55)
    parser.add_argument("--blink-threshold", type=int, default=100)
    parser.add_argument("--poor-signal-threshold", type=int, default=100)
    parser.add_argument("--decision-margin", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--blink-debounce-sec", type=float, default=0.5)
    parser.add_argument("--gripper-servo-id", type=int, default=6)
    parser.add_argument("--gripper-open-angle", type=int, default=120)
    parser.add_argument("--gripper-close-angle", type=int, default=60)
    parser.add_argument("--neuropy-dir", default=None)
    parser.add_argument("--arm-lib-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def config_from_args(args: argparse.Namespace) -> BrainArmConfig:
    return BrainArmConfig(
        mindwave_port=args.mindwave_port,
        mindwave_baud=args.mindwave_baud,
        arm_port=args.arm_port,
        model_path=args.model_path,
        dry_run=args.dry_run,
        angle_step=args.angle_step,
        move_time_ms=args.move_time_ms,
        attention_threshold=args.attention_threshold,
        meditation_threshold=args.meditation_threshold,
        blink_threshold=args.blink_threshold,
        poor_signal_threshold=args.poor_signal_threshold,
        decision_margin=args.decision_margin,
        window_size=args.window_size,
        sample_interval=args.sample_interval,
        blink_debounce_sec=args.blink_debounce_sec,
        gripper_servo_id=args.gripper_servo_id,
        gripper_open_angle=args.gripper_open_angle,
        gripper_close_angle=args.gripper_close_angle,
        neuropy_dir=args.neuropy_dir,
        arm_lib_dir=args.arm_lib_dir,
        log_level=args.log_level,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    try:
        config = config_from_args(args)
        config.validate()
        configure_dependency_paths(config)
        reader = BrainSignalReader(config)
        arm = ArmController(config)
        eeg_model = OptionalEEGModel(config.model_path, autoload=False)
        controller = BrainArmController(config, reader, arm, eeg_model)
        controller.run()
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

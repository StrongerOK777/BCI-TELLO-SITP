from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from bin.eeg import BrainSignalReader, EEGSnapshot, FeatureWindowResult, WindowResult, build_feature_vector
from bin.keyboard import KeyboardReader
from bin.models import combine_feature_window
from bin.transport import JsonHttpClient


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT_DIR / relative_path
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


car_brain = load_module("car_brain_control", "MI-CarControl/brain_control.py")
drone_brain = load_module("drone_brain_control", "MI-DroneControl/brain_control.py")
drone_hardware = load_module("drone_hardware_test", "MI-DroneControl/drone_hardware.py")


class FakeReader:
    def __init__(self, results):
        self.results = list(results)

    def collect_rule_window(self, _mode_name):
        return self.results.pop(0)

    def collect_feature_window(self, _mode_name):
        return self.results.pop(0)

    def start(self):
        pass

    def stop(self):
        pass


class FakePredictor:
    def __init__(self, predictions=None, error=None):
        self.predictions = list(predictions or [])
        self.error = error

    def predict_window(self, _feature_window):
        if self.error is not None:
            raise self.error
        return self.predictions.pop(0)


class FakeSender:
    def __init__(self):
        self.signals = []

    def send_signal(self, signal):
        self.signals.append(signal)
        return {"signal": signal}


class FakeKeyboardBackend:
    def __init__(self, keys):
        self.keys = list(keys)
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def read_key(self):
        return self.keys.pop(0) if self.keys else None

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        return FakeResponse({"ok": True})


class ModularCoreTests(unittest.TestCase):
    def test_build_feature_vector_adds_derived_features(self):
        features = build_feature_vector(10, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        self.assertEqual(len(features), 14)
        self.assertEqual(features[:11], [10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertAlmostEqual(features[11], 11 / 2)
        self.assertAlmostEqual(features[12], 7 / 2)
        self.assertAlmostEqual(features[13], 11 / (7 + 1e-6))

    def test_combine_feature_window_returns_mean_and_std(self):
        combined = combine_feature_window([[1, 2], [3, 4], [5, 6]])

        np.testing.assert_allclose(combined[:2], np.array([3, 4], dtype=np.float32))
        np.testing.assert_allclose(combined[2:], np.array([1.6329932, 1.6329932], dtype=np.float32))

    def test_blink_count_uses_debounce_and_active_state(self):
        reader = BrainSignalReader(blink_threshold=100, blink_debounce_sec=0.5)

        self.assertTrue(reader._count_blink(EEGSnapshot(blinkStrength=120, timestamp=1.0)))
        self.assertFalse(reader._count_blink(EEGSnapshot(blinkStrength=130, timestamp=1.1)))
        self.assertFalse(reader._count_blink(EEGSnapshot(blinkStrength=0, timestamp=1.2)))
        self.assertFalse(reader._count_blink(EEGSnapshot(blinkStrength=120, timestamp=1.3)))
        self.assertFalse(reader._count_blink(EEGSnapshot(blinkStrength=0, timestamp=1.4)))
        self.assertTrue(reader._count_blink(EEGSnapshot(blinkStrength=120, timestamp=1.6)))

    def test_keyboard_reader_accepts_replaceable_backend(self):
        backend = FakeKeyboardBackend(["i"])
        with KeyboardReader(backend) as reader:
            self.assertEqual(reader.read_key(), "i")
        self.assertTrue(backend.started)
        self.assertTrue(backend.closed)

    def test_json_http_client_accepts_replaceable_session(self):
        session = FakeSession()
        client = JsonHttpClient("http://device.local/", timeout=2.5, session=session)

        self.assertEqual(client.post_json("/signal", {"value": 1}), {"ok": True})
        self.assertEqual(session.calls, [("http://device.local/signal", {"value": 1}, 2.5)])


class CarBrainControlTests(unittest.TestCase):
    def test_car_state_machine_switches_modes_and_sends_expected_signals(self):
        sender = FakeSender()
        results = [
            WindowResult("前后", attention_count=30, meditation_count=20),
            WindowResult("前后", blink_count=2),
            FeatureWindowResult("转向", feature_window=[[1.0] * 14 for _ in range(30)]),
            FeatureWindowResult("转向", blink_count=2),
            WindowResult("前后", attention_count=20, meditation_count=30),
        ]
        controller = car_brain.CarBrainController(
            car_brain.CarBrainConfig(),
            FakeReader(results),
            FakePredictor([0]),
            sender,
        )

        controller.step()
        controller.step()
        self.assertEqual(controller.mode, car_brain.MODE_TURNING)
        controller.step()
        controller.step()
        self.assertEqual(controller.mode, car_brain.MODE_FORWARD_BACKWARD)
        controller.step()

        self.assertEqual(sender.signals, ["前进", "左转", "后退"])

    def test_car_invalid_window_and_model_error_do_not_send_new_actions(self):
        sender = FakeSender()
        results = [
            WindowResult("前后", valid=False, poor_signal=200, reason="poorSignal=200"),
            WindowResult("前后", blink_count=2),
            FeatureWindowResult("转向", feature_window=[[1.0] * 14 for _ in range(30)]),
        ]
        controller = car_brain.CarBrainController(
            car_brain.CarBrainConfig(),
            FakeReader(results),
            FakePredictor(error=FileNotFoundError("missing model")),
            sender,
        )

        controller.step()
        controller.step()
        controller.step()

        self.assertEqual(sender.signals, [])


class DroneControlTests(unittest.TestCase):
    def test_simulated_drone_state_machine(self):
        results = [
            WindowResult("升降", attention_count=30, meditation_count=0, blink_count=0),
            WindowResult("升降", attention_count=0, meditation_count=0, blink_count=2),
            FeatureWindowResult("转弯", blink_count=0, feature_window=[[1.0] * 14 for _ in range(30)]),
            FeatureWindowResult("转弯", blink_count=2, feature_window=[[1.0] * 14 for _ in range(30)]),
            WindowResult("前后", attention_count=30, meditation_count=30, blink_count=0),
        ]
        drone = drone_hardware.SimulatedDroneController()
        controller = drone_brain.MIDroneController(
            drone_brain.MIDroneConfig(action_pause_sec=0),
            FakeReader(results),
            drone,
            FakePredictor([1]),
        )

        controller.step()
        controller.step()
        self.assertEqual(controller.mode, drone_brain.MODE_TURNING)
        controller.step()
        controller.step()
        self.assertEqual(controller.mode, drone_brain.MODE_FORWARD_BACKWARD)
        controller.step()

        self.assertEqual(drone.actions, ["takeoff", "right", "forward"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bin.decoding import FocusSteer, VoteWindow, gate_by_confidence
from bin.eeg import BrainSignalReader, EEGSnapshot, FeatureWindowResult, WindowResult, build_feature_vector
from bin.keyboard import KeyboardReader
from bin.models import FinalUnifiedModel, ModelPredictor, combine_feature_window
from bin.training import (
    RecordedTrial,
    feature_stats,
    load_trials,
    significance_threshold,
    trial_grouped_split,
    trial_windows,
)
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
# Load arm_hardware under its real name first so brain_control's
# ``from arm_hardware import ...`` reuses this exact module object.
arm_hardware = load_module("arm_hardware", "MI-DOFBOT/arm_hardware.py")
arm_brain = load_module("arm_brain_control", "MI-DOFBOT/brain_control.py")


def focus_window(mode_name, attention, *, blink_count=0, samples=30):
    """A rule window whose per-sample attention is constant (for FocusSteer)."""

    return WindowResult(
        mode_name,
        blink_count=blink_count,
        samples=[EEGSnapshot(attention=attention, meditation=50) for _ in range(samples)],
    )


def feature_line(timestamp, attention=50):
    return f"{timestamp:.2f},{attention}|50|1000|1000|500|500|500|500|200|100|0|0"


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
    def __init__(self, predictions=None, error=None, probas=None):
        self.predictions = list(predictions or [])
        self.probas = list(probas or [])
        self.error = error

    def predict_window(self, _feature_window):
        if self.error is not None:
            raise self.error
        return self.predictions.pop(0)

    def predict_proba(self, _feature_window):
        if self.error is not None:
            raise self.error
        return self.probas.pop(0)


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

    def test_gate_by_confidence_rejects_low_and_empty(self):
        labels = ["left", "right", "rest"]
        self.assertEqual(gate_by_confidence([0.8, 0.1, 0.1], labels, 0.5).intent, "left")
        self.assertEqual(gate_by_confidence([0.4, 0.35, 0.25], labels, 0.5).intent, "rest")
        self.assertEqual(gate_by_confidence([], labels, 0.5).intent, "rest")

    def test_vote_window_requires_min_votes_and_ignores_lone_flip(self):
        vote = VoteWindow(size=3, min_votes=2, idle_intent="rest")

        self.assertEqual(vote.push("up"), "rest")   # 1 vote < min_votes
        self.assertEqual(vote.push("up"), "up")     # 2 votes -> commit
        self.assertEqual(vote.push("down"), "up")   # up=2 still beats lone down
        self.assertEqual(vote.push("down"), "down")  # buffer [up,down,down]

    def test_predict_proba_returns_distribution(self):
        predictor = ModelPredictor(None, autoload=False)
        predictor.model = FinalUnifiedModel(28)
        predictor.model.eval()

        probs = predictor.predict_proba([[1.0] * 14 for _ in range(30)])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertEqual(predictor.predict_proba([]), [])


    def test_load_trials_splits_on_gap_session_reset_and_max_rows(self):
        times = [0.0, 0.1, 0.2, 3.5, 3.6, 1.0, 1.1, 1.2, 1.3]  # gap, then a new session
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "left.txt"
            path.write_text("\n".join(feature_line(t) for t in times) + "\nbad line\n")
            trials = load_trials(path, 0, max_rows=3)

        self.assertEqual([len(t.rows) for t in trials], [3, 2, 3, 1])
        self.assertEqual([t.start_time for t in trials], [0.0, 3.5, 1.0, 1.3])
        self.assertTrue(all(t.label == 0 and t.rows.shape[1] == 14 for t in trials))

    def test_trial_grouped_split_keeps_trials_whole_and_tests_every_class(self):
        trials = [
            RecordedTrial(label, np.full((30, 14), float(label * 100 + i), dtype=np.float32), float(i))
            for label in range(3)
            for i in range(10)
        ]
        train, val, test = trial_grouped_split(trials, seed=0)

        ids = lambda items: {id(t) for t in items}  # noqa: E731
        self.assertEqual(len(ids(train) | ids(val) | ids(test)), len(trials))
        self.assertFalse(ids(train) & ids(test) or ids(train) & ids(val) or ids(val) & ids(test))
        self.assertEqual(sorted({t.label for t in test}), [0, 1, 2])
        self.assertEqual(sorted({t.label for t in val}), [0, 1, 2])

    def test_trial_windows_never_cross_trials(self):
        first = RecordedTrial(0, np.zeros((40, 14), dtype=np.float32), 0.0)
        second = RecordedTrial(1, np.ones((30, 14), dtype=np.float32), 5.0)
        too_short = RecordedTrial(2, np.ones((10, 14), dtype=np.float32), 9.0)

        samples = trial_windows([first, second, too_short], window_size=30, stride=5)

        self.assertEqual([label for _, label in samples], [0, 0, 0, 1])
        for features, label in samples:
            np.testing.assert_allclose(features[:14], float(label))  # pure per-trial mean
            np.testing.assert_allclose(features[14:], 0.0)  # no mixing -> zero std

    def test_significance_threshold_matches_exact_binomial_tail(self):
        self.assertAlmostEqual(significance_threshold(45, 2), 29 / 45)
        self.assertAlmostEqual(significance_threshold(160, 2), 91 / 160)
        self.assertAlmostEqual(significance_threshold(120, 3), 50 / 120)
        self.assertAlmostEqual(significance_threshold(5, 3), 4 / 5)
        self.assertEqual(significance_threshold(0, 3), 1.0)

    def test_feature_stats_use_raw_training_vectors(self):
        samples = [(np.array([1000.0, 2.0]), 0), (np.array([3000.0, 4.0]), 1)]

        mean, std = feature_stats(samples)

        np.testing.assert_allclose(mean, [2000.0, 3.0])
        np.testing.assert_allclose(std, [1000.0, 1.0])

    def test_focus_steer_blink_flips_and_focus_executes(self):
        steer = FocusSteer(("left", "right"), focus_threshold=55, min_focus_count=20)

        decisions = [
            steer.decide([70] * 30),           # focused -> current direction
            steer.decide([70] * 30, 1),        # single blink -> flip, no move
            steer.decide([70] * 30),           # focused -> new direction
            steer.decide([70] * 19 + [40] * 11),  # 19 < 20 focused samples -> idle
            steer.decide([]),                  # no samples -> idle
        ]

        self.assertEqual([d.intent for d in decisions], ["left", "rest", "right", "rest", "rest"])
        self.assertEqual(steer.direction, "right")


class CarBrainControlTests(unittest.TestCase):
    def test_car_forward_backward_stops_only_when_both_counts_are_low(self):
        sender = FakeSender()
        results = [
            WindowResult("前后", attention_count=25, meditation_count=10),
            WindowResult("前后", attention_count=10, meditation_count=10),
        ]
        controller = car_brain.CarBrainController(
            car_brain.CarBrainConfig(min_decision_count=20),
            FakeReader(results),
            FakePredictor(),
            sender,
        )

        controller.step()
        controller.step()

        self.assertEqual(sender.signals, ["前进", "停止"])

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
            car_brain.CarBrainConfig(turning_source="model"),
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
            car_brain.CarBrainConfig(turning_source="model"),
            FakeReader(results),
            FakePredictor(error=FileNotFoundError("missing model")),
            sender,
        )

        controller.step()
        controller.step()
        controller.step()

        self.assertEqual(sender.signals, [])


    def test_car_focus_turning_blink_selects_and_focus_executes(self):
        sender = FakeSender()
        results = [
            WindowResult("前后", blink_count=2),       # -> 转向
            focus_window("转向", 70),                  # 左转
            focus_window("转向", 70, blink_count=1),   # 换方向 -> 右，不动作
            focus_window("转向", 70),                  # 右转
            focus_window("转向", 40),                  # 专注不足 -> 停止
            focus_window("转向", 70, blink_count=2),   # -> 前后
        ]
        controller = car_brain.CarBrainController(
            car_brain.CarBrainConfig(), FakeReader(results), FakePredictor(), sender
        )

        for _ in results:
            controller.step()

        self.assertEqual(sender.signals, ["左转", "右转", "停止"])
        self.assertEqual(controller.mode, car_brain.MODE_FORWARD_BACKWARD)


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
            drone_brain.MIDroneConfig(action_pause_sec=0, turning_source="model"),
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


    def test_drone_focus_turning_blink_selects_and_focus_executes(self):
        results = [
            focus_window("转弯", 70),                  # left
            focus_window("转弯", 70, blink_count=1),   # flip -> right, no move
            focus_window("转弯", 70),                  # right
            focus_window("转弯", 40),                  # hover
            focus_window("转弯", 40, blink_count=2),   # -> 前后
        ]
        drone = drone_hardware.SimulatedDroneController()
        controller = drone_brain.MIDroneController(
            drone_brain.MIDroneConfig(action_pause_sec=0), FakeReader(results), drone, FakePredictor()
        )
        controller.mode = drone_brain.MODE_TURNING

        for _ in results:
            controller.step()

        self.assertEqual(drone.actions, ["left", "right"])
        self.assertEqual(controller.mode, drone_brain.MODE_FORWARD_BACKWARD)


class ArmBrainControlTests(unittest.TestCase):
    def _config(self, **overrides):
        params = dict(action_pause_sec=0, vote_window=1, vote_min=1)
        params.update(overrides)
        return arm_brain.ArmBrainConfig(**params)

    def _controller(self, results, arm, predictor=None, **config_overrides):
        return arm_brain.ArmBrainController(
            self._config(**config_overrides),
            FakeReader(results),
            arm,
            predictor or FakePredictor(),
        )

    def test_state_machine_cycles_modes_and_executes_actions(self):
        arm = arm_hardware.SimulatedArmController()
        results = [
            WindowResult("升降", attention_count=30, meditation_count=0),  # arm_up
            WindowResult("升降", blink_count=2),                            # -> 转弯
            FeatureWindowResult("转弯", feature_window=[[1.0] * 14 for _ in range(30)]),  # left
            FeatureWindowResult("转弯", blink_count=2),                     # -> 前后
            WindowResult("前后", attention_count=0, meditation_count=30),   # arm_backward
        ]
        controller = self._controller(
            results, arm, FakePredictor(probas=[[0.9, 0.05, 0.05]]), turning_source="model"
        )

        controller.step()
        controller.step()
        self.assertEqual(controller.mode, arm_brain.MODE_ROTATION)
        controller.step()
        controller.step()
        self.assertEqual(controller.mode, arm_brain.MODE_FORWARD_BACKWARD)
        controller.step()

        self.assertEqual(arm.actions, ["arm_up", "base_left", "arm_backward"])

    def test_single_blink_toggles_gripper(self):
        arm = arm_hardware.SimulatedArmController()
        results = [WindowResult("升降", blink_count=1), WindowResult("升降", blink_count=1)]
        controller = self._controller(results, arm)

        controller.step()
        controller.step()

        self.assertEqual(arm.actions, ["grip_open", "grip_close"])

    def test_vote_window_suppresses_single_noisy_window(self):
        arm = arm_hardware.SimulatedArmController()
        results = [
            WindowResult("升降", attention_count=30, meditation_count=0),  # up
            WindowResult("升降", attention_count=30, meditation_count=0),  # up
            WindowResult("升降", attention_count=0, meditation_count=30),  # lone down
        ]
        controller = self._controller(results, arm, vote_window=3, vote_min=2)

        controller.step()
        controller.step()
        controller.step()

        # first window is below the vote threshold, the lone "down" never wins.
        self.assertEqual(arm.actions, ["arm_up", "arm_up"])

    def test_rotation_confidence_gate_blocks_low_confidence(self):
        arm = arm_hardware.SimulatedArmController()
        results = [
            FeatureWindowResult("转弯", feature_window=[[1.0] * 14 for _ in range(30)]),
            FeatureWindowResult("转弯", feature_window=[[1.0] * 14 for _ in range(30)]),
        ]
        controller = self._controller(
            results, arm, FakePredictor(probas=[[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]), turning_source="model"
        )
        controller.mode = arm_brain.MODE_ROTATION

        controller.step()
        controller.step()

        self.assertEqual(arm.actions, ["base_left"])

    def test_rotation_focus_steer_blink_flips_direction_not_gripper(self):
        arm = arm_hardware.SimulatedArmController()
        results = [
            focus_window("转弯", 70),                  # base_left
            focus_window("转弯", 70, blink_count=1),   # flip -> right (no grip)
            focus_window("转弯", 70),                  # base_right
            focus_window("转弯", 40),                  # rest
            focus_window("转弯", 40, blink_count=2),   # -> 前后
            WindowResult("前后", blink_count=1),       # outside rotation: gripper
        ]
        controller = self._controller(results, arm)
        controller.mode = arm_brain.MODE_ROTATION

        for _ in results:
            controller.step()

        self.assertEqual(arm.actions, ["base_left", "base_right", "grip_open"])
        self.assertEqual(controller.mode, arm_brain.MODE_FORWARD_BACKWARD)

    def test_disconnect_homes_and_resets_mode(self):
        arm = arm_hardware.SimulatedArmController()
        results = [
            FeatureWindowResult(
                "转弯", valid=False, disconnected=True, poor_signal=200, reason="poorSignal=200"
            )
        ]
        controller = self._controller(results, arm)
        controller.mode = arm_brain.MODE_ROTATION

        controller.step()

        self.assertEqual(controller.mode, arm_brain.MODE_VERTICAL)
        self.assertIn("home", arm.actions)

    def test_invalid_window_without_disconnect_does_nothing(self):
        arm = arm_hardware.SimulatedArmController()
        results = [WindowResult("升降", valid=False, poor_signal=50, reason="poorSignal=50")]
        controller = self._controller(results, arm)

        controller.step()

        self.assertEqual(arm.actions, [])

    def test_run_connects_homes_and_closes(self):
        arm = arm_hardware.SimulatedArmController()
        results = [WindowResult("升降", attention_count=0, meditation_count=0)]
        controller = self._controller(results, arm)

        controller.run(max_windows=1)

        self.assertEqual(arm.actions[0], "connect")
        self.assertIn("home", arm.actions)
        self.assertEqual(arm.actions[-1], "close")


if __name__ == "__main__":
    unittest.main()

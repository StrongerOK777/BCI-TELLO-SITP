from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bin.decoding import FocusSteer, VoteWindow, gate_by_confidence
from bin.evaluation import (
    CueTrialResult,
    build_cue_schedule,
    format_summary_markdown,
    run_cue_trial,
    summarize,
    wolpaw_itr,
)
from bin import control
from bin.eeg import (
    BrainSignalReader,
    EEGSnapshot,
    FeatureWindowResult,
    SimulatedBrainSignalReader,
    WindowResult,
    build_feature_vector,
)
from bin.hub import ControlHub, DeviceOption
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


class FakeClock:
    def __init__(self, step_sec=3.0):
        self.now = 0.0
        self.step_sec = step_sec

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.step_sec


def scripted_trial(cue, moves, *, timeout, valid=None):
    """Run run_cue_trial where window i emits moves[i] (None = no movement)."""

    clock = FakeClock()
    pending = list(moves)
    valid = list(valid or [])
    emitted = []

    def step():
        clock.tick()
        emitted.append(pending.pop(0) if pending else None)
        return valid.pop(0) if valid else True

    def poll():
        return emitted.pop() if emitted else None

    return run_cue_trial(1, cue, step=step, poll_move=poll, timeout_sec=timeout, clock=clock)


class CueEvaluationTests(unittest.TestCase):
    def test_schedule_is_balanced_and_avoids_long_runs(self):
        schedule = build_cue_schedule(10, seed=3, max_run=2)

        self.assertEqual(sorted(schedule), sorted(["left", "right", "rest"] * 10))
        self.assertTrue(all(len(set(schedule[i : i + 3])) > 1 for i in range(len(schedule) - 2)))
        self.assertEqual(schedule, build_cue_schedule(10, seed=3, max_run=2))

    def test_move_cue_outcomes_and_latency(self):
        hit = scripted_trial("left", [None, "left"], timeout=15)
        wrong = scripted_trial("left", ["right"], timeout=15)
        timeout = scripted_trial("right", [None] * 10, timeout=15, valid=[True, False, True, True, True])

        self.assertEqual((hit.outcome, hit.response, hit.latency_sec, hit.windows), ("correct", "left", 6.0, 2))
        self.assertEqual((wrong.outcome, wrong.latency_sec), ("wrong", 3.0))
        self.assertEqual((timeout.outcome, timeout.latency_sec, timeout.windows), ("timeout", None, 5))
        self.assertEqual(timeout.invalid_windows, 1)

    def test_rest_cue_must_hold_still_for_the_whole_period(self):
        held = scripted_trial("rest", [None, None, None], timeout=9)
        moved = scripted_trial("rest", [None, "right"], timeout=9)

        self.assertEqual((held.outcome, held.windows, held.duration_sec), ("correct", 3, 9.0))
        self.assertEqual((moved.outcome, moved.response), ("false_move", "right"))

    def test_movement_emitted_before_the_cue_is_ignored(self):
        stale = ["left"]
        clock = FakeClock()

        def step():
            clock.tick()

        result = run_cue_trial(
            1, "rest", step=step, poll_move=lambda: stale.pop() if stale else None, timeout_sec=6, clock=clock
        )

        self.assertEqual(result.outcome, "correct")

    def test_wolpaw_itr_reference_values(self):
        self.assertAlmostEqual(wolpaw_itr(2, 1.0, 60.0), 1.0)
        self.assertEqual(wolpaw_itr(3, 1 / 3, 6.0), 0.0)
        # 3 classes, 90% accuracy, 6 s per selection
        bits = 1.58496 + 0.9 * -0.15200 + 0.1 * -4.32193
        self.assertAlmostEqual(wolpaw_itr(3, 0.9, 6.0), bits * 10, places=3)

    def test_summary_counts_confusion_and_markdown(self):
        results = [
            CueTrialResult(1, "left", "correct", "left", 4.0, 4.0, 2),
            CueTrialResult(2, "left", "wrong", "right", 3.0, 3.0, 1),
            CueTrialResult(3, "right", "correct", "right", 6.0, 6.0, 2),
            CueTrialResult(4, "right", "timeout", None, None, 15.0, 5, invalid_windows=1),
            CueTrialResult(5, "rest", "correct", None, None, 9.0, 3),
            CueTrialResult(6, "rest", "false_move", "left", 3.0, 3.0, 1),
        ]

        summary = summarize(results)

        self.assertEqual((summary["n_trials"], summary["correct"]), (6, 3))
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["per_cue"]["left"]["mean_latency_sec"], 4.0)
        self.assertEqual(summary["per_cue"]["rest"]["outcomes"], {"correct": 1, "false_move": 1})
        self.assertEqual(summary["confusion"]["right"], {"left": 0, "right": 1, "none": 1})
        self.assertEqual(summary["mean_move_latency_sec"], 5.0)
        self.assertEqual((summary["invalid_windows"], summary["total_windows"]), (1, 14))
        self.assertFalse(summary["significant"])
        markdown = format_summary_markdown(summary)
        self.assertIn("| 左转 | 2 | 1 | 50.0% |", markdown)
        self.assertIn("误动作 1", markdown)


FEATURES = [[1.0] * 14 for _ in range(30)]


def rule(mode_name, attention=0, meditation=0, blinks=0):
    return WindowResult(mode_name, attention_count=attention, meditation_count=meditation, blink_count=blinks)


def car_session(results, *, sender=None, predictor=None, **config):
    sender = sender or FakeSender()
    session = car_brain.build_controller(
        car_brain.CarBrainConfig(**config), reader=FakeReader(results), sender=sender, predictor=predictor
    )
    return session, sender


def drone_session(results, *, predictor=None, **config):
    drone = drone_hardware.SimulatedDroneController()
    session = drone_brain.build_controller(
        drone_brain.MIDroneConfig(**config), reader=FakeReader(results), drone=drone, predictor=predictor
    )
    return session, drone


def arm_session(results, *, predictor=None, **config):
    config.setdefault("vote_window", 1)
    config.setdefault("vote_min", 1)
    arm = arm_hardware.SimulatedArmController()
    session = arm_brain.build_controller(
        arm_brain.ArmBrainConfig(**config), reader=FakeReader(results), arm=arm, predictor=predictor
    )
    return session, arm


def run_steps(session, count):
    return [session.step() for _ in range(count)]


class UnifiedSessionTests(unittest.TestCase):
    def test_modes_follow_device_capabilities_in_fixed_order(self):
        car, _ = car_session([])
        drone, _ = drone_session([])
        arm, _ = arm_session([])

        self.assertEqual(car.modes, (control.MODE_FORWARD_BACKWARD, control.MODE_TURNING))
        self.assertEqual(drone.modes, control.MODE_ORDER)
        self.assertEqual(arm.modes, control.MODE_ORDER)
        self.assertEqual((car.mode, drone.mode, arm.mode), ("forward_backward", "vertical", "vertical"))

    def test_emergency_halts_until_resume_and_resets_mode(self):
        session, sender = car_session([rule("前后", 30, 0), rule("前后", 30, 0)])
        session.mode = control.MODE_TURNING

        result = session.emergency_stop()
        halted = session.step()
        session.resume()
        resumed = session.step()

        self.assertTrue(result.executed)
        self.assertEqual(halted.event, "halted")
        self.assertEqual(sender.signals, ["停止", "前进"])  # emergency stop, then the resumed window
        self.assertEqual(resumed.mode, control.MODE_FORWARD_BACKWARD)

    def test_manual_commands_respect_capabilities(self):
        session, sender = car_session([])

        refused = session.execute_manual("up")
        moved = session.execute_manual("left")
        no_aux = session.execute_manual("aux")

        self.assertFalse(refused.executed)
        self.assertTrue(moved.executed)
        self.assertFalse(no_aux.executed)
        self.assertEqual(sender.signals, ["左转"])

    def test_apply_config_changes_focus_threshold_live(self):
        session, sender = car_session([focus_window("转弯", 60), focus_window("转弯", 60)])
        session.mode = control.MODE_TURNING

        session.step()                      # 60 >= 55 -> 左转
        session.apply_config(focus_threshold=65)
        session.step()                      # 60 < 65 -> 停止

        self.assertEqual(sender.signals, ["左转", "停止"])

    def test_listeners_receive_every_report(self):
        reports = []
        session, _ = car_session([rule("前后", 30, 0), rule("前后", blinks=2)])
        session.listeners.append(reports.append)

        run_steps(session, 2)

        self.assertEqual([r.event for r in reports], ["decision", "mode_switch"])
        self.assertEqual(reports[0].intent, "forward")
        self.assertIn("前后 → 转弯", reports[1].message)

    def test_simulated_reader_feeds_sample_listener(self):
        samples = []
        reader = SimulatedBrainSignalReader(window_size=5, sample_interval=0.001, sample_listener=samples.append)
        reader.headset.set(attention=90, meditation=10)
        reader.start()

        window = reader.collect_rule_window("测试")

        self.assertTrue(window.valid)
        self.assertEqual(window.attention_count, 5)
        self.assertEqual(len(samples), 5)
        reader.headset.set(poor_signal=200)
        self.assertTrue(reader.collect_rule_window("测试").disconnected)

    def test_simulated_queued_double_blink_lands_in_one_window(self):
        reader = SimulatedBrainSignalReader(window_size=40, sample_interval=0.05)  # 2 s window
        reader.headset.set(attention=60, meditation=60)
        reader.start()
        reader.headset.queue_blinks(2)

        self.assertEqual(reader.collect_rule_window("测试").blink_count, 2)
        self.assertEqual(reader.collect_rule_window("测试").blink_count, 0)


class CarBrainControlTests(unittest.TestCase):
    def test_car_rule_mode_uses_dead_band_and_margin(self):
        session, sender = car_session([
            rule("前后", 25, 10),   # attention wins -> 前进
            rule("前后", 10, 10),   # both below 20 -> 停止
            rule("前后", 22, 21),   # difference < margin 3 -> 停止
            rule("前后", 10, 25),   # meditation wins -> 后退
        ])

        run_steps(session, 4)

        self.assertEqual(sender.signals, ["前进", "停止", "停止", "后退"])

    def test_car_cycles_modes_and_steers_with_focus(self):
        session, sender = car_session([
            rule("前后", 30, 20),                     # 前进
            rule("前后", blinks=2),                   # -> 转弯 (停止 on switch)
            focus_window("转弯", 70),                 # 左转
            focus_window("转弯", 70, blink_count=1),  # flip -> right, no command
            focus_window("转弯", 70),                 # 右转
            focus_window("转弯", 40),                 # not focused -> 停止
            focus_window("转弯", 40, blink_count=2),  # -> 前后 (停止 on switch)
        ])

        run_steps(session, 7)

        self.assertEqual(sender.signals, ["前进", "停止", "左转", "右转", "停止", "停止"])
        self.assertEqual(session.mode, control.MODE_FORWARD_BACKWARD)

    def test_car_invalid_window_warns_and_disconnect_stops(self):
        session, sender = car_session([
            WindowResult("转弯", valid=False, poor_signal=50, reason="poorSignal=50"),
            WindowResult("转弯", valid=False, disconnected=True, poor_signal=200, reason="poorSignal=200"),
        ])
        session.mode = control.MODE_TURNING

        invalid, disconnected = run_steps(session, 2)

        self.assertEqual((invalid.event, disconnected.event), ("invalid", "disconnect"))
        self.assertEqual(sender.signals, ["停止"])
        self.assertEqual(session.mode, control.MODE_FORWARD_BACKWARD)

    def test_car_model_turning_is_confidence_gated_and_errors_rest(self):
        gated, gated_sender = car_session(
            [FeatureWindowResult("转弯", feature_window=FEATURES)] * 2,
            predictor=FakePredictor(probas=[[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]),
            turning_source="model",
        )
        broken, broken_sender = car_session(
            [FeatureWindowResult("转弯", feature_window=FEATURES)],
            predictor=FakePredictor(error=FileNotFoundError("missing model")),
            turning_source="model",
        )
        gated.mode = broken.mode = control.MODE_TURNING

        run_steps(gated, 2)
        report = broken.step()

        self.assertEqual(gated_sender.signals, ["左转", "停止"])
        self.assertEqual(broken_sender.signals, ["停止"])
        self.assertIn("模型不可用", report.message)


class DroneControlTests(unittest.TestCase):
    def test_drone_full_cycle_takeoff_moves_and_landing(self):
        session, drone = drone_session([
            rule("升降", 30, 0),                      # takeoff
            rule("升降", 30, 0),                      # up
            rule("升降", 0, 30),                      # down (back to take-off height)
            rule("升降", blinks=2),                   # -> 前后
            rule("前后", 30, 0),                      # forward
            rule("前后", blinks=2),                   # -> 转弯
            focus_window("转弯", 70),                 # left
            focus_window("转弯", 70, blink_count=1),  # flip -> right
            focus_window("转弯", 70),                 # right
            focus_window("转弯", 40, blink_count=2),  # -> 升降 (order is 升降 -> 前后 -> 转弯)
            rule("升降", 0, 30),                      # land
        ])

        run_steps(session, 11)

        self.assertEqual(drone.actions, ["takeoff", "up", "down", "forward", "left", "right", "land"])
        self.assertEqual(session.device.height, 0)

    def test_drone_refuses_moves_until_airborne(self):
        session, drone = drone_session([rule("前后", 30, 0), focus_window("转弯", 70)])
        session.mode = control.MODE_FORWARD_BACKWARD

        forward = session.step()
        session.mode = control.MODE_TURNING
        turn = session.step()

        self.assertEqual(drone.actions, [])
        self.assertFalse(forward.executed or turn.executed)
        self.assertIn("未起飞", forward.message)

    def test_drone_disconnect_lands_and_resets_mode(self):
        session, drone = drone_session([
            rule("升降", 30, 0),
            rule("升降", blinks=2),
            WindowResult("前后", valid=False, disconnected=True, poor_signal=200, reason="poorSignal=200"),
        ])

        run_steps(session, 3)

        self.assertEqual(drone.actions, ["takeoff", "land"])
        self.assertEqual(session.mode, control.MODE_VERTICAL)

    def test_drone_run_lands_on_exit(self):
        session, drone = drone_session([rule("升降", 30, 0)])

        session.run(max_windows=1)

        self.assertEqual(drone.actions, ["connect", "takeoff", "land", "close"])


class ArmBrainControlTests(unittest.TestCase):
    def test_arm_cycles_modes_gripper_and_steering(self):
        session, arm = arm_session([
            rule("升降", 30, 0),                      # arm_up
            rule("升降", blinks=2),                   # -> 前后
            rule("前后", 0, 30),                      # arm_backward
            rule("前后", blinks=2),                   # -> 转弯
            focus_window("转弯", 70),                 # base_left
            focus_window("转弯", 70, blink_count=1),  # flip direction (gripper untouched)
            focus_window("转弯", 70),                 # base_right
            focus_window("转弯", 40, blink_count=2),  # -> 升降
            rule("升降", blinks=1),                   # aux: gripper opens
            rule("升降", blinks=1),                   # aux: gripper closes
        ])

        run_steps(session, 10)

        self.assertEqual(
            arm.actions, ["arm_up", "arm_backward", "base_left", "base_right", "grip_open", "grip_close"]
        )

    def test_arm_default_vote_window_suppresses_single_noisy_window(self):
        session, arm = arm_session(
            [rule("升降", 30, 0), rule("升降", 30, 0), rule("升降", 0, 30)], vote_window=3, vote_min=2
        )

        run_steps(session, 3)

        # first window is below the vote threshold, the lone "down" never wins.
        self.assertEqual(arm.actions, ["arm_up", "arm_up"])
        self.assertEqual((arm_brain.ArmBrainConfig().vote_window, arm_brain.ArmBrainConfig().vote_min), (3, 2))

    def test_arm_model_rotation_confidence_gate(self):
        session, arm = arm_session(
            [FeatureWindowResult("转弯", feature_window=FEATURES)] * 2,
            predictor=FakePredictor(probas=[[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]]),
            turning_source="model",
        )
        session.mode = control.MODE_TURNING

        run_steps(session, 2)

        self.assertEqual(arm.actions, ["base_left"])

    def test_arm_disconnect_homes_and_resets_mode(self):
        session, arm = arm_session(
            [WindowResult("转弯", valid=False, disconnected=True, poor_signal=200, reason="poorSignal=200")]
        )
        session.mode = control.MODE_TURNING

        session.step()

        self.assertEqual(session.mode, control.MODE_VERTICAL)
        self.assertEqual(arm.actions, ["home"])

    def test_arm_invalid_window_without_disconnect_does_nothing(self):
        session, arm = arm_session([WindowResult("升降", valid=False, poor_signal=50, reason="poorSignal=50")])

        session.step()

        self.assertEqual(arm.actions, [])

    def test_arm_run_connects_homes_and_closes(self):
        session, arm = arm_session([rule("升降")])

        session.run(max_windows=1)

        self.assertEqual(arm.actions[:2], ["connect", "home"])
        self.assertEqual(arm.actions[-2:], ["home", "close"])


def hub_with_devices(**kwargs):
    def car_factory(real):
        settings = car_brain.CarBrainConfig(dry_run=not real, window_size=3)
        return settings, car_brain.build_device(settings)

    def arm_factory(real):
        settings = arm_brain.ArmBrainConfig(simulated=not real, window_size=3, vote_window=1, vote_min=1,
                                            min_decision_count=2)
        return settings, arm_brain.build_device(settings)

    options = [DeviceOption("car", "小车", car_factory), DeviceOption("arm", "机械臂", arm_factory)]
    return ControlHub(options, sim_sample_interval=0.002, **kwargs)


def wait_for(predicate, timeout=3.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class ControlHubTests(unittest.TestCase):
    def test_real_hardware_requires_allow_real(self):
        hub = hub_with_devices()
        with self.assertRaises(PermissionError):
            hub.connect("car", real_hardware=True)
        self.assertFalse(hub.state()["connected"])

    def test_manual_commands_only_while_brain_paused(self):
        hub = hub_with_devices()
        hub.connect("arm")
        try:
            self.assertTrue(hub.command("aux")["executed"])
            hub.set_sim_headset(attention=90, meditation=10)
            hub.start_brain()
            self.assertTrue(wait_for(lambda: hub.last_report and hub.last_report["executed"]))
            with self.assertRaises(RuntimeError):
                hub.command("up")
            hub.stop_brain()
            self.assertFalse(hub.running)
            self.assertIn("arm_up", hub.session.device.arm.actions)
        finally:
            hub.shutdown()

    def test_emergency_works_while_running_and_events_are_published(self):
        hub = hub_with_devices()
        events = hub.subscribe()
        hub.connect("arm")
        try:
            hub.set_sim_headset(attention=90, meditation=10)
            hub.start_brain()
            self.assertTrue(wait_for(lambda: hub.last_report is not None))
            result = hub.emergency()
            self.assertTrue(wait_for(lambda: not hub.running))
            self.assertTrue(result["executed"])
            self.assertTrue(hub.state()["halted"])
            self.assertIn("home", hub.session.device.arm.actions)
            seen = set()
            while not events.empty():
                seen.add(events.get_nowait()[0])
            self.assertTrue({"state", "sample", "report", "action", "log"} <= seen)
        finally:
            hub.shutdown()

    def test_config_updates_apply_to_session_and_reconnect(self):
        hub = hub_with_devices()
        hub.connect("car")
        try:
            hub.update_config(focus_threshold=70)
            self.assertEqual(hub.session.config.focus_threshold, 70)
            hub.connect("arm")
            self.assertEqual(hub.session.config.focus_threshold, 70)
            with self.assertRaises(KeyError):
                hub.update_config(window_size=5)
        finally:
            hub.shutdown()


if __name__ == "__main__":
    unittest.main()

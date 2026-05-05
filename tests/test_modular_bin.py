import unittest

import numpy as np

from bin.eeg import BrainSignalReader, EEGSnapshot, FeatureWindowResult, WindowResult, build_feature_vector
from bin.hardware import SimulatedDroneController
from bin.mi_drone_control import MIDroneConfig, MIDroneController, MODE_FORWARD_BACKWARD, MODE_TURNING
from bin.models import combine_feature_window


class FakeReader:
    def __init__(self, results):
        self.results = list(results)

    def collect_rule_window(self, mode_name):
        return self.results.pop(0)

    def collect_feature_window(self, mode_name):
        return self.results.pop(0)

    def start(self):
        pass

    def stop(self):
        pass


class FakePredictor:
    def __init__(self, predictions):
        self.predictions = list(predictions)

    def predict_window(self, feature_window):
        return self.predictions.pop(0)


class ModularBinTests(unittest.TestCase):
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

    def test_simulated_drone_state_machine(self):
        results = [
            WindowResult("升降", attention_count=30, meditation_count=0, blink_count=0),
            WindowResult("升降", attention_count=0, meditation_count=0, blink_count=2),
            FeatureWindowResult("转弯", blink_count=0, feature_window=[[1.0] * 14 for _ in range(30)]),
            FeatureWindowResult("转弯", blink_count=2, feature_window=[[1.0] * 14 for _ in range(30)]),
            WindowResult("前后", attention_count=30, meditation_count=30, blink_count=0),
        ]
        drone = SimulatedDroneController()
        controller = MIDroneController(
            MIDroneConfig(action_pause_sec=0),
            FakeReader(results),
            drone,
            FakePredictor([1]),
        )

        controller.step()
        controller.step()
        self.assertEqual(controller.mode, MODE_TURNING)
        controller.step()
        controller.step()
        self.assertEqual(controller.mode, MODE_FORWARD_BACKWARD)
        controller.step()

        self.assertEqual(drone.actions, ["takeoff", "right", "forward"])


if __name__ == "__main__":
    unittest.main()

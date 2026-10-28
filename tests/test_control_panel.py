from __future__ import annotations

import importlib.util
import json
import time
import unittest

from bin.hub import ControlHub

HAS_FLASK = importlib.util.find_spec("flask") is not None


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@unittest.skipUnless(HAS_FLASK, "flask 未安装，跳过控制面板测试")
class ControlPanelRouteTests(unittest.TestCase):
    def setUp(self):
        from control_panel.app import create_app
        from control_panel.devices import device_options

        self.hub = ControlHub(device_options(), sim_sample_interval=0.002)
        self.client = create_app(self.hub).test_client()

    def tearDown(self):
        self.hub.shutdown()

    def test_index_and_state(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        state = self.client.get("/api/state").get_json()
        self.assertEqual([d["key"] for d in state["devices"]], ["car", "drone", "arm"])
        self.assertFalse(state["allow_real"])
        self.assertFalse(state["connected"])

    def test_real_hardware_needs_flag_and_confirmation(self):
        no_confirm = self.client.post("/api/connect", json={"device": "car", "real_hardware": True})
        not_allowed = self.client.post("/api/connect", json={"device": "car", "real_hardware": True, "confirm": True})

        self.assertEqual(no_confirm.status_code, 400)
        self.assertEqual(not_allowed.status_code, 403)
        self.assertFalse(self.hub.state()["connected"])

    def test_connect_command_brain_and_emergency(self):
        state = self.client.post("/api/connect", json={"device": "drone"}).get_json()
        self.assertEqual([m["key"] for m in state["modes"]], ["vertical", "forward_backward", "turning"])

        takeoff = self.client.post("/api/command", json={"intent": "up"}).get_json()
        refused = self.client.post("/api/command", json={"intent": "fly"})
        self.assertTrue(takeoff["executed"])
        self.assertFalse(refused.get_json()["executed"])

        self.client.post("/api/sim-headset", json={"attention": 90, "meditation": 10})
        self.assertTrue(self.client.post("/api/brain/start").get_json()["running"])
        self.assertTrue(wait_for(lambda: self.hub.last_report is not None))
        self.assertEqual(self.client.post("/api/command", json={"intent": "up"}).status_code, 409)

        emergency = self.client.post("/api/emergency").get_json()
        self.assertTrue(emergency["executed"])
        self.assertTrue(wait_for(lambda: not self.hub.running))
        self.assertEqual(self.hub.session.device.height, 0)  # landed

    def test_config_validation(self):
        self.client.post("/api/connect", json={"device": "car"})

        ok = self.client.post("/api/config", json={"focus_threshold": 65})
        bad_key = self.client.post("/api/config", json={"window_size": 10})
        bad_value = self.client.post("/api/config", json={"turning_source": "magic"})

        self.assertEqual(ok.get_json()["config"]["focus_threshold"], 65)
        self.assertEqual((bad_key.status_code, bad_value.status_code), (400, 400))

    def test_stream_sends_state_first(self):
        response = self.client.get("/api/stream?once=1")
        text = response.get_data(as_text=True)

        self.assertEqual(response.mimetype, "text/event-stream")
        self.assertTrue(text.startswith("event: state\n"))
        payload = json.loads(text.split("data: ", 1)[1].split("\n", 1)[0])
        self.assertIn("devices", payload)


if __name__ == "__main__":
    unittest.main()

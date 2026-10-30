"""Diagnostics regression checks: no hardware or installed XR packages required."""
import asyncio
import importlib.util
import multiprocessing as mp
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from host.recorder_diagnostics import (
    PoseEventDiagnostics, TerminalDiagnostics, observation_status, pose_status, xr_status,
)
from host.teleop_record_dex3_tactile import KeyState, complete_observation
from host.xr_legacy import LegacyStereoXR


def report_from_child(events):
    events.update("hand", "right wrist pose: invalid homogeneous row")


class DiagnosticTests(unittest.TestCase):
    def test_key_logging_keeps_existing_transitions_even_if_logging_fails(self):
        def broken_log(message):
            raise OSError("terminal closed")
        plain, logged = KeyState(), KeyState(broken_log, record_enabled=False)
        for key in ("s", "r", "s", "s", "x", "q", "s"):
            plain.on_press(key)
            logged.on_press(key)
            self.assertEqual((plain.start, plain.stop, plain.record_toggle),
                             (logged.start, logged.stop, logged.record_toggle))
        messages = []
        keys = KeyState(messages.append, record_enabled=False)
        for key in ("s", "r", "s", "q"):
            keys.on_press(key)
        self.assertIn("ignored", messages[0])
        self.assertIn("r received", messages[1])
        self.assertIn("DISABLED", messages[2])
        self.assertIn("q received", messages[3])

    def test_accepted_age_boundaries_and_latest_rejection_are_distinct(self):
        now = 2_000_000_000
        self.assertEqual(pose_status("head", 0, "", now)[0][0], "missing")
        self.assertEqual(pose_status("head", now - 499_000_000, "", now)[0][0], "ready")
        self.assertEqual(pose_status("head", now - 500_000_000, "", now)[0][0], "stale")
        self.assertEqual(pose_status("head", now - 250_000_000, "", now, 250)[0][0], "stale")
        status, message = pose_status("hand", now, "leftState: expected dict", now)
        self.assertEqual(status[0], "ready")  # rejection does not erase last good event
        self.assertIn("latest event REJECTED", message)

    def test_log_rate_does_not_follow_continuously_changing_ages(self):
        messages, now = [], [0.0]
        logger = TerminalDiagnostics(messages.append, lambda: now[0])
        logger.emit("start", [("missing", "no hand")], repeat=True)
        for tick in range(1, 150):
            now[0] = tick / 30
            logger.emit("start", [("missing", f"no hand age={tick}")], repeat=True)
        self.assertEqual(len(messages), 1)
        now[0] = 5.0
        logger.emit("start", [("missing", "no hand")], repeat=True)
        now[0] = 5.6
        logger.emit("start", [("ready", "hand received")])
        now[0] = 50
        logger.emit("start", [("ready", "hand received age=20")])
        self.assertEqual(len(messages), 3)

    def test_validation_reason_reaches_parent_process(self):
        events = PoseEventDiagnostics()
        process = mp.Process(target=report_from_child, args=(events,))
        process.start()
        process.join(10)
        try:
            self.assertEqual(process.exitcode, 0)
            self.assertIn("right wrist", events.error("hand"))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2)

    def test_recording_reasons_match_existing_gate_including_exact_limit(self):
        frame = SimpleNamespace(bgr="frame")
        packet = SimpleNamespace(present_deg=(1, 2), torque_enabled=True)
        touch = {s: {"pressure": [[0] * 12], "age_ms": 500.0}
                 for s in ("left_ee", "right_ee")}
        self.assertTrue(complete_observation(frame, 500, (packet, 500), touch))
        self.assertTrue(all(not issues for issues, _ in observation_status(frame, 500, (packet, 500), touch)))
        packet.present_deg = None
        packet.torque_enabled = False
        touch["right_ee"] = None
        touch["left_ee"]["age_ms"] = 501
        before = complete_observation(frame, 501, (packet, 501), touch)
        messages = " | ".join(m for _, m in observation_status(frame, 501, (packet, 501), touch))
        self.assertFalse(before)
        self.assertEqual(before, complete_observation(frame, 501, (packet, 501), touch))
        for expected in ("encoder MISSING", "torque DISABLED", "packet STALE", "tactile left_ee", "tactile right_ee: NOT RECEIVED"):
            self.assertIn(expected, messages)


class FakeTeleVuer:
    def __init__(self, **kwargs):
        self.head_calls = self.hand_calls = 0
        self.process = SimpleNamespace(is_alive=lambda: False, join=lambda timeout: None)

    async def on_cam_move(self, event, session, fps=60):
        self.head_calls += 1

    async def on_hand_move(self, event, session, fps=60):
        self.hand_calls += 1


TeleVuer = FakeTeleVuer


class FakeWrapper:
    def __init__(self, **kwargs):
        self.tvuer = TeleVuer(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
class EventInstrumentationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        schemas = types.ModuleType("vuer.schemas")
        schemas.Hands = schemas.ImageBackground = object
        # No real Vuer server or robot is constructed; exercise actual callbacks.
        with patch.dict("sys.modules", {"vuer.schemas": schemas}), patch("host.xr_legacy.mp.get_start_method", return_value="fork"):
            self.xr = LegacyStereoXR(FakeWrapper, (2, 4, 3), "unused", "unused")
        self.addCleanup(self.xr.close)
        self.pose = np.eye(4).flatten(order="F").tolist()

    def hand(self, **changes):
        payload = {"left": self.pose * 25, "right": self.pose * 25,
                   "leftState": {}, "rightState": {}}
        payload.update(changes)
        asyncio.run(self.xr.tvuer.on_hand_move(SimpleNamespace(value=payload), None))

    def test_valid_data_preserves_calls_timestamps_and_tracking_gate(self):
        tv = self.xr.tvuer
        asyncio.run(tv.on_cam_move(SimpleNamespace(value={"camera": {"matrix": self.pose}}), None))
        self.hand()
        self.assertEqual((tv.head_calls, tv.hand_calls), (1, 1))
        self.assertTrue(self.xr.tracking_ready())
        before = (tv.head_received_ns.value, tv.hand_received_ns.value)
        report = xr_status(self.xr)
        self.assertTrue(all(status[0] == "ready" for status, _ in report))
        self.assertEqual(before, (tv.head_received_ns.value, tv.hand_received_ns.value))

    def test_rejections_identify_side_and_do_not_touch_accepted_timestamp(self):
        self.hand()
        tv = self.xr.tvuer
        stamp = tv.hand_received_ns.value
        for changes, expected in [
            ({"left": []}, "left hand payload"),
            ({"right": [float("nan")] * 400}, "right hand payload"),
            ({"right": [0] * 400}, "right wrist pose"),
            ({"leftState": []}, "leftState"),
            ({"rightState": None}, "rightState"),
        ]:
            with self.subTest(expected=expected):
                self.hand(**changes)
                self.assertIn(expected, tv.pose_event_diagnostics.error("hand"))
                self.assertEqual(tv.hand_received_ns.value, stamp)
                self.assertEqual(tv.hand_calls, 1)
        self.hand()
        self.assertEqual(tv.hand_calls, 2)
        self.assertEqual(tv.pose_event_diagnostics.error("hand"), "")

    def test_rejected_head_does_not_call_original_handler(self):
        tv = self.xr.tvuer
        asyncio.run(tv.on_cam_move(SimpleNamespace(value={"camera": {"matrix": [0] * 16}}), None))
        self.assertEqual(tv.head_calls, 0)
        self.assertEqual(tv.head_received_ns.value, 0)
        self.assertIn("homogeneous row", tv.pose_event_diagnostics.error("head"))
        self.assertFalse(self.xr.tracking_ready())


if __name__ == "__main__":
    unittest.main()

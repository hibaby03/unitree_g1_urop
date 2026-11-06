import asyncio
import unittest

from host import display_feed as feed_module
from host.display_feed import (DisplayFeedReceiver, eye_layers, format_report,
                               parse_display_message, relay_to_vuer)
from pc2.display_stream import DISPLAY_STRUCT, pack_display_message

LEFT = b"\xff\xd8left-jpeg\xff\xd9"
RIGHT = b"\xff\xd8right-jpeg\xff\xd9"


def message(sequence=1, left=LEFT, right=RIGHT):
    return pack_display_message(left, right, sequence, 1000 + sequence, 7, 640, 360)


class FakeImageBackground:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Session:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def upsert(self, elements, to=None):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise AssertionError("client disconnected")
        self.calls.append((elements, to))


def make_receiver():
    class Idle:  # never used: ingest() is driven directly
        pass
    receiver = DisplayFeedReceiver.__new__(DisplayFeedReceiver)
    import threading, time
    receiver._owns_socket, receiver._lock = False, threading.Lock()
    receiver._latest, receiver._taken = None, 0
    receiver.received = receiver.bytes_received = receiver.bad_messages = receiver.forwarded = 0
    receiver.started, receiver.last_received = time.monotonic(), 0.0
    return receiver


class ParseTests(unittest.TestCase):
    def test_round_trip_keeps_both_eyes_and_frame_number(self):
        frame = parse_display_message(message(9))
        self.assertEqual((frame.sequence, frame.left_jpeg, frame.right_jpeg), (9, LEFT, RIGHT))
        self.assertEqual((frame.capture_pc2_monotonic_ns, frame.zed_image_time_ns), (1009, 7))
        self.assertAlmostEqual(frame.aspect, 640 / 360)
        self.assertEqual(frame.stamp.frame_sequence, 9)

    def test_rejects_malformed_messages(self):
        good = message()
        bad = {
            "short": good[:10],
            "magic": b"XXXX" + good[4:],
            "truncated": good[:-1],
            "trailing": good + b"x",
            "not jpeg": message(left=b"nope"),
            "empty eye": DISPLAY_STRUCT.pack(b"ADSP", 1, 0, 0, 1, 1, 0, 0, 3, 640, 360) + b"\xff\xd8x",
        }
        for name, data in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_display_message(data)


class ReceiverTests(unittest.TestCase):
    def test_each_frame_is_handed_out_once_and_only_the_newest(self):
        receiver = make_receiver()
        self.assertIsNone(receiver.take_new())
        receiver.ingest(message(1))
        receiver.ingest(message(2))      # frame 1 was never taken: superseded
        self.assertEqual(receiver.take_new().sequence, 2)
        self.assertIsNone(receiver.take_new())   # no repeat without a new frame
        receiver.ingest(message(3))
        self.assertEqual(receiver.take_new().sequence, 3)

    def test_bad_messages_are_counted_and_ignored(self):
        receiver = make_receiver()
        self.assertFalse(receiver.ingest(b"garbage"))
        self.assertIsNone(receiver.take_new())
        self.assertEqual((receiver.bad_messages, receiver.received), (1, 0))


class RelayTests(unittest.TestCase):
    def test_vuer_layers_get_the_received_jpeg_bytes_unchanged(self):
        layers = eye_layers(parse_display_message(message()), FakeImageBackground,
                            screen_y=-0.1, distance=1.0)
        left, right = (layer.kwargs for layer in layers)
        self.assertIs(left["src"].__class__, bytes)
        self.assertEqual((left["src"], right["src"]), (LEFT, RIGHT))
        self.assertNotIn("data", left)
        self.assertNotIn("format", left)       # no (re-)encoding requested from Vuer
        self.assertNotIn("quality", left)
        self.assertEqual((left["layers"], right["layers"]), (1, 2))
        self.assertEqual((left["key"], right["key"]), ("zed-left", "zed-right"))
        self.assertEqual(left["position"], [0, -0.1, 0])
        self.assertEqual(left["distanceToCamera"], 1.0)

    def run_relay(self, receiver, session, frames_then_stop):
        async def go():
            task = asyncio.create_task(relay_to_vuer(
                session, receiver, FakeImageBackground, screen_y=-0.1, poll_s=0.001,
                report_s=0.02, log=lines.append))
            await frames_then_stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        lines = []
        asyncio.run(go())
        return lines

    def test_forwards_new_frames_once_and_never_repeats(self):
        receiver, session = make_receiver(), Session()

        async def script():
            receiver.ingest(message(1))
            await asyncio.sleep(0.05)       # no new frame: nothing more may be sent
            receiver.ingest(message(2))
            await asyncio.sleep(0.05)
        lines = self.run_relay(receiver, session, script)
        sent = [(c[0][0].kwargs["src"], c[1]) for c in session.calls]
        self.assertEqual(len(sent), 2)
        self.assertEqual(receiver.forwarded, 2)
        self.assertTrue(all(target == "bgChildren" for _, target in sent))
        self.assertTrue(any("forwarded to Vuer" in line and "not the Quest" in line for line in lines))

    def test_returns_when_the_client_disconnects(self):
        receiver, session = make_receiver(), Session(fail_after=0)

        async def go():
            receiver.ingest(message(1))
            await asyncio.wait_for(relay_to_vuer(session, receiver, FakeImageBackground,
                                                 screen_y=0, poll_s=0.001), 1)
        asyncio.run(go())
        self.assertEqual(receiver.forwarded, 0)

    def test_warns_when_pc2_sends_no_display_stream(self):
        receiver, session = make_receiver(), Session()
        receiver.started -= feed_module.NO_FRAME_WARNING_S + 1

        async def script():
            await asyncio.sleep(0.02)
        lines = self.run_relay(receiver, session, script)
        self.assertEqual(sum("no display frames" in line for line in lines), 1)
        self.assertEqual(session.calls, [])


class ReportTests(unittest.TestCase):
    def test_report_has_received_forwarded_and_size_separately(self):
        text = format_report({"received": 0, "bytes": 0, "forwarded": 0, "bad": 0},
                             {"received": 60, "bytes": 60 * 4096, "forwarded": 50, "bad": 0},
                             2.0, age_ms=42.0)
        for expected in ("received 30.0 fps", "forwarded to Vuer 25.0 fps", "4.0 KB/msg",
                         "0.98 Mbit/s", "image age at Host 42 ms"):
            self.assertIn(expected, text)


class ViewerArgsTests(unittest.TestCase):
    def test_relay_is_the_default_and_legacy_is_opt_in(self):
        from host import zmq_stereo_vr as viewer
        args = viewer.parse_args([])
        self.assertEqual((args.display_port, args.legacy_display), (55556, False))
        self.assertTrue(viewer.parse_args(["--legacy-display"]).legacy_display)


if __name__ == "__main__":
    unittest.main()

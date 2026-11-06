import threading
import time
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from host.display_feed import parse_display_message
from pc2 import display_stream as ds
from pc2.display_stream import (DISPLAY_HEADER_SIZE, DisplayStatsReporter,
                                DisplayStreamPublisher, pack_display_message)


def fake_encode(eye, quality):
    # "JPEG" whose payload records the eye's first pixel and the quality.
    return b"\xff\xd8" + bytes([int(eye[0, 0, 0]), quality]) + b"\xff\xd9"


def fake_resize(eye, scale):
    return eye[::int(round(1 / scale)), ::int(round(1 / scale))]


class FakeSocket:
    def __init__(self, full_after=None):
        self.sent, self.closed, self.full_after = [], False, full_after

    def send(self, data, flags=0):
        if self.full_after is not None and len(self.sent) >= self.full_after:
            raise BlockingIOError
        self.sent.append(data)

    def close(self):
        self.closed = True


def frame(left_value, right_value, height=4, half=8):
    bgr = np.zeros((height, 2 * half, 3), np.uint8)
    bgr[:, :half], bgr[:, half:] = left_value, right_value
    return bgr


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.002)
    return False


@unittest.skipIf(np is None, "needs numpy")
class DisplayStreamPublisherTests(unittest.TestCase):
    def make(self, socket=None, **kwargs):
        socket = socket or FakeSocket()
        clock = kwargs.pop("clock", None) or time.monotonic
        publisher = DisplayStreamPublisher(socket=socket, resize=fake_resize, encode=fake_encode,
                                           clock=clock, **kwargs)
        self.addCleanup(publisher.close)
        return publisher, socket

    def test_left_and_right_eye_travel_in_one_message_of_one_frame(self):
        publisher, socket = self.make(scale=0.5, jpeg_quality=60, fps=1000)
        publisher.submit(frame(11, 22), 7, 123456, 999)
        self.assertTrue(wait_for(lambda: socket.sent))
        self.assertEqual(len(socket.sent), 1)
        shown = parse_display_message(socket.sent[0])
        self.assertEqual((shown.sequence, shown.capture_pc2_monotonic_ns, shown.zed_image_time_ns),
                         (7, 123456, 999))
        self.assertEqual(shown.left_jpeg, b"\xff\xd8\x0b\x3c\xff\xd9")   # left=11, quality 60
        self.assertEqual(shown.right_jpeg, b"\xff\xd8\x16\x3c\xff\xd9")  # right=22
        self.assertEqual((shown.eye_width, shown.eye_height), (4, 2))   # half-scale of 8x4

    def test_fps_cap_skips_frames_before_encoding(self):
        now = [0.0]
        encoded = []

        def counting_encode(eye, quality):
            encoded.append(1)
            return fake_encode(eye, quality)

        socket = FakeSocket()
        publisher = DisplayStreamPublisher(socket=socket, resize=fake_resize, encode=counting_encode,
                                           clock=lambda: now[0], fps=10)
        self.addCleanup(publisher.close)
        publisher.submit(frame(1, 2), 1, 1)
        self.assertTrue(wait_for(lambda: publisher.sent == 1))
        for sequence in (2, 3):  # 0.01 s and 0.02 s later: inside the 0.1 s period
            now[0] += 0.01
            publisher.submit(frame(1, 2), sequence, sequence)
            self.assertTrue(wait_for(lambda: publisher.dropped >= sequence - 1))
        self.assertEqual((publisher.sent, len(encoded)), (1, 2))  # 2 eyes, one frame
        now[0] += 0.2
        publisher.submit(frame(1, 2), 4, 4)
        self.assertTrue(wait_for(lambda: publisher.sent == 2))
        self.assertEqual(parse_display_message(socket.sent[1]).sequence, 4)

    def test_newest_frame_replaces_unprocessed_one(self):
        gate = threading.Event()

        def blocking_encode(eye, quality):
            gate.wait(2)
            return fake_encode(eye, quality)

        socket = FakeSocket()
        publisher = DisplayStreamPublisher(socket=socket, resize=fake_resize, encode=blocking_encode, fps=1000)
        self.addCleanup(publisher.close)
        publisher.submit(frame(1, 1), 1, 1)
        self.assertTrue(wait_for(lambda: publisher.submitted == 1))
        time.sleep(0.05)               # frame 1 is now inside the encoder
        publisher.submit(frame(2, 2), 2, 2)
        publisher.submit(frame(3, 3), 3, 3)  # supersedes frame 2
        gate.set()
        self.assertTrue(wait_for(lambda: publisher.sent == 2))
        self.assertEqual([parse_display_message(m).sequence for m in socket.sent], [1, 3])
        self.assertEqual(publisher.dropped, 1)

    def test_full_zmq_queue_drops_without_blocking_or_counting_as_sent(self):
        publisher, socket = self.make(socket=FakeSocket(full_after=0), fps=1000)
        publisher.submit(frame(1, 2), 1, 1)
        self.assertTrue(wait_for(lambda: publisher.dropped == 1))
        self.assertEqual((publisher.sent, publisher.bytes_sent), (0, 0))

    def test_encode_failure_does_not_stop_the_stream(self):
        calls = []

        def flaky(eye, quality):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return fake_encode(eye, quality)

        socket = FakeSocket()
        publisher = DisplayStreamPublisher(socket=socket, resize=fake_resize, encode=flaky, fps=1000)
        self.addCleanup(publisher.close)
        publisher.submit(frame(1, 2), 1, 1)
        self.assertTrue(wait_for(lambda: publisher.errors == 1))
        publisher.submit(frame(1, 2), 2, 2)
        self.assertTrue(wait_for(lambda: publisher.sent == 1))

    def test_rejects_bad_settings(self):
        for kwargs in ({"scale": 0.05}, {"scale": 1.5}, {"jpeg_quality": 0}, {"fps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DisplayStreamPublisher(socket=FakeSocket(), **kwargs)

    def test_odd_width_is_rejected_as_an_error_not_a_crash(self):
        publisher, socket = self.make(fps=1000)
        publisher.submit(np.zeros((4, 7, 3), np.uint8), 1, 1)
        self.assertTrue(wait_for(lambda: publisher.errors == 1))


class WireFormatTests(unittest.TestCase):
    def test_pack_layout(self):
        message = pack_display_message(b"\xff\xd8L", b"\xff\xd8RR", 5, 10, None, 640, 360)
        self.assertEqual(len(message), DISPLAY_HEADER_SIZE + 3 + 4)
        shown = parse_display_message(message)
        self.assertEqual((shown.left_jpeg, shown.right_jpeg), (b"\xff\xd8L", b"\xff\xd8RR"))
        self.assertEqual(shown.zed_image_time_ns, 0)


class StatsTests(unittest.TestCase):
    def test_report_separates_capture_sent_and_sizes(self):
        before = {"submitted": 0, "sent": 0, "dropped": 0, "errors": 0, "bytes": 0, "encode_seconds": 0.0}
        after = {"submitted": 60, "sent": 20, "dropped": 40, "errors": 0, "bytes": 20 * 2048,
                 "encode_seconds": 0.1}
        text = DisplayStatsReporter.format(before, after, 2.0)
        for expected in ("captured 30.0 fps", "display sent 10.0 fps", "dropped 40",
                         "2.0 KB/msg", "0.16 Mbit/s", "encode 5.0 ms/msg"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()

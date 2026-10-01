import json
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = types.ModuleType("yaml")
    yaml.YAMLError = ValueError
    yaml.safe_load = json.load
    sys.modules["yaml"] = yaml

try:
    import cv2
except ImportError:
    cv2 = None
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

try:
    import numpy as np
except ImportError:
    np = None

from host import frame_timing as host_side
from host.frame_timing import (
    ClockEstimate,
    ClockSample,
    ClockSyncClient,
    FrameStamp,
    StampedFrame,
    StampedFrameReader,
    best_estimate,
    frame_timing_entry,
    parse_jpeg_stamp,
    probe_sample,
)
from host.teleop_record_dex3_tactile import parse_args as record_parse_args
from host.teleop_record_dex3_tactile import summarize_ages
from pc2 import frame_timing as pc2_side
from pc2.frame_timing import ClockSyncServer, stamp_jpeg


FAKE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF-body\xff\xd9"


class WireFormatTests(unittest.TestCase):
    def test_both_sides_share_the_wire_formats(self):
        for name in ("STAMP_STRUCT", "SYNC_REQUEST_STRUCT", "SYNC_REPLY_STRUCT"):
            self.assertEqual(getattr(pc2_side, name).format, getattr(host_side, name).format)
        for name in ("STAMP_MAGIC", "STAMP_VERSION", "SYNC_MAGIC", "SYNC_VERSION",
                     "DEFAULT_SYNC_PORT"):
            self.assertEqual(getattr(pc2_side, name), getattr(host_side, name))
        self.assertEqual(pc2_side.STAMP_SIZE, 28)


class JpegStampTests(unittest.TestCase):
    def test_round_trip(self):
        stamped = stamp_jpeg(FAKE_JPEG, 7, 123_456_789_000, 42)
        self.assertEqual(parse_jpeg_stamp(stamped), FrameStamp(7, 123_456_789_000, 42))
        # Original segments follow the inserted COM segment untouched.
        self.assertTrue(stamped.endswith(FAKE_JPEG[2:]))

    def test_missing_zed_time_is_none(self):
        self.assertIsNone(parse_jpeg_stamp(stamp_jpeg(FAKE_JPEG, 1, 5)).zed_image_time_ns)

    def test_sequence_wraps(self):
        self.assertEqual(parse_jpeg_stamp(stamp_jpeg(FAKE_JPEG, 2**32 + 3, 5)).frame_sequence, 3)

    def test_unstamped_or_foreign_data_is_none(self):
        self.assertIsNone(parse_jpeg_stamp(None))
        self.assertIsNone(parse_jpeg_stamp(FAKE_JPEG))
        self.assertIsNone(parse_jpeg_stamp(b"\xff\xd8\xff\xfe\x00\x05abc" + b"\x00" * 40))
        foreign = stamp_jpeg(FAKE_JPEG, 1, 5).replace(b"AFTS", b"XXXX")
        self.assertIsNone(parse_jpeg_stamp(foreign))

    def test_rejects_non_jpeg(self):
        with self.assertRaises(ValueError):
            stamp_jpeg(b"\x89PNG", 1, 5)

    @unittest.skipIf(cv2 is None or np is None or not hasattr(cv2, "imencode"),
                     "needs OpenCV")
    def test_decoders_ignore_the_stamp(self):
        image = np.random.default_rng(0).integers(0, 255, (16, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        plain = encoded.tobytes()
        stamped = stamp_jpeg(plain, 1, 5)
        decode = host_side._decode_jpeg
        np.testing.assert_array_equal(decode(stamped), decode(plain))


class ClockMathTests(unittest.TestCase):
    def test_symmetric_delay_gives_exact_offset(self):
        # PC2 clock is 1 s ahead; 2 ms each way; PC2 takes 0.1 ms to answer.
        sample = probe_sample(10_000_000, 1_012_000_000, 1_012_100_000, 14_100_000)
        self.assertEqual(sample.offset_ns, 1_000_000_000)
        self.assertEqual(sample.delay_ns, 4_000_000)

    def test_asymmetric_delay_error_is_within_half_delay(self):
        # 1 ms out, 9 ms back.
        sample = probe_sample(0, 1_001_000_000, 1_001_000_000, 10_000_000)
        self.assertLessEqual(abs(sample.offset_ns - 1_000_000_000), sample.delay_ns // 2)

    def test_best_estimate_uses_lowest_delay_fresh_sample(self):
        samples = [ClockSample(100, 50, 1), ClockSample(9_000, 70, 8), ClockSample(9_500, 60, 4)]
        self.assertEqual(best_estimate(samples, 10_000, 5_000), ClockEstimate(60, 2))
        self.assertEqual(best_estimate(samples, 10_000, 20_000), ClockEstimate(50, 0))
        self.assertIsNone(best_estimate(samples, 100_000, 5_000))
        self.assertIsNone(best_estimate([], 0, 1))

    def test_pc2_to_host(self):
        self.assertEqual(ClockEstimate(1_000, 0).pc2_to_host_ns(5_000), 4_000)


class ClockSyncUdpTests(unittest.TestCase):
    def test_loopback_offset_is_near_zero(self):
        server = ClockSyncServer("127.0.0.1", 0)
        self.addCleanup(server.close)
        client = ClockSyncClient("127.0.0.1", server.port, interval_s=0.02)
        self.addCleanup(client.close)
        deadline = time.monotonic() + 2.0
        estimate = None
        while estimate is None and time.monotonic() < deadline:
            time.sleep(0.02)
            estimate = client.estimate()
        self.assertIsNotNone(estimate)
        # Same machine, same CLOCK_MONOTONIC.
        self.assertLess(abs(estimate.offset_ns), 5_000_000)
        self.assertLess(estimate.uncertainty_ns, 5_000_000)

    def test_server_ignores_garbage(self):
        import socket
        server = ClockSyncServer("127.0.0.1", 0)
        self.addCleanup(server.close)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"nope", ("127.0.0.1", server.port))
            sock.sendto(b"XXXX" + b"\x00" * 16, ("127.0.0.1", server.port))
        deadline = time.monotonic() + 1.0
        while server.rejected < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(server.rejected, 2)

    def test_client_without_server_has_no_estimate(self):
        client = ClockSyncClient("127.0.0.1", 9, interval_s=0.02)
        self.addCleanup(client.close)
        time.sleep(0.1)
        self.assertIsNone(client.estimate())


class StampedFrameReaderTests(unittest.TestCase):
    def test_decodes_each_new_jpeg_once(self):
        first = stamp_jpeg(FAKE_JPEG, 1, 100)
        second = stamp_jpeg(FAKE_JPEG, 2, 200)
        current = {"jpg": first}
        decoded = []

        def decode(jpeg):
            decoded.append(jpeg)
            return f"bgr{len(decoded)}"

        reader = StampedFrameReader(lambda: SimpleNamespace(jpg=current["jpg"]), decode)
        self.assertEqual(reader.read(), StampedFrame("bgr1", FrameStamp(1, 100, None)))
        self.assertEqual(reader.read().bgr, "bgr1")
        current["jpg"] = second
        self.assertEqual(reader.read(), StampedFrame("bgr2", FrameStamp(2, 200, None)))
        self.assertEqual(len(decoded), 2)

    def test_no_frame_yet(self):
        reader = StampedFrameReader(lambda: SimpleNamespace(jpg=None), lambda _: self.fail())
        self.assertEqual(reader.read(), StampedFrame(None, None))

    def test_unstamped_frame_still_decodes(self):
        reader = StampedFrameReader(lambda: SimpleNamespace(jpg=FAKE_JPEG), lambda _: "bgr")
        self.assertEqual(reader.read(), StampedFrame("bgr", None))


class TimingEntryTests(unittest.TestCase):
    def test_ages_on_host_clock(self):
        clock = ClockEstimate(offset_ns=1_000_000_000, uncertainty_ns=250_000)
        frame = StampedFrame("bgr", FrameStamp(9, 1_050_000_000, None))
        neck = SimpleNamespace(present_pc2_monotonic_ns=1_080_000_000,
                               command_pc2_monotonic_ns=1_070_000_000)

        entry = frame_timing_entry(100_000_000, frame, clock, neck)

        self.assertEqual(entry["host_monotonic_ns"], 100_000_000)
        self.assertEqual(entry["image_frame_sequence"], 9)
        self.assertEqual(entry["image_host_monotonic_ns"], 50_000_000)
        self.assertEqual(entry["image_age_ms"], 50.0)
        self.assertEqual(entry["neck_present_age_ms"], 20.0)
        self.assertEqual(entry["neck_command_age_ms"], 30.0)
        self.assertEqual(entry["clock_uncertainty_ms"], 0.25)
        json.dumps(entry)

    def test_unknown_without_clock_stamp_or_encoder(self):
        neck = SimpleNamespace(present_pc2_monotonic_ns=None, command_pc2_monotonic_ns=5)
        entry = frame_timing_entry(1, StampedFrame("bgr", FrameStamp(1, 5, None)), None, neck)
        self.assertEqual(entry["image_pc2_monotonic_ns"], 5)
        for key in ("image_age_ms", "image_host_monotonic_ns", "neck_present_age_ms",
                    "neck_command_age_ms", "clock_offset_ns"):
            self.assertIsNone(entry[key])
        entry = frame_timing_entry(1, StampedFrame("bgr", None), ClockEstimate(0, 0), None)
        self.assertIsNone(entry["image_age_ms"])
        self.assertIsNone(frame_timing_entry(1, None, None)["image_frame_sequence"])


class RecorderTimingTests(unittest.TestCase):
    def test_summarize_ages(self):
        self.assertIn("unknown for all 2", summarize_ages([None, None]))
        text = summarize_ages([10.0, 20.0, 30.0, None])
        self.assertIn("median 20.0 ms", text)
        self.assertIn("max 30.0 ms", text)
        self.assertIn("unknown for 1/4", text)

    def test_clock_sync_args(self):
        args = record_parse_args([])
        self.assertEqual(args.clock_sync_port, 5007)
        self.assertFalse(args.no_clock_sync)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            record_parse_args(["--clock-sync-port", "0"])


class ZedServerStampTests(unittest.TestCase):
    @unittest.skipIf(np is None, "needs numpy")
    def test_zmq_frames_carry_the_capture_stamp(self):
        from pc2 import zed_teleimager_server as server
        from pc2.zed_stereo import ZedCaptureOptions

        class Buffer:
            def __init__(self):
                self.data = None

            def write(self, data):
                self.data = data

        class BaseCamera:
            def __init__(self, topic, shape, fps, enable_zmq, zmq_port, enable_webrtc,
                         webrtc_port, webrtc_codec):
                self._img_shape, self._fps = shape, fps
                self._enable_zmq, self._enable_webrtc = enable_zmq, enable_webrtc
                self._zmq_buffer, self._webrtc_buffer = Buffer(), Buffer()
                self._ready = SimpleNamespace(set=lambda: None)

        class FakeCapture:
            def __init__(self, _options):
                self.latest_pc2_monotonic_ns = None
                self.latest_zed_image_time_ns = None

            def open(self):
                pass

            def grab(self):
                self.latest_pc2_monotonic_ns = 777
                self.latest_zed_image_time_ns = 888
                return np.zeros((2, 4, 3), dtype=np.uint8)

        fake_cv2 = SimpleNamespace(
            imencode=lambda _ext, _img: (True, np.frombuffer(FAKE_JPEG, dtype=np.uint8)))
        with patch.object(server, "ZedStereoCapture", FakeCapture), \
                patch.object(server, "cv2", fake_cv2):
            camera_class = server.make_zed_camera_class(
                SimpleNamespace(BaseCamera=BaseCamera), ZedCaptureOptions())
            camera = camera_class("head_camera", "", [2, 4], 30, enable_zmq=True)
            camera._update_frame()
            camera._update_frame()

        self.assertEqual(parse_jpeg_stamp(camera._zmq_buffer.data), FrameStamp(2, 777, 888))


if __name__ == "__main__":
    unittest.main()

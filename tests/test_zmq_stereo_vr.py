import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

try:
    import zmq
except ImportError:
    zmq = None

from host import zmq_stereo_vr as viewer
from host.frame_timing import FrameStamp
from host.head_pose_udp import PACKET_STRUCT
from pc2.frame_timing import stamp_jpeg


FAKE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF-body\xff\xd9"
# Column-major OpenXR matrix: identity rotation, position (0.1, 1.5, -0.2).
CAMERA_MOVE_FLAT = [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0.1, 1.5, -0.2, 1]


class ArgsTests(unittest.TestCase):
    def test_defaults(self):
        args = viewer.parse_args([])
        self.assertEqual(args.img_server_ip, "192.168.123.164")
        self.assertEqual(args.zmq_port, 55555)
        self.assertIsNone(args.neck_pose_ip)

    def test_rejects_bad_values(self):
        for argv in (["--zmq-port", "0"], ["--jpeg-quality", "0"], ["--distance", "0"]):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                viewer.parse_args(argv)

    def test_resolve_tls(self):
        with tempfile.TemporaryDirectory() as tmp:
            cert, key = Path(tmp) / "c.pem", Path(tmp) / "k.pem"
            cert.write_text("x")
            key.write_text("y")
            self.assertEqual(viewer.resolve_tls(str(cert), str(key)),
                             (cert.resolve(), key.resolve()))
            with self.assertRaises(FileNotFoundError):
                viewer.resolve_tls(str(cert), str(Path(tmp) / "missing.pem"))


class HelperTests(unittest.TestCase):
    @unittest.skipIf(np is None, "needs numpy")
    def test_split_eyes_rgb(self):
        frame = np.zeros((2, 4, 3), dtype=np.uint8)
        frame[:, :2, 0] = 10   # left eye, blue channel
        frame[:, 2:, 2] = 20   # right eye, red channel
        left, right = viewer.split_eyes_rgb(frame)
        self.assertEqual(left.shape, (2, 2, 3))
        np.testing.assert_array_equal(left[..., 2], 10)
        np.testing.assert_array_equal(right[..., 0], 20)

    @unittest.skipIf(np is None, "needs numpy")
    def test_split_rejects_odd_width(self):
        with self.assertRaises(ValueError):
            viewer.split_eyes_rgb(np.zeros((2, 3, 3), dtype=np.uint8))

    def test_camera_move_matrix_is_column_major(self):
        matrix = viewer.camera_move_matrix(CAMERA_MOVE_FLAT)
        self.assertEqual([row[3] for row in matrix[:3]], [0.1, 1.5, -0.2])
        with self.assertRaises(ValueError):
            viewer.camera_move_matrix(CAMERA_MOVE_FLAT[:15])


class HeadPoseForwarderTests(unittest.TestCase):
    def test_sends_only_fresh_poses(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sink:
            sink.bind(("127.0.0.1", 0))
            sink.settimeout(0.5)
            forwarder = viewer.HeadPoseForwarder("127.0.0.1", sink.getsockname()[1], 100.0)
            self.addCleanup(forwarder.close)

            self.assertIsNone(forwarder.fresh_matrix())
            forwarder.update(viewer.camera_move_matrix(CAMERA_MOVE_FLAT))
            datagram = sink.recv(128)
            self.assertEqual(len(datagram), PACKET_STRUCT.size)

            with patch.object(viewer.time, "monotonic",
                              return_value=time.monotonic() + viewer.HEAD_POSE_MAX_AGE_S + 1):
                self.assertIsNone(forwarder.fresh_matrix())


@unittest.skipIf(zmq is None or np is None, "needs pyzmq and numpy")
class ZmqReceiverTests(unittest.TestCase):
    def test_receives_newest_frame_and_stamp(self):
        context = zmq.Context.instance()
        publisher = context.socket(zmq.PUB)
        self.addCleanup(publisher.close, 0)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        receiver = viewer.ZmqStereoReceiver("127.0.0.1", port, decode=lambda jpeg: len(jpeg))
        self.addCleanup(receiver.close)

        stamped = stamp_jpeg(FAKE_JPEG, 5, 123)
        deadline = time.monotonic() + 3.0
        while receiver.frames == 0 and time.monotonic() < deadline:
            publisher.send(stamped)  # PUB drops messages until SUB has joined
            time.sleep(0.02)
        self.assertGreater(receiver.frames, 0)
        self.assertEqual(receiver.latest(), (len(stamped), FrameStamp(5, 123, None)))

    def test_undecodable_frames_are_skipped(self):
        context = zmq.Context.instance()
        publisher = context.socket(zmq.PUB)
        self.addCleanup(publisher.close, 0)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        receiver = viewer.ZmqStereoReceiver("127.0.0.1", port, decode=lambda _jpeg: None)
        self.addCleanup(receiver.close)
        for _ in range(20):
            publisher.send(FAKE_JPEG)
            time.sleep(0.01)
        self.assertEqual(receiver.latest(), (None, None))


if __name__ == "__main__":
    unittest.main()

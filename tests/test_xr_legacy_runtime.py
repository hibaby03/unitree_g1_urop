"""Integration checks against the installed tv environment (no robot commands).

Run in conda tv. These checks start loopback ZMQ and HTTPS services only.
"""
import asyncio
import functools
import importlib.util
import os
from pathlib import Path
import socket
import signal
import ssl
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request

from host.frame_timing import StampedFrameReader
from host.xr_legacy import LegacyStereoXR
from host.zmq_stereo_vr import resolve_tls
from pc2.frame_timing import stamp_jpeg


HAS_XR = all(importlib.util.find_spec(name) is not None
             for name in ("televuer", "teleimager", "numpy", "cv2", "vuer"))


@unittest.skipUnless(HAS_XR, "requires the Host conda tv packages")
class InstalledRuntimeTests(unittest.TestCase):
    def test_teleimager_zmq_jpeg_and_timestamp_roundtrip(self):
        import cv2
        import numpy as np
        import zmq
        from teleimager import image_client as module

        context = zmq.Context()
        rep, pub = context.socket(zmq.REP), context.socket(zmq.PUB)
        request_port = rep.bind_to_random_port("tcp://127.0.0.1")
        image_port = pub.bind_to_random_port("tcp://127.0.0.1")
        config = {
            "head_camera": {"enable_zmq": True, "enable_webrtc": False,
                            "zmq_port": image_port, "image_shape": [48, 128],
                            "binocular": True, "fps": 30},
            "left_wrist_camera": {"enable_zmq": False},
            "right_wrist_camera": {"enable_zmq": False},
        }
        image = np.zeros((48, 128, 3), dtype=np.uint8)
        image[:, :64] = (0, 0, 200)
        image[:, 64:] = (200, 0, 0)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = stamp_jpeg(encoded.tobytes(), 42, 123456789)
        stop = threading.Event()

        def serve():
            while not stop.is_set():
                if rep.poll(10):
                    rep.recv()
                    rep.send_json(config)
                pub.send(jpeg)
                stop.wait(.02)

        server = threading.Thread(target=serve, daemon=True)
        client = None
        server.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                # The real client caches received config; keep this test out
                # of the user's installed camera configuration.
                with patch.object(module, "CONFIG_CLIENT_PATH", str(Path(directory) / "camera.yaml")):
                    client = module.ImageClient(host="127.0.0.1", request_port=request_port,
                                                request_bgr=False)
                reader = StampedFrameReader(client.get_head_frame)
                deadline = time.monotonic() + 3
                frame = reader.read()
                while frame.bgr is None and time.monotonic() < deadline:
                    time.sleep(.02)
                    frame = reader.read()
                self.assertIsNotNone(frame.bgr)
                self.assertEqual(frame.bgr.shape, (48, 128, 3))
                self.assertEqual(frame.stamp.frame_sequence, 42)
                self.assertEqual(frame.stamp.capture_pc2_monotonic_ns, 123456789)
                self.assertGreater(int(frame.bgr[0, 0, 2]), 180)
                self.assertGreater(int(frame.bgr[0, 100, 0]), 180)
                self.assertEqual(client.get_cam_config(), config)
        finally:
            if client is not None:
                client.close()
                client.close()  # supported by teleimager 1.6.0
            stop.set()
            server.join(2)
            rep.close(0)
            pub.close(0)
            context.term()

    def test_real_legacy_wrapper_tls_events_shared_memory_and_shutdown(self):
        import numpy as np
        from televuer import TeleVuer, TeleVuerWrapper
        from vuer import Vuer

        cert, key = resolve_tls(None, None)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        underlying = sys.modules[TeleVuer.__module__]
        xr = None
        try:
            # Select a temporary HTTPS port without editing installed sources.
            with patch.object(underlying, "Vuer", functools.partial(Vuer, port=port)):
                xr = LegacyStereoXR(TeleVuerWrapper, (48, 128, 3), cert, key)
            process = xr.tvuer.process
            self.assertTrue(process.is_alive())
            deadline = time.monotonic() + 5
            while True:
                try:
                    with urllib.request.urlopen(f"https://127.0.0.1:{port}/",
                                                context=ssl._create_unverified_context(), timeout=.5) as response:
                        self.assertEqual(response.status, 200)
                        break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.05)
            image = np.full((48, 128, 3), 42, dtype=np.uint8)
            xr.render(image)
            np.testing.assert_array_equal(xr.image, image)

            # Exercise real Hands/ImageBackground schema constructors, not
            # merely HTTP startup. The async display runs on Quest connect.
            rendered = []

            class Session:
                def upsert(self, value, **kwargs):
                    rendered.append(value)

            async def render_once():
                task = asyncio.create_task(xr.tvuer.main_image_binocular(Session()))
                await asyncio.sleep(.01)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            asyncio.run(render_once())
            self.assertEqual(len(rendered[1]), 2)
            self.assertFalse(xr.tracking_ready())
            pose = np.eye(4).flatten(order="F").tolist()
            asyncio.run(xr.tvuer.on_cam_move(SimpleNamespace(value={"camera": {"matrix": pose}}), None))
            value = {"left": pose * 25, "right": pose * 25,
                     "leftState": {}, "rightState": {}}
            asyncio.run(xr.tvuer.on_hand_move(SimpleNamespace(value=value), None))
            self.assertTrue(xr.tracking_ready())
            tele_data = xr.get_motion_state_data()
            for name in ("head_pose", "left_arm_pose", "right_arm_pose"):
                self.assertEqual(getattr(tele_data, name).shape, (4, 4))
            self.assertEqual(tele_data.left_hand_pos.shape, (25, 3))
            head_time = xr.tvuer.head_received_ns.value
            hand_time = xr.tvuer.hand_received_ns.value
            asyncio.run(xr.tvuer.on_cam_move(SimpleNamespace(value={"camera": {"matrix": [0] * 16}}), None))
            asyncio.run(xr.tvuer.on_hand_move(SimpleNamespace(value={"left": []}), None))
            self.assertEqual(xr.tvuer.head_received_ns.value, head_time)
            self.assertEqual(xr.tvuer.hand_received_ns.value, hand_time)
            xr.tvuer.head_received_ns.value -= 1_000_000_000
            self.assertFalse(xr.tracking_ready())
            os.kill(process.pid, signal.SIGINT)
            process.join(timeout=.05)
            self.assertTrue(process.is_alive())  # parent owns graceful shutdown
            xr.close()
            self.assertFalse(process.is_alive())
        finally:
            if xr is not None:
                xr.close()


if __name__ == "__main__":
    unittest.main()

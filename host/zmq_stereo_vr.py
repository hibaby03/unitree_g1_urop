#!/usr/bin/env python3
"""Show the ZED stereo frames the Host receives over ZMQ in the Quest.

Unlike --video-only / --input-mode=none of run_teleop_with_neck.py, where the
Quest plays PC2's WebRTC stream directly, this viewer displays exactly the
frames the Host recorder receives: PC2 teleimager ZMQ JPEG (55555) -> Host
decode -> Vuer ImageBackground on the left/right eye layers. Nothing else of
xr_teleoperate (TeleVuer, DDS, IK, recorder) is started, so no robot is needed.

Optionally (--neck-pose-ip) the Quest head pose is forwarded to PC2's
head_pose_receiver exactly as in --input-mode=none, so the active camera
follows the operator's head.

Open https://<host-ip>:8012/?ws=wss://<host-ip>:8012 in the Quest browser.
Do not run it together with a teleop script; both serve Vuer on port 8012.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional, Sequence

try:
    from .frame_timing import (DEFAULT_SYNC_PORT, ClockSyncClient, FrameStamp,
                               parse_jpeg_stamp)
    from .head_pose_udp import HeadPoseUdpSender, _validated_pose_matrix
except ImportError:
    from frame_timing import (  # type: ignore[no-redef]
        DEFAULT_SYNC_PORT, ClockSyncClient, FrameStamp, parse_jpeg_stamp)
    from head_pose_udp import (  # type: ignore[no-redef]
        HeadPoseUdpSender, _validated_pose_matrix)


HEAD_POSE_MAX_AGE_S = 0.25  # same freshness rule as run_teleop_with_neck.py


def positive_rate(value: str) -> float:
    parsed = float(value)
    if not 1.0 <= parsed <= 240.0:
        raise argparse.ArgumentTypeError("must be in 1..240")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the Host-received ZED ZMQ stereo stream in the Quest via Vuer.")
    parser.add_argument("--img-server-ip", default="192.168.123.164", help="PC2 address")
    parser.add_argument("--zmq-port", type=int, default=55555,
                        help="teleimager head_camera ZMQ port")
    parser.add_argument("--display-fps", type=positive_rate, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=80,
                        help="JPEG quality of the per-eye images sent to the Quest")
    parser.add_argument("--distance", type=float, default=1.0,
                        help="virtual distance of the image plane [m]")
    parser.add_argument("--cert", help="TLS certificate; default XR_TELEOP_CERT or "
                        "~/.config/xr_teleoperate/cert.pem")
    parser.add_argument("--key", help="TLS key; default XR_TELEOP_KEY or "
                        "~/.config/xr_teleoperate/key.pem")
    parser.add_argument("--neck-pose-ip", default=None,
                        help="forward the Quest head pose to PC2 (e.g. 192.168.123.164)")
    parser.add_argument("--neck-pose-port", type=int, default=5005)
    parser.add_argument("--neck-pose-rate", type=positive_rate, default=60.0)
    parser.add_argument("--clock-sync-port", type=int, default=DEFAULT_SYNC_PORT)
    parser.add_argument("--no-clock-sync", action="store_true",
                        help="do not measure the image age")
    args = parser.parse_args(argv)
    for name in ("zmq_port", "neck_pose_port", "clock_sync_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in 1..100")
    if not args.distance > 0.0:
        parser.error("--distance must be positive")
    return args


def resolve_tls(cert: Optional[str], key: Optional[str]) -> tuple[Path, Path]:
    """Same lookup order as TeleVuer: arguments, environment, user config."""

    cert = cert or os.environ.get("XR_TELEOP_CERT")
    key = key or os.environ.get("XR_TELEOP_KEY")
    config_dir = Path.home() / ".config" / "xr_teleoperate"
    cert_path = Path(cert).expanduser() if cert else config_dir / "cert.pem"
    key_path = Path(key).expanduser() if key else config_dir / "key.pem"
    for path in (cert_path, key_path):
        if not path.is_file():
            raise FileNotFoundError(f"TLS file not found: {path} (use --cert/--key)")
    return cert_path.resolve(), key_path.resolve()


# ---------------------------------------------------------------------------
# Pure helpers (unit tested without hardware)
# ---------------------------------------------------------------------------

def split_eyes_rgb(bgr: Any) -> tuple[Any, Any]:
    """Left/right RGB halves of a combined left|right BGR frame."""

    width = bgr.shape[1]
    if width % 2:
        raise ValueError(f"combined frame width must be even, got {width}")
    rgb = bgr[:, :, ::-1]
    return rgb[:, :width // 2], rgb[:, width // 2:]


def camera_move_matrix(flat: Sequence[float]) -> list[list[float]]:
    """Validated row-major 4x4 pose from a Vuer CAMERA_MOVE matrix.

    Vuer sends the OpenXR camera matrix in column-major order.
    """

    if len(flat) != 16:
        raise ValueError("camera matrix must have 16 values")
    return _validated_pose_matrix([[flat[column * 4 + row] for column in range(4)]
                                   for row in range(4)])


# ---------------------------------------------------------------------------
# Image source
# ---------------------------------------------------------------------------

class ZmqStereoReceiver:
    """Keep the newest decoded ZMQ frame (and its PC2 stamp) from teleimager."""

    def __init__(self, ip: str, port: int,
                 decode: Optional[Callable[[bytes], Any]] = None) -> None:
        import zmq

        if decode is None:
            import cv2
            import numpy as np

            def decode(jpeg: bytes) -> Any:
                return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        self._decode = decode
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        # Same options as teleimager's subscriber: only the newest frame.
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{ip}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._lock = threading.Lock()
        self._frame: Any = None
        self._stamp: Optional[FrameStamp] = None
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="zmq-stereo", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=100)):
                continue
            try:
                jpeg = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop.is_set():
                    return
                raise
            # Decode here, once per received frame, so the async display loop
            # never blocks on it.
            frame = self._decode(jpeg)
            if frame is None:
                continue
            with self._lock:
                self._frame, self._stamp = frame, parse_jpeg_stamp(jpeg)
                self.frames += 1

    def latest(self) -> tuple[Any, Optional[FrameStamp]]:
        with self._lock:
            return self._frame, self._stamp

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close()
        self._context.term()


# ---------------------------------------------------------------------------
# Head pose forwarding
# ---------------------------------------------------------------------------

class HeadPoseForwarder:
    """Send the newest CAMERA_MOVE pose to PC2 while it is fresh."""

    def __init__(self, ip: str, port: int, rate_hz: float) -> None:
        self._sender = HeadPoseUdpSender(ip, port)
        self._period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._matrix: Optional[list[list[float]]] = None
        self._received = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="head-pose-udp", daemon=True)
        self._thread.start()

    def update(self, matrix: list[list[float]]) -> None:
        with self._lock:
            self._matrix, self._received = matrix, time.monotonic()

    def fresh_matrix(self) -> Optional[list[list[float]]]:
        """None when no pose arrived in the last 250 ms, so PC2's watchdog
        stops the motors after the Quest disconnects instead of being renewed
        with a cached pose."""

        with self._lock:
            if self._matrix is None or time.monotonic() - self._received > HEAD_POSE_MAX_AGE_S:
                return None
            return self._matrix

    def _loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            matrix = self.fresh_matrix()
            if matrix is not None:
                try:
                    self._sender.send_openxr_matrix(matrix)
                except (OSError, ValueError):
                    pass
            deadline += self._period
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                deadline = time.monotonic()
                continue
            self._stop.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._sender.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    cert, key = resolve_tls(args.cert, args.key)
    from vuer import Vuer
    from vuer.schemas import ImageBackground

    receiver = ZmqStereoReceiver(args.img_server_ip, args.zmq_port)
    clock_sync = (None if args.no_clock_sync
                  else ClockSyncClient(args.img_server_ip, args.clock_sync_port))
    forwarder = (HeadPoseForwarder(args.neck_pose_ip, args.neck_pose_port, args.neck_pose_rate)
                 if args.neck_pose_ip else None)

    app = Vuer(host="0.0.0.0", cert=str(cert), key=str(key),
               queries=dict(grid=False), queue_len=3)

    async def on_camera_move(event, _session):
        if forwarder is None:
            return
        try:
            forwarder.update(camera_move_matrix(event.value["camera"]["matrix"]))
        except (KeyError, TypeError, ValueError, IndexError):
            pass

    app.add_handler("CAMERA_MOVE")(on_camera_move)

    async def show_stereo(session):
        period = 1.0 / args.display_fps
        last_report = time.monotonic()
        shown = 0
        last_frames = sent_frame = receiver.frames
        while True:
            frame, stamp = receiver.latest()
            # Vuer re-encodes each eye as JPEG, so only send frames that are new.
            if frame is not None and receiver.frames != sent_frame:
                sent_frame = receiver.frames
                left, right = split_eyes_rgb(frame)
                aspect = left.shape[1] / left.shape[0]
                try:
                    # layers=1 / 2 are rendered only by the left / right eye camera.
                    session.upsert([
                        ImageBackground(left, aspect=aspect, height=1,
                                        distanceToCamera=args.distance, layers=1,
                                        format="jpeg", quality=args.jpeg_quality,
                                        key="zed-left", interpolate=True),
                        ImageBackground(right, aspect=aspect, height=1,
                                        distanceToCamera=args.distance, layers=2,
                                        format="jpeg", quality=args.jpeg_quality,
                                        key="zed-right", interpolate=True),
                    ], to="bgChildren")
                except AssertionError:
                    return  # Vuer: this client disconnected
                shown += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                received = receiver.frames - last_frames
                text = (f"[zmq_stereo_vr] received {received / (now - last_report):.1f} fps, "
                        f"sent {shown / (now - last_report):.1f} fps")
                clock = clock_sync.estimate() if clock_sync is not None else None
                if stamp is not None and clock is not None:
                    age = (time.monotonic_ns() - clock.pc2_to_host_ns(
                        stamp.capture_pc2_monotonic_ns)) / 1e6
                    text += f", image age at Host {age:.0f} ms"
                if forwarder is not None:
                    text += ", head pose " + ("live" if forwarder.fresh_matrix() else "none")
                print(text, flush=True)
                last_report, shown, last_frames = now, 0, receiver.frames
            await asyncio.sleep(period)

    app.spawn(start=False)(show_stereo)
    print(f"[zmq_stereo_vr] ZMQ tcp://{args.img_server_ip}:{args.zmq_port} -> Vuer :8012; "
          "open https://<host-ip>:8012/?ws=wss://<host-ip>:8012 in the Quest", flush=True)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        if forwarder is not None:
            forwarder.close()
        if clock_sync is not None:
            clock_sync.close()
        receiver.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileNotFoundError, ImportError) as error:
        print(f"zmq_stereo_vr error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

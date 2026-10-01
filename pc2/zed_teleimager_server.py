#!/usr/bin/env python3
"""Run Unitree teleimager with a synchronized ZED Mini stereo head camera.

This is a compatibility launcher, not a second image protocol. It injects a
ZED-backed head camera into teleimager and then reuses teleimager's existing
ZMQ config service, JPEG publisher, and WebRTC publisher unchanged.

Each ZMQ JPEG carries the frame's PC2 capture time in a comment segment, and a
UDP clock-sync responder lets the Host convert it to its own clock (see
PROTOCOL.md). Both are invisible to teleimager clients that ignore them.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import signal
import sys
from typing import Any, Optional, Sequence

import cv2
import yaml

try:
    from .frame_timing import DEFAULT_SYNC_PORT, ClockSyncServer, stamp_jpeg
    from .zed_stereo import ZedCaptureOptions, ZedStereoCapture
except ImportError:
    from frame_timing import (  # type: ignore[no-redef]
        DEFAULT_SYNC_PORT, ClockSyncServer, stamp_jpeg)
    from zed_stereo import ZedCaptureOptions, ZedStereoCapture  # type: ignore[no-redef]


DEFAULT_CONFIG = Path(__file__).with_name("cam_config_zed.yaml")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose ZED Mini stereo through the existing teleimager pipeline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--serial-number", type=int, default=None)
    parser.add_argument("--clock-sync-port", type=int, default=DEFAULT_SYNC_PORT,
                        help="UDP port answering Host clock-sync probes")
    parser.add_argument("--no-clock-sync", action="store_true",
                        help="do not run the clock-sync responder")
    args = parser.parse_args(argv)
    if not 1 <= args.clock_sync_port <= 65535:
        parser.error("--clock-sync-port must be in 1..65535")
    return args


def load_and_validate_config(path: Path) -> tuple[dict[str, Any], ZedCaptureOptions]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("camera config must be a mapping")

    head = config.get("head_camera")
    if not isinstance(head, dict):
        raise ValueError("camera config requires head_camera")
    if not head.get("binocular", False):
        raise ValueError("head_camera.binocular must be true")
    if not head.get("enable_zmq", False):
        raise ValueError(
            "head_camera.enable_zmq must be true so xr_teleoperate can record both eyes"
        )
    if not head.get("enable_webrtc", False):
        raise ValueError(
            "head_camera.enable_webrtc must be true for low-latency Quest display"
        )
    for topic in ("left_wrist_camera", "right_wrist_camera"):
        camera = config.get(topic, {})
        if camera.get("enable_zmq", False) or camera.get("enable_webrtc", False):
            raise ValueError(
                f"{topic} must be disabled for the paper's active-stereo-only baseline"
            )

    options = ZedCaptureOptions(
        resolution=str(head.get("zed_resolution", "hd720")).lower(),
        fps=int(head.get("fps", 60)),
        camera_id=int(head.get("zed_camera_id", 0)),
        serial_number=(
            int(head["zed_serial_number"])
            if head.get("zed_serial_number") is not None
            else None
        ),
    )
    configured_shape = tuple(int(value) for value in head.get("image_shape", ()))
    if configured_shape != options.combined_shape:
        raise ValueError(
            "head_camera.image_shape must describe the combined left-right frame: "
            f"{options.combined_shape}, got {configured_shape}"
        )

    # ImageServer already knows how to construct and publish an OpenCVCamera.
    # The launcher replaces that constructor with the ZED adapter below. The
    # video_id is only a placeholder and is not used to open the ZED.
    head["type"] = "opencv"
    head["video_id"] = 0
    head["serial_number"] = None
    head["physical_path"] = None
    return config, options


def make_zed_camera_class(image_server_module: Any, options: ZedCaptureOptions):
    class ZedMiniCamera(image_server_module.BaseCamera):
        def __init__(
            self,
            cam_topic: str,
            _video_path: str,
            img_shape: list[int],
            fps: int,
            enable_zmq: bool = True,
            zmq_port: int = 55555,
            enable_webrtc: bool = False,
            webrtc_port: int = 66666,
            webrtc_codec: Optional[str] = None,
        ) -> None:
            super().__init__(
                cam_topic,
                img_shape,
                fps,
                enable_zmq,
                zmq_port,
                enable_webrtc,
                webrtc_port,
                webrtc_codec,
            )
            self._capture = ZedStereoCapture(options)
            self._capture.open()
            self._frame_sequence = 0

        def __str__(self) -> str:
            return (
                f"[ZedMiniCamera: {self._cam_topic}] synchronized rectified stereo "
                f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS"
            )

        def _update_frame(self) -> None:
            bgr = self._capture.grab()
            if self._enable_webrtc:
                self._webrtc_buffer.write(bgr)
            if self._enable_zmq:
                ok, encoded = cv2.imencode(".jpg", bgr)
                if not ok:
                    raise RuntimeError("could not JPEG-encode the ZED stereo frame")
                self._frame_sequence += 1
                self._zmq_buffer.write(stamp_jpeg(
                    encoded.tobytes(),
                    self._frame_sequence,
                    self._capture.latest_pc2_monotonic_ns,
                    self._capture.latest_zed_image_time_ns,
                ))
            self._ready.set()

        def release(self) -> None:
            self._capture.close()

    return ZedMiniCamera


class _ZedOnlyCameraFinder:
    """Bypass /dev/video discovery; PyZED owns and discovers the camera."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def is_vpath_exist(self, _video_path: str) -> bool:
        return True


def run(args: argparse.Namespace) -> int:
    config, options = load_and_validate_config(args.config)
    if args.camera_id is not None and args.serial_number is not None:
        raise ValueError("use only one of --camera-id and --serial-number")
    if args.camera_id is not None:
        options = ZedCaptureOptions(options.resolution, options.fps, args.camera_id)
    if args.serial_number is not None:
        options = ZedCaptureOptions(
            options.resolution, options.fps, 0, args.serial_number
        )

    try:
        import teleimager.image_server as image_server
    except ImportError as error:
        raise RuntimeError(
            "teleimager is not installed. Install the teleimager submodule from "
            "the same xr_teleoperate checkout used by the Host."
        ) from error

    image_server.OpenCVCamera = make_zed_camera_class(image_server, options)
    image_server.CameraFinder = _ZedOnlyCameraFinder
    # Image capture and the neck process both use PC2 CLOCK_MONOTONIC, so one
    # responder here serves image and neck timestamps alike.
    clock_sync = None if args.no_clock_sync else ClockSyncServer(port=args.clock_sync_port)
    try:
        server = image_server.ImageServer(config)
        signal.signal(signal.SIGINT, partial(image_server.signal_handler, server))
        signal.signal(signal.SIGTERM, partial(image_server.signal_handler, server))
        server.start()
        server.wait()
    finally:
        if clock_sync is not None:
            clock_sync.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"ZED teleimager server error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

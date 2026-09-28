"""Synchronized, rectified ZED stereo capture helpers.

The public output format is one BGR image with the left eye in the left half
and the right eye in the right half. That is the binocular head-camera format
consumed by Unitree teleimager/televuer and split by xr_teleoperate's recorder.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np


class ZedStereoError(RuntimeError):
    """Raised when the ZED cannot produce a valid synchronized stereo frame."""


@dataclass(frozen=True)
class ZedCaptureOptions:
    resolution: str = "hd720"
    fps: int = 60
    camera_id: int = 0
    serial_number: Optional[int] = None

    def __post_init__(self) -> None:
        if self.resolution not in {"hd720", "vga"}:
            raise ValueError("resolution must be 'hd720' or 'vga'")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.camera_id < 0:
            raise ValueError("camera_id cannot be negative")
        if self.serial_number is not None and self.serial_number <= 0:
            raise ValueError("serial_number must be positive")

    @property
    def per_eye_shape(self) -> tuple[int, int]:
        return (720, 1280) if self.resolution == "hd720" else (376, 672)

    @property
    def combined_shape(self) -> tuple[int, int]:
        height, width = self.per_eye_shape
        return height, width * 2


def _as_bgr(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ZedStereoError(f"expected HxWx3/4 image, got {array.shape}")
    # PyZED retrieve_image normally returns BGRA. The first three channels are
    # already ordered B, G, R.
    return np.ascontiguousarray(array[:, :, :3])


def pack_left_right(
    left: Any,
    right: Any,
    expected_per_eye_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Return synchronized left/right BGR images packed horizontally."""

    left_bgr = _as_bgr(left)
    right_bgr = _as_bgr(right)
    if left_bgr.shape != right_bgr.shape:
        raise ZedStereoError(
            f"left/right shapes differ: {left_bgr.shape} != {right_bgr.shape}"
        )
    if expected_per_eye_shape is not None:
        actual = left_bgr.shape[:2]
        if actual != expected_per_eye_shape:
            raise ZedStereoError(
                f"unexpected per-eye shape: {actual} != {expected_per_eye_shape}"
            )
    return np.concatenate((left_bgr, right_bgr), axis=1)


class ZedStereoCapture:
    """Own one ZED and retrieve both rectified eyes after each single grab()."""

    def __init__(
        self,
        options: ZedCaptureOptions,
        sl_module: Optional[Any] = None,
    ) -> None:
        self.options = options
        if sl_module is None:
            try:
                import pyzed.sl as sl_module  # type: ignore[no-redef]
            except ImportError as error:
                raise ZedStereoError(
                    "PyZED is unavailable. Install the ZED SDK Python API that "
                    "matches the JetPack/ZED SDK on PC2."
                ) from error

        self._sl = sl_module
        self._camera = sl_module.Camera()
        self._left = sl_module.Mat()
        self._right = sl_module.Mat()
        self._runtime = sl_module.RuntimeParameters()
        # Canonical time for future dataset synchronization. This clock is
        # shared with the motor process because both run on PC2.
        self.latest_pc2_monotonic_ns: Optional[int] = None
        # ZED hardware/SDK time is retained for diagnostics only. It must not
        # be mixed directly with PC2 monotonic time.
        self.latest_zed_image_time_ns: Optional[int] = None
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return
        sl = self._sl
        init = sl.InitParameters()
        init.camera_resolution = (
            sl.RESOLUTION.HD720
            if self.options.resolution == "hd720"
            else sl.RESOLUTION.VGA
        )
        init.camera_fps = self.options.fps
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER

        if self.options.serial_number is not None:
            if not hasattr(init, "set_from_serial_number"):
                raise ZedStereoError("this PyZED version cannot select a serial number")
            init.set_from_serial_number(self.options.serial_number)
        elif self.options.camera_id:
            if not hasattr(init, "set_from_camera_id"):
                raise ZedStereoError("this PyZED version cannot select a camera id")
            init.set_from_camera_id(self.options.camera_id)

        status = self._camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"could not open ZED: {status}")
        self._opened = True

    def grab(self) -> np.ndarray:
        if not self._opened:
            raise ZedStereoError("ZED is not open")
        sl = self._sl
        status = self._camera.grab(self._runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ZedStereoError(f"ZED grab failed: {status}")
        # grab() returns for the newly acquired synchronized pair. Timestamp it
        # immediately in the PC2 monotonic domain before conversion/encoding.
        self.latest_pc2_monotonic_ns = time.monotonic_ns()

        # Both views come from one successful grab, preserving hardware
        # synchronization and rectification.
        self._camera.retrieve_image(self._left, sl.VIEW.LEFT, sl.MEM.CPU)
        self._camera.retrieve_image(self._right, sl.VIEW.RIGHT, sl.MEM.CPU)
        self.latest_zed_image_time_ns = self._read_image_timestamp_ns()
        return pack_left_right(
            self._left.get_data(),
            self._right.get_data(),
            self.options.per_eye_shape,
        )

    def _read_image_timestamp_ns(self) -> Optional[int]:
        try:
            timestamp = self._camera.get_timestamp(self._sl.TIME_REFERENCE.IMAGE)
            return int(timestamp.get_nanoseconds())
        except (AttributeError, TypeError, ValueError):
            return None

    def close(self) -> None:
        if self._opened:
            self._camera.close()
            self._opened = False

    def __enter__(self) -> "ZedStereoCapture":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

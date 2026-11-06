"""Quest display stream: per-eye JPEGs prepared on PC2, relayed unchanged by the Host.

The recording stream (teleimager ZMQ, port 55555) keeps the full-resolution
left|right frame. This second stream exists only for the Quest display. In its
own thread the publisher takes the newest ZED frame, splits it into the left and
right eye, scales both down, JPEG-encodes each eye and publishes the two JPEGs in
ONE ZMQ message together with the frame number. The Host forwards those JPEG
bytes to Vuer as they are (no decode, colour conversion, resize or re-encode).

One message per frame means the two eyes can never come from different frames,
and it keeps ZMQ ``CONFLATE`` usable on the subscriber (conflate drops
multipart messages). Wire format: PROTOCOL.md "Display stream v1".
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any, Callable, Optional

try:
    import zmq
except ImportError:  # unit tests inject a fake socket
    zmq = None

_NOBLOCK = getattr(zmq, "NOBLOCK", 1)
_AGAIN = getattr(zmq, "Again", BlockingIOError)

DEFAULT_DISPLAY_PORT = 55556

DISPLAY_MAGIC = b"ADSP"
DISPLAY_VERSION = 1
# magic, version, flags, reserved, frame_sequence, capture_pc2_monotonic_ns,
# zed_image_time_ns, left_len, right_len, eye_width, eye_height
DISPLAY_STRUCT = struct.Struct("!4sBBHIQQIIHH")
DISPLAY_HEADER_SIZE = DISPLAY_STRUCT.size


def pack_display_message(left_jpeg: bytes, right_jpeg: bytes, frame_sequence: int,
                         capture_pc2_monotonic_ns: int, zed_image_time_ns: Optional[int],
                         eye_width: int, eye_height: int) -> bytes:
    header = DISPLAY_STRUCT.pack(
        DISPLAY_MAGIC, DISPLAY_VERSION, 0, 0, frame_sequence & 0xFFFFFFFF,
        capture_pc2_monotonic_ns, zed_image_time_ns or 0,
        len(left_jpeg), len(right_jpeg), eye_width, eye_height)
    return header + left_jpeg + right_jpeg


def _default_split(bgr: Any) -> tuple[Any, Any]:
    width = bgr.shape[1]
    if width % 2:
        raise ValueError(f"combined frame width must be even, got {width}")
    half = width // 2
    return bgr[:, :half], bgr[:, half:]


def _default_resize(eye: Any, scale: float) -> Any:
    import cv2
    return cv2.resize(eye, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _default_encode(eye: Any, quality: int) -> bytes:
    import cv2
    import numpy as np
    ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(eye),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not JPEG-encode a display eye image")
    return encoded.tobytes()


class DisplayStreamPublisher:
    """Publish per-eye display JPEGs; the newest frame wins, older ones are dropped."""

    def __init__(self, port: int = DEFAULT_DISPLAY_PORT, scale: float = 0.5,
                 jpeg_quality: int = 70, fps: float = 30.0,
                 socket: Optional[Any] = None,
                 split: Callable[[Any], tuple] = _default_split,
                 resize: Callable[[Any, float], Any] = _default_resize,
                 encode: Callable[[Any, int], bytes] = _default_encode,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not 0.1 <= scale <= 1.0:
            raise ValueError("display scale must be in 0.1..1.0")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("display JPEG quality must be in 1..100")
        if not fps > 0:
            raise ValueError("display fps must be positive")
        if socket is None:
            if zmq is None:
                raise RuntimeError("pyzmq is required for the display stream")
            self._context = zmq.Context()
            socket = self._context.socket(zmq.PUB)
            # Never queue for a slow viewer: PUB drops once two frames are pending.
            socket.setsockopt(zmq.SNDHWM, 2)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(f"tcp://*:{port}")
        else:
            self._context = None
        self._socket = socket
        self.port = port
        self._scale, self._quality = scale, jpeg_quality
        self._period = 1.0 / fps
        self._split, self._resize, self._encode, self._clock = split, resize, encode, clock
        self._condition = threading.Condition()
        self._pending: Optional[tuple] = None
        self._last_sent = float("-inf")
        self._stop = False
        # Counters are written by the publisher thread (sent/bytes/encode) or the
        # capture thread (submitted); reads are for logging only.
        self.submitted = 0   # frames handed over by the capture loop
        self.sent = 0        # display messages published
        self.dropped = 0     # superseded, over the fps cap, or ZMQ queue full
        self.errors = 0
        self.bytes_sent = 0
        self.encode_seconds = 0.0
        self._thread = threading.Thread(target=self._loop, name="display-stream", daemon=True)
        self._thread.start()

    def submit(self, bgr: Any, frame_sequence: int, capture_pc2_monotonic_ns: int,
               zed_image_time_ns: Optional[int] = None) -> None:
        """Hand over a frame without blocking the capture loop.

        ``bgr`` must not be modified afterwards (ZED packing makes a new array
        every grab), and is shared read-only with the recording encoder.
        """

        with self._condition:
            self.submitted += 1
            if self._pending is not None:
                self.dropped += 1
            self._pending = (bgr, frame_sequence, capture_pc2_monotonic_ns, zed_image_time_ns)
            self._condition.notify()

    def _loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                bgr, sequence, capture_ns, zed_ns = self._pending
                self._pending = None
            # Frame-rate cap (10 % slack so capture jitter does not drop frames).
            # Checked before any pixel work: skipped frames cost no encoding.
            now = self._clock()
            if now - self._last_sent < 0.9 * self._period:
                self.dropped += 1
                continue
            try:
                started = time.perf_counter()
                left, right = self._split(bgr)
                if self._scale != 1.0:
                    left, right = self._resize(left, self._scale), self._resize(right, self._scale)
                left_jpeg, right_jpeg = self._encode(left, self._quality), self._encode(right, self._quality)
                height, width = left.shape[0], left.shape[1]
                message = pack_display_message(left_jpeg, right_jpeg, sequence, capture_ns,
                                               zed_ns, width, height)
                self.encode_seconds += time.perf_counter() - started
                self._send(message)
            except Exception:
                # A failed display frame must never stop the capture loop.
                self.errors += 1
                continue
            self._last_sent = now

    def _send(self, message: bytes) -> None:
        try:
            self._socket.send(message, _NOBLOCK)
        except _AGAIN:
            self.dropped += 1
            return
        self.sent += 1
        self.bytes_sent += len(message)

    def snapshot(self) -> dict:
        return {"submitted": self.submitted, "sent": self.sent, "dropped": self.dropped,
                "errors": self.errors, "bytes": self.bytes_sent,
                "encode_seconds": self.encode_seconds}

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        self._socket.close()
        if self._context is not None:
            self._context.term()


class DisplayStatsReporter:
    """Log PC2 capture/display rates and sizes every ``interval`` seconds.

    Capture fps counts frames the capture loop handed over; sent fps counts
    display messages actually published. Sizes are per message (both eyes).
    """

    def __init__(self, publisher: DisplayStreamPublisher, interval: float = 2.0,
                 log: Callable[[str], None] = lambda text: print(text, flush=True)) -> None:
        self._publisher, self._interval, self._log = publisher, interval, log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="display-stats", daemon=True)
        self._thread.start()

    @staticmethod
    def format(previous: dict, current: dict, seconds: float) -> str:
        sent = current["sent"] - previous["sent"]
        sent_bytes = current["bytes"] - previous["bytes"]
        text = (f"[display] PC2 captured {(current['submitted'] - previous['submitted']) / seconds:.1f} fps, "
                f"display sent {sent / seconds:.1f} fps, "
                f"dropped {current['dropped'] - previous['dropped']}, "
                f"errors {current['errors'] - previous['errors']}")
        if sent:
            encode_ms = 1000.0 * (current["encode_seconds"] - previous["encode_seconds"]) / sent
            text += (f", {sent_bytes / sent / 1024:.1f} KB/msg (both eyes), "
                     f"{sent_bytes * 8 / seconds / 1e6:.2f} Mbit/s, encode {encode_ms:.1f} ms/msg")
        return text

    def _loop(self) -> None:
        previous, then = self._publisher.snapshot(), time.monotonic()
        while not self._stop.wait(self._interval):
            current, now = self._publisher.snapshot(), time.monotonic()
            self._log(self.format(previous, current, now - then))
            previous, then = current, now

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

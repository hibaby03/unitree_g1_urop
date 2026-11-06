"""Host side of the Quest display stream: receive per-eye JPEGs, forward unchanged to Vuer.

PC2 (pc2/display_stream.py) publishes one ZMQ message per frame holding the left
and right eye JPEGs and the frame number. This module:

- keeps only the newest message (ZMQ CONFLATE, plus a single latest slot);
- hands each message out at most once, so a frame is never re-sent;
- builds Vuer ``ImageBackground(src=<jpeg bytes>)`` layers with the received bytes
  as they are. There is no JPEG decode, BGR/RGB conversion, resize or
  re-encode on the Host for display.

``ImageBackground`` takes the bytes through its ``src`` argument: in the Vuer
0.0.60 ``Image`` element a non-None ``src`` skips every image conversion and is
serialized as binary (vuer/schemas/html_components.py). It runs in its own thread
(receiver) and asyncio task (relay), apart from the IK/control/recording loop.

Wire format: PROTOCOL.md "Display stream v1".
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from .frame_timing import FrameStamp
except ImportError:
    from frame_timing import FrameStamp  # type: ignore[no-redef]

DEFAULT_DISPLAY_PORT = 55556

DISPLAY_MAGIC = b"ADSP"
DISPLAY_VERSION = 1
DISPLAY_STRUCT = struct.Struct("!4sBBHIQQIIHH")  # same layout as pc2/display_stream.py
DISPLAY_HEADER_SIZE = DISPLAY_STRUCT.size
JPEG_SOI = b"\xff\xd8"
NO_FRAME_WARNING_S = 3.0


@dataclass(frozen=True)
class DisplayFrame:
    """One display frame: both eyes of the SAME PC2 capture."""

    sequence: int
    capture_pc2_monotonic_ns: int
    zed_image_time_ns: int
    left_jpeg: bytes
    right_jpeg: bytes
    eye_width: int
    eye_height: int

    @property
    def stamp(self) -> FrameStamp:
        return FrameStamp(self.sequence, self.capture_pc2_monotonic_ns,
                          self.zed_image_time_ns or None)

    @property
    def aspect(self) -> float:
        return self.eye_width / self.eye_height


def parse_display_message(data: bytes) -> DisplayFrame:
    """Validate and split a display message; raises ValueError if malformed."""

    if len(data) < DISPLAY_HEADER_SIZE:
        raise ValueError("display message shorter than its header")
    (magic, version, _flags, _reserved, sequence, capture_ns, zed_ns,
     left_len, right_len, width, height) = DISPLAY_STRUCT.unpack_from(data)
    if magic != DISPLAY_MAGIC or version != DISPLAY_VERSION:
        raise ValueError("not a display stream v1 message")
    if len(data) != DISPLAY_HEADER_SIZE + left_len + right_len:
        raise ValueError("display message length does not match the header")
    if not left_len or not right_len or not width or not height:
        raise ValueError("display message has an empty eye image")
    view = memoryview(data)
    left = bytes(view[DISPLAY_HEADER_SIZE:DISPLAY_HEADER_SIZE + left_len])
    right = bytes(view[DISPLAY_HEADER_SIZE + left_len:])
    if not (left.startswith(JPEG_SOI) and right.startswith(JPEG_SOI)):
        raise ValueError("display eye image is not a JPEG")
    return DisplayFrame(sequence, capture_ns, zed_ns, left, right, width, height)


class DisplayFeedReceiver:
    """Newest-only receiver of the PC2 display stream (no pixel processing)."""

    def __init__(self, ip: str, port: int = DEFAULT_DISPLAY_PORT,
                 socket: Optional[Any] = None) -> None:
        self._owns_socket = socket is None
        if socket is None:
            import zmq

            self._context = zmq.Context()
            socket = self._context.socket(zmq.SUB)
            # Single-part messages, so CONFLATE keeps just the newest frame.
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{ip}:{port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket = socket
        self._lock = threading.Lock()
        self._latest: Optional[DisplayFrame] = None
        self._taken = 0          # receive counter value last handed out
        self.received = 0        # valid messages received from PC2
        self.bytes_received = 0
        self.bad_messages = 0
        self.forwarded = 0       # frames handed to Vuer (NOT the Quest display rate)
        self.started = time.monotonic()
        self.last_received = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="display-feed", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=100)):
                continue
            try:
                data = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop.is_set():
                    return
                raise
            self.ingest(data)

    def ingest(self, data: bytes) -> bool:
        """Store a message as the newest frame (called by the thread; testable)."""

        try:
            frame = parse_display_message(data)
        except ValueError:
            self.bad_messages += 1
            return False
        with self._lock:
            self._latest = frame
            self.received += 1
            self.bytes_received += len(data)
            self.last_received = time.monotonic()
        return True

    def take_new(self) -> Optional[DisplayFrame]:
        """The newest frame if it was not handed out before, else None."""

        with self._lock:
            if self._latest is None or self.received == self._taken:
                return None
            self._taken = self.received
            return self._latest

    def note_forwarded(self) -> None:
        self.forwarded += 1

    def snapshot(self) -> dict:
        return {"received": self.received, "bytes": self.bytes_received,
                "forwarded": self.forwarded, "bad": self.bad_messages}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        if self._owns_socket:
            self._socket.close()
            self._context.term()


def eye_layers(frame: DisplayFrame, image_background: Callable[..., Any], *,
               screen_y: float, distance: float) -> list:
    """Vuer layers for both eyes built from the received JPEG bytes as they are.

    layers=1 / 2 are rendered only by the left / right eye camera. The same
    plane position, distance and keys as the earlier display path are kept.
    """

    common = dict(aspect=frame.aspect, height=1, position=[0, screen_y, 0],
                  distanceToCamera=distance, interpolate=True)
    return [image_background(src=frame.left_jpeg, layers=1, key="zed-left", **common),
            image_background(src=frame.right_jpeg, layers=2, key="zed-right", **common)]


def format_report(previous: dict, current: dict, seconds: float,
                  age_ms: Optional[float] = None, tag: str = "display") -> str:
    """One status line. 'forwarded' counts frames given to Vuer, not Quest FPS."""

    received = current["received"] - previous["received"]
    size = current["bytes"] - previous["bytes"]
    text = (f"[{tag}] Host received {received / seconds:.1f} fps from PC2, "
            f"forwarded to Vuer {(current['forwarded'] - previous['forwarded']) / seconds:.1f} fps "
            "(handed to Vuer, not the Quest's displayed fps)")
    if received:
        text += f", {size / received / 1024:.1f} KB/msg, {size * 8 / seconds / 1e6:.2f} Mbit/s"
    if current["bad"] != previous["bad"]:
        text += f", {current['bad'] - previous['bad']} malformed"
    if age_ms is not None:
        text += f", image age at Host {age_ms:.0f} ms"
    return text


async def relay_to_vuer(session: Any, feed: DisplayFeedReceiver,
                        image_background: Callable[..., Any], *, screen_y: float,
                        distance: float = 1.0, poll_s: float = 0.004,
                        report_s: float = 2.0, tag: str = "display",
                        age_ms: Optional[Callable[[DisplayFrame], Optional[float]]] = None,
                        log: Callable[[str], None] = lambda text: print(text, flush=True)) -> None:
    """Forward each new display frame to this Vuer session, newest only.

    A frame is sent once; with no new frame nothing is sent, so a stalled
    stream cannot pile up or repeat frames. Returns when the client is gone.
    """

    previous, then = feed.snapshot(), time.monotonic()
    last_frame: Optional[DisplayFrame] = None
    warned = False
    while True:
        frame = feed.take_new()
        if frame is not None:
            last_frame, warned = frame, False
            try:
                session.upsert(eye_layers(frame, image_background,
                                          screen_y=screen_y, distance=distance),
                               to="bgChildren")
            except AssertionError:
                return  # Vuer: this client disconnected
            feed.note_forwarded()
        now = time.monotonic()
        if feed.received == 0 and not warned and now - feed.started > NO_FRAME_WARNING_S:
            warned = True
            log(f"[{tag}] no display frames from PC2 yet: restart PC2 "
                "zed_teleimager_server.py (display stream) or use --legacy-display")
        if now - then >= report_s:
            current = feed.snapshot()
            log(format_report(previous, current, now - then,
                              age_ms(last_frame) if age_ms and last_frame else None, tag))
            previous, then = current, now
        await asyncio.sleep(poll_s)

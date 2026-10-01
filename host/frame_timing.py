"""Host side of image timestamps and Host/PC2 clock synchronization.

PC2 stamps every ZMQ JPEG with its CLOCK_MONOTONIC capture time
(pc2/frame_timing.py) and answers NTP-style UDP probes. This module:

- ``parse_jpeg_stamp`` reads the stamp from ``TeleImage.jpg``;
- ``ClockSyncClient`` estimates offset = PC2 monotonic - Host monotonic from
  the lowest-delay probe in a short window, so PC2 times (image capture, neck
  command/encoder) can be expressed on the Host monotonic clock;
- ``StampedFrameReader`` decodes the JPEG itself, so the image and its stamp
  always belong to the same frame (teleimager's own BGR decoder runs on a
  separate thread and may lag the raw JPEG);
- ``frame_timing_entry`` builds the per-item ``states["timing"]`` record.

Wire formats are in PROTOCOL.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import socket
import struct
import threading
import time
from typing import Any, Callable, Optional


JPEG_SOI = b"\xff\xd8"
JPEG_COM = b"\xff\xfe"
STAMP_MAGIC = b"AFTS"
STAMP_VERSION = 1
STAMP_STRUCT = struct.Struct("!4sBBHIQQ")

SYNC_MAGIC = b"ACLK"
SYNC_VERSION = 1
SYNC_REQUEST_STRUCT = struct.Struct("!4sBBHIQ")
SYNC_REPLY_STRUCT = struct.Struct("!4sBBHIQQQ")
DEFAULT_SYNC_PORT = 5007


# ---------------------------------------------------------------------------
# JPEG stamp
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameStamp:
    frame_sequence: int
    capture_pc2_monotonic_ns: int
    zed_image_time_ns: Optional[int]


def parse_jpeg_stamp(jpeg: Optional[bytes]) -> Optional[FrameStamp]:
    """Stamp from the COM segment directly after SOI, or None if absent."""

    header_end = 2 + 2 + 2 + STAMP_STRUCT.size
    if not jpeg or len(jpeg) < header_end:
        return None
    if jpeg[:2] != JPEG_SOI or jpeg[2:4] != JPEG_COM:
        return None
    (length,) = struct.unpack_from("!H", jpeg, 4)
    if length != STAMP_STRUCT.size + 2:
        return None
    magic, version, _flags, _reserved, sequence, capture_ns, zed_ns = (
        STAMP_STRUCT.unpack_from(jpeg, 6))
    if magic != STAMP_MAGIC or version != STAMP_VERSION or capture_ns == 0:
        return None
    return FrameStamp(sequence, capture_ns, zed_ns or None)


# ---------------------------------------------------------------------------
# Clock synchronization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClockSample:
    host_receive_ns: int
    offset_ns: int  # PC2 monotonic - Host monotonic
    delay_ns: int   # round trip minus PC2 processing


@dataclass(frozen=True)
class ClockEstimate:
    offset_ns: int
    uncertainty_ns: int  # half the round-trip delay of the chosen probe

    def pc2_to_host_ns(self, pc2_ns: int) -> int:
        return pc2_ns - self.offset_ns


def probe_sample(host_send_ns: int, pc2_receive_ns: int, pc2_send_ns: int,
                 host_receive_ns: int) -> ClockSample:
    """NTP offset/delay from one probe (t0, t1, t2, t3)."""

    offset = ((pc2_receive_ns - host_send_ns) + (pc2_send_ns - host_receive_ns)) // 2
    delay = (host_receive_ns - host_send_ns) - (pc2_send_ns - pc2_receive_ns)
    return ClockSample(host_receive_ns, offset, max(0, delay))


def best_estimate(samples, now_ns: int, max_age_ns: int) -> Optional[ClockEstimate]:
    """Offset from the lowest-delay sample that is younger than ``max_age_ns``.

    Queueing only ever adds delay, and an asymmetric extra delay biases the
    offset by at most half of it, so the fastest round trip is the most
    trustworthy. Keeping the window short bounds the error from clock drift.
    """

    fresh = [s for s in samples if now_ns - s.host_receive_ns <= max_age_ns]
    if not fresh:
        return None
    best = min(fresh, key=lambda s: s.delay_ns)
    return ClockEstimate(best.offset_ns, best.delay_ns // 2)


class ClockSyncClient:
    """Background prober of PC2's ClockSyncServer."""

    def __init__(self, ip: str, port: int = DEFAULT_SYNC_PORT, interval_s: float = 0.25,
                 window: int = 32, max_age_s: float = 10.0) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect((ip, port))
        self._interval = interval_s
        self._max_age_ns = int(max_age_s * 1e9)
        self._samples: deque[ClockSample] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._sequence = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="clock-sync-probe", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            deadline = time.monotonic() + self._interval
            send_ns = time.monotonic_ns()
            try:
                self._socket.send(SYNC_REQUEST_STRUCT.pack(
                    SYNC_MAGIC, SYNC_VERSION, 0, 0, self._sequence, send_ns))
            except OSError:
                # e.g. ICMP port unreachable while PC2 is not running yet.
                self._stop.wait(self._interval)
                continue
            while not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._socket.settimeout(remaining)
                try:
                    datagram = self._socket.recv(64)
                except (socket.timeout, OSError):
                    break
                receive_ns = time.monotonic_ns()
                if len(datagram) != SYNC_REPLY_STRUCT.size:
                    continue
                magic, version, _f, _r, sequence, echoed_ns, pc2_rx, pc2_tx = (
                    SYNC_REPLY_STRUCT.unpack(datagram))
                # Late replies to earlier probes are dropped: their t3 is wrong.
                if (magic != SYNC_MAGIC or version != SYNC_VERSION
                        or sequence != self._sequence or echoed_ns != send_ns):
                    continue
                with self._lock:
                    self._samples.append(probe_sample(send_ns, pc2_rx, pc2_tx, receive_ns))
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                self._stop.wait(remaining)

    def estimate(self) -> Optional[ClockEstimate]:
        with self._lock:
            samples = list(self._samples)
        return best_estimate(samples, time.monotonic_ns(), self._max_age_ns)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close()


# ---------------------------------------------------------------------------
# Frames and per-item timing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StampedFrame:
    bgr: Any  # np.ndarray or None
    stamp: Optional[FrameStamp]


def _decode_jpeg(jpeg: bytes) -> Any:
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


class StampedFrameReader:
    """Latest head frame decoded from the same JPEG its stamp came from.

    ``get_frame`` is ``ImageClient.get_head_frame``; create the client with
    ``request_bgr=False`` so teleimager does not decode every frame a second
    time. Decoding is cached until a new JPEG arrives.
    """

    def __init__(self, get_frame: Callable[[], Any],
                 decode: Callable[[bytes], Any] = _decode_jpeg) -> None:
        self._get_frame = get_frame
        self._decode = decode
        self._jpeg: Optional[bytes] = None
        self._frame = StampedFrame(None, None)

    def read(self) -> StampedFrame:
        jpeg = self._get_frame().jpg
        if not jpeg:
            return StampedFrame(None, None)
        if jpeg is not self._jpeg:
            self._jpeg = jpeg
            self._frame = StampedFrame(self._decode(jpeg), parse_jpeg_stamp(jpeg))
        return self._frame


def _age_ms(host_now_ns: int, pc2_ns: Optional[int],
            clock: Optional[ClockEstimate]) -> Optional[float]:
    if pc2_ns is None or clock is None:
        return None
    return round((host_now_ns - clock.pc2_to_host_ns(pc2_ns)) / 1e6, 3)


def frame_timing_entry(host_now_ns: int, frame: Optional[StampedFrame],
                       clock: Optional[ClockEstimate],
                       neck_packet: Any = None) -> dict:
    """Timing of one recorded item; all ``*_host_monotonic_ns`` share one clock.

    Ages are ``host_now_ns`` minus the PC2 sample time mapped to the Host
    clock, i.e. how old each observation was when the item was assembled.
    They are None when PC2 sent no stamp or no clock estimate exists yet.
    """

    stamp = frame.stamp if frame is not None else None
    capture_pc2 = stamp.capture_pc2_monotonic_ns if stamp is not None else None
    entry = {
        "host_monotonic_ns": host_now_ns,
        "clock_offset_ns": clock.offset_ns if clock is not None else None,
        "clock_uncertainty_ms": round(clock.uncertainty_ns / 1e6, 3) if clock is not None else None,
        "image_frame_sequence": stamp.frame_sequence if stamp is not None else None,
        "image_pc2_monotonic_ns": capture_pc2,
        "image_host_monotonic_ns": (clock.pc2_to_host_ns(capture_pc2)
                                    if capture_pc2 is not None and clock is not None else None),
        "image_age_ms": _age_ms(host_now_ns, capture_pc2, clock),
        "neck_present_age_ms": None,
        "neck_command_age_ms": None,
    }
    if neck_packet is not None:
        entry["neck_present_age_ms"] = _age_ms(
            host_now_ns, neck_packet.present_pc2_monotonic_ns, clock)
        entry["neck_command_age_ms"] = _age_ms(
            host_now_ns, neck_packet.command_pc2_monotonic_ns, clock)
    return entry

"""PC2 side of image timestamps and Host/PC2 clock synchronization.

- ``stamp_jpeg`` puts the frame's PC2 capture time into a JPEG comment (COM)
  segment right after SOI. Decoders ignore COM segments, so teleimager's ZMQ
  stream, its ImageClient and xr_teleoperate keep working unchanged while the
  Host recorder can read the stamp from ``TeleImage.jpg``.
- ``ClockSyncServer`` answers NTP-style UDP probes with PC2
  ``time.monotonic_ns()`` so the Host can map PC2 times (image capture, neck
  command/encoder) onto its own monotonic clock.

Wire formats are in PROTOCOL.md; the Host decoder is host/frame_timing.py.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional


JPEG_SOI = b"\xff\xd8"
JPEG_COM = b"\xff\xfe"
STAMP_MAGIC = b"AFTS"
STAMP_VERSION = 1
STAMP_STRUCT = struct.Struct("!4sBBHIQQ")
STAMP_SIZE = STAMP_STRUCT.size

SYNC_MAGIC = b"ACLK"
SYNC_VERSION = 1
SYNC_REQUEST_STRUCT = struct.Struct("!4sBBHIQ")
SYNC_REPLY_STRUCT = struct.Struct("!4sBBHIQQQ")
DEFAULT_SYNC_PORT = 5007


def stamp_jpeg(
    jpeg: bytes,
    frame_sequence: int,
    capture_pc2_monotonic_ns: int,
    zed_image_time_ns: Optional[int] = None,
) -> bytes:
    """Return ``jpeg`` with a timestamp COM segment inserted after SOI."""

    if not jpeg.startswith(JPEG_SOI):
        raise ValueError("not a JPEG (missing SOI marker)")
    payload = STAMP_STRUCT.pack(
        STAMP_MAGIC,
        STAMP_VERSION,
        0,
        0,
        frame_sequence & 0xFFFFFFFF,
        capture_pc2_monotonic_ns,
        zed_image_time_ns or 0,
    )
    # The segment length counts its own two bytes but not the marker.
    segment = JPEG_COM + struct.pack("!H", len(payload) + 2) + payload
    return JPEG_SOI + segment + jpeg[len(JPEG_SOI):]


class ClockSyncServer:
    """Background UDP responder for Host clock-offset probes."""

    def __init__(self, bind: str = "0.0.0.0", port: int = DEFAULT_SYNC_PORT) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((bind, port))
        self._socket.settimeout(0.1)
        self.port = self._socket.getsockname()[1]
        self.rejected = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="clock-sync-udp", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                datagram, address = self._socket.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            # Read the clock first so parsing is not counted as network delay.
            receive_ns = time.monotonic_ns()
            if len(datagram) != SYNC_REQUEST_STRUCT.size:
                self.rejected += 1
                continue
            magic, version, _flags, _reserved, sequence, host_send_ns = (
                SYNC_REQUEST_STRUCT.unpack(datagram))
            if magic != SYNC_MAGIC or version != SYNC_VERSION:
                self.rejected += 1
                continue
            reply = SYNC_REPLY_STRUCT.pack(SYNC_MAGIC, SYNC_VERSION, 0, 0, sequence,
                                           host_send_ns, receive_ns, time.monotonic_ns())
            try:
                self._socket.sendto(reply, address)
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close()

"""Record four-timestamp probes. Offset is PC2 monotonic minus Host monotonic."""
import os
import socket
import struct
import threading
import time

REQUEST = struct.Struct("!4sBBHIQ")
REPLY = struct.Struct("!4sBBHIQQQ")


def sync_port():
    port = int(os.environ.get("NECK_TRACE_SYNC_PORT", "5010"))
    if not 1 <= port <= 65535:
        raise ValueError("NECK_TRACE_SYNC_PORT must be in 1..65535")
    return port


def sample(t0, t1, t2, t3):
    delay = (t3 - t0) - (t2 - t1)
    if t3 < t0 or t2 < t1 or delay < 0:
        raise ValueError("invalid clock sample")
    return {"event": "clock_sample", "t0": t0, "t1": t1, "t2": t2, "t3": t3,
            "offset_ns": ((t1 - t0) + (t2 - t3)) // 2, "rtt_ns": delay,
            "uncertainty_ns": (delay + 1) // 2}


class ClockProbe:
    def __init__(self, ip, writer, port=None):
        self.writer = writer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((ip, sync_port() if port is None else port))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="neck-clock-probe", daemon=True)
        self.thread.start()

    def _loop(self):
        sequence = 0
        while not self.stop.is_set():
            sequence = (sequence + 1) & 0xFFFFFFFF
            deadline = time.monotonic() + 0.25
            t0 = time.monotonic_ns()
            try:
                self.sock.send(REQUEST.pack(b"ACLK", 1, 0, 0, sequence, t0))
                while not self.stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.sock.settimeout(remaining)
                    data = self.sock.recv(128)
                    t3 = time.monotonic_ns()
                    if len(data) != REPLY.size:
                        continue
                    magic, version, flags, reserved, seq, echoed, t1, t2 = REPLY.unpack(data)
                    if (magic, version, flags, reserved, seq, echoed) != (b"ACLK", 1, 0, 0, sequence, t0):
                        continue
                    try:
                        self.writer.emit(sample(t0, t1, t2, t3))
                    except ValueError:
                        pass
                    break
            except OSError:
                pass
            self.stop.wait(max(0, deadline - time.monotonic()))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.sock.close()

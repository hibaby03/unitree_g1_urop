"""Bounded background JSONL writer and process-shared CAMERA_MOVE snapshots."""
from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid


def enabled():
    return bool(os.environ.get("NECK_TRACE_DIR"))


class TraceWriter:
    def __init__(self, role, directory=None, capacity=8192):
        directory = Path(directory or os.environ["NECK_TRACE_DIR"]).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.path = directory / f"{role}-{self.run_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self.dropped = 0
        self.error = None
        self._closed = False
        self._thread = threading.Thread(target=self._write, name="neck-trace-writer", daemon=True)
        self._thread.start()
        self.emit({"event": "trace_start", "role": role, "run_id": self.run_id,
                   "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns(),
                   "pid": os.getpid(), "schema": 1})
        atexit.register(self.close)
        print(f"[neck trace] {self.path}", file=sys.stderr)

    def emit(self, record):
        if self._closed or self.error:
            self.dropped += 1
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=0.2)
                except queue.Empty:
                    self._file.flush()
                    continue
                self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                if time.monotonic() - last_flush >= 0.5:
                    self._file.flush()
                    last_flush = time.monotonic()
            self._file.write(json.dumps({"event": "trace_end", "dropped": self.dropped,
                                         "mono_ns": time.monotonic_ns()}) + "\n")
        except Exception as error:
            self.error = str(error)
            print(f"[neck trace] recording failed: {error}", file=sys.stderr)
        finally:
            self._file.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            print("[neck trace] writer did not finish; log may be incomplete", file=sys.stderr)
        if self.dropped:
            print(f"[neck trace] dropped {self.dropped} log records", file=sys.stderr)
        atexit.unregister(self.close)


class SharedPose:
    """Integer nanoseconds and matrix copied atomically across the Vuer fork."""
    def __init__(self):
        self.lock = mp.RLock()
        self.matrix = mp.Array("d", 16, lock=False)
        self.metadata = mp.Array("Q", 3, lock=False)  # event id, receipt mono, receipt wall

    def update(self, matrix, receive_ns, wall_ns):
        with self.lock:
            self.matrix[:] = [float(x) for row in matrix for x in row]
            self.metadata[0] += 1  # identical poses still get a new event id
            self.metadata[1:] = [receive_ns, wall_ns]

    def snapshot(self):
        with self.lock:
            values = list(self.matrix)
            event_id, mono, wall = self.metadata[:]
        if not event_id:
            return None
        return ([values[i:i + 4] for i in range(0, 16, 4)],
                {"pose_id": event_id, "pose_rx_host_ns": mono, "pose_rx_host_wall_ns": wall})


def install_televuer_hook(cls):
    """Instrument the installed class in this process, before it forks Vuer.

    Does not change any external repository. TeleVuer must register its bound
    on_cam_move handler and use fork, as the existing launcher already requires.
    """
    if not enabled() or getattr(cls, "_neck_trace_hook", False):
        return
    if mp.get_start_method() != "fork":
        raise RuntimeError("TeleVuer neck timing requires multiprocessing fork")
    original_init, original_move = cls.__init__, cls.on_cam_move

    def init(self, *args, **kwargs):
        self.neck_trace_pose = SharedPose()
        original_init(self, *args, **kwargs)

    async def move(self, event, session, *args, **kwargs):
        rx, wall = time.monotonic_ns(), time.time_ns()
        result = await original_move(self, event, session, *args, **kwargs)
        try:
            flat = event.value["camera"]["matrix"]
            if len(flat) != 16:
                return result
            matrix = [[float(flat[c * 4 + r]) for c in range(4)] for r in range(4)]
            # Validation stays in the sender, exactly as in normal forwarding.
            self.neck_trace_pose.update(matrix, rx, wall)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            pass
        return result

    cls.__init__, cls.on_cam_move = init, move
    cls._neck_trace_hook = True


def send_televuer_pose(sender, viewer):
    tracker = getattr(viewer, "neck_trace_pose", None)
    if enabled() and tracker is not None:
        sample = tracker.snapshot()
        if sample is None:
            return
        sender.send_openxr_matrix(sample[0], pose_stamp=sample[1])
    else:
        sender.send_openxr_matrix(viewer.head_pose)

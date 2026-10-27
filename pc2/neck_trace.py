"""PC2-local bounded JSONL writer and neck clock-probe port setting."""
from __future__ import annotations

import atexit
import json
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


def sync_port():
    port = int(os.environ.get("NECK_TRACE_SYNC_PORT", "5010"))
    if not 1 <= port <= 65535:
        raise ValueError("NECK_TRACE_SYNC_PORT must be in 1..65535")
    return port


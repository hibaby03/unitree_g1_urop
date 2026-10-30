"""Read-only terminal diagnostics; never used to decide robot control or saving."""
from __future__ import annotations

import math
import multiprocessing as mp
import time


class PoseEventDiagnostics:
    """Latest validation result, shared with the forked Vuer event process."""

    def __init__(self):
        self._head = mp.Array("c", 1024)
        self._hand = mp.Array("c", 1024)

    def update(self, source, error=""):
        # Diagnostics must not turn a rejected event into a control exception.
        try:
            buffer = getattr(self, "_" + source)
            with buffer.get_lock():
                buffer.value = str(error).encode("utf-8", errors="replace")[:1023]
        except Exception:
            pass

    def error(self, source):
        buffer = getattr(self, "_" + source)
        with buffer.get_lock():
            return buffer.value.decode("utf-8", errors="replace")


def pose_status(label, accepted_ns, error, now_ns, limit_ms=500.0):
    """Describe existing accepted-event timestamps without updating them."""
    age = (now_ns - accepted_ns) / 1e6 if accepted_ns else None
    if age is None:
        state, text = "missing", f"{label} NOT RECEIVED (no accepted event)"
    elif age < 0:
        state, text = "future", f"{label} FAILED (accepted timestamp is in the future)"
    elif age >= limit_ms:
        state, text = "stale", f"{label} STALE (age={age:.0f}ms, requires <{limit_ms:g}ms)"
    else:
        state, text = "ready", f"{label} received (age={age:.0f}ms)"
    if error:
        text += f"; latest event REJECTED: {error}"
    return (state, error), text


def xr_status(xr, now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    tv = xr.tvuer
    return [pose_status(label, getattr(tv, source + "_received_ns").value,
                        tv.pose_event_diagnostics.error(source), now_ns)
            for source, label in (("head", "head pose"), ("hand", "hand pose (both hands)"))]


def observation_status(frame, image_age_ms, neck_latest, touch):
    """Explain complete_observation's existing checks, without replacing it."""
    fields = []

    def add(label, issues, detail=""):
        fields.append((tuple(issues), f"{label}: " + (", ".join(issues) if issues else "OK") + detail))

    issues = []
    if frame is None or frame.bgr is None:
        issues.append("NOT RECEIVED")
    if image_age_ms is None:
        issues.append("age UNKNOWN")
    elif not image_age_ms <= 500.0:
        issues.append("age FAILED (requires <=500ms)")
    add("image", issues, "" if image_age_ms is None else f" (age={image_age_ms:.0f}ms)")
    packet, age = neck_latest
    issues = []
    if packet is None:
        issues.append("packet NOT RECEIVED")
    else:
        if packet.present_deg is None:
            issues.append("encoder MISSING")
        if not packet.torque_enabled:
            issues.append("torque DISABLED")
    if age is None:
        issues.append("age UNKNOWN")
    elif not age <= 500.0:
        issues.append("packet STALE (requires <=500ms)")
    add("neck", issues, "" if age is None else f" (packet age={age:.0f}ms)")
    for side in ("left_ee", "right_ee"):
        sample = touch.get(side)
        issues = []
        if not sample:
            issues.append("NOT RECEIVED")
        else:
            if not sample.get("pressure"):
                issues.append("pressure EMPTY")
            if not sample.get("age_ms", math.inf) <= 500.0:
                issues.append("STALE/age UNKNOWN (requires <=500ms)")
        age = sample.get("age_ms") if sample else None
        add(f"tactile {side}", issues, "" if age is None else f" (age={age:.0f}ms)")
    return fields


class TerminalDiagnostics:
    """Bound console traffic; changing ages alone do not trigger new lines."""

    def __init__(self, log, clock=time.monotonic):
        self.log, self.clock = log, clock
        self._last = {}

    def emit(self, scope, fields, repeat=False):
        try:
            now = self.clock()
            signature = tuple(key for key, _ in fields)
            previous = self._last.get(scope)
            if previous is not None:
                old_signature, last_time = previous
                elapsed = now - last_time
                if elapsed < 0.5 or (signature == old_signature and not (repeat and elapsed >= 5)):
                    return
            self.log(f"[{scope}] " + " | ".join(message for _, message in fields))
            self._last[scope] = (signature, now)
        except Exception:
            # Logging is observational and must never gate control or recording.
            pass

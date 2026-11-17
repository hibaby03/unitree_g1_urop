#!/usr/bin/env python3
"""Standalone hand-tracking pipeline probe (Host side). Changes no project file.

It starts the SAME fork TeleVuerWrapper that teleop_record_dex3_tactile.py
uses, but with no robot, no camera, no PC2 and no DDS, and reports where the
hand pipeline stops:

  Quest browser -> Vuer HAND_MOVE event -> fork on_hand_move -> left/right_hand_pos
                                                              -> arm poses (IK input)

For every hand event it records whether the fork's own handler accepted it or
raised (full traceback, any exception type), prints the raw payload layout once,
and from the parent process watches whether the resulting hand/wrist arrays are
non-zero, finite and actually change while you move your hands.

Usage (Host, same venv as the recorder; open the Quest browser page as usual):
    python3 hand_tracking_probe.py --cert cert.pem --key key.pem --seconds 40
Move both hands around during the run. Ctrl+C ends early and still prints the verdict.

Limit: it checks the Quest->host side only. If this passes but the robot still
does not move, the problem is after it (start gate, IK, robot_bridge/DDS link).
"""
from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import sys
import time
import traceback


def describe(value, depth=0):
    """Short description of a payload's layout (types, sizes), not its content."""
    import numpy as np
    if isinstance(value, (bytes, bytearray)):
        return f"bytes[{len(value)}] head={bytes(value[:16])!r}"
    if isinstance(value, str):
        return f"str[{len(value)}] head={value[:40]!r}"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape} {value.dtype}"
    if isinstance(value, dict):
        if depth >= 2:
            return f"dict[{len(value)}]"
        return "{" + ", ".join(f"{k!r}: {describe(v, depth + 1)}" for k, v in list(value.items())[:12]) + "}"
    if isinstance(value, (list, tuple)):
        first = describe(value[0], depth + 1) if value else "-"
        return f"{type(value).__name__}[{len(value)}] first={first}"
    return f"{type(value).__name__}({str(value)[:40]})"


def build_probe_class(original, shm, stats):
    import numpy as np
    from vuer.schemas import Hands

    shape = tuple(shm_shape[0])

    class ProbeTeleVuer(original):
        def __init__(self, *args, **kwargs):
            self._probe = stats
            self._probe_seen_errors = set()
            self._probe_printed_payload = {"hand": False, "head": False}
            # The legacy TeleVuer keeps an ndarray over a constructor-local
            # SharedMemory; retain the owner's mapping (same fix as xr_legacy).
            self._probe_shm = shm
            super().__init__(*args, **kwargs)
            self._probe_use_image()

        def _probe_use_image(self):
            self.img_array = np.ndarray(shape, dtype=np.uint8, buffer=self._probe_shm.buf)

        def vuer_run(self):
            self._probe_use_image()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                super().vuer_run()
            finally:
                loop.close()

        def _probe_report(self, kind, event, error):
            key = (kind, f"{type(error).__name__}: {error}")
            if key in self._probe_seen_errors:
                return
            self._probe_seen_errors.add(key)
            print(f"\n[child] {kind} handler RAISED {key[1]}", file=sys.stderr)
            print(f"[child] payload layout: {describe(getattr(event, 'value', event))}", file=sys.stderr)
            traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)

        async def on_cam_move(self, event, session, fps=60):
            try:
                await super().on_cam_move(event, session, fps)
            except Exception as error:  # noqa: BLE001 - diagnostics: any type
                with self._probe["cam_exc"].get_lock():
                    self._probe["cam_exc"].value += 1
                self._probe_report("head", event, error)
                return
            with self._probe["cam_ok"].get_lock():
                self._probe["cam_ok"].value += 1
            self._probe["cam_ns"].value = time.monotonic_ns()

        async def on_hand_move(self, event, session, fps=60):
            with self._probe["hand_seen"].get_lock():
                self._probe["hand_seen"].value += 1
            if not self._probe_printed_payload["hand"]:
                self._probe_printed_payload["hand"] = True
                print(f"[child] first hand event layout: {describe(getattr(event, 'value', event))}",
                      file=sys.stderr, flush=True)
            try:
                await super().on_hand_move(event, session, fps)
            except Exception as error:  # noqa: BLE001
                with self._probe["hand_exc"].get_lock():
                    self._probe["hand_exc"].value += 1
                self._probe_report("hand", event, error)
                return
            with self._probe["hand_ok"].get_lock():
                self._probe["hand_ok"].value += 1
            self._probe["hand_ns"].value = time.monotonic_ns()

        async def main_image_binocular(self, session, fps=60):
            # No camera: only subscribe to the Quest hand stream (hands visible).
            session.upsert(Hands(stream=True, key="hands", hideLeft=False, hideRight=False),
                           to="bgChildren")
            while True:
                await asyncio.sleep(1.0)

    return ProbeTeleVuer


shm_shape = [None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_dir = os.path.expanduser("~/.config/xr_teleoperate")
    parser.add_argument("--cert", default=os.path.join(default_dir, "cert.pem"))
    parser.add_argument("--key", default=os.path.join(default_dir, "key.pem"))
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--img-shape", type=int, nargs=2, default=(480, 1280), metavar=("H", "W"),
                        help="dummy stereo image size (W = both eyes), no camera is used")
    parser.add_argument("--gate-age", type=float, default=0.5,
                        help="freshness window the recorder's start gate uses (s)")
    args = parser.parse_args()

    for path in (args.cert, args.key):
        if not os.path.isfile(path):
            parser.error(f"TLS file not found: {path} (pass --cert/--key as for the recorder)")
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)  # this fork of TeleVuer requires it

    import numpy as np
    from multiprocessing import shared_memory
    try:
        import televuer
        from televuer import TeleVuerWrapper
    except ImportError as error:
        print(f"FAIL: cannot import televuer ({error}). Use the recorder's Python environment.")
        return 2

    shape = (args.img_shape[0], args.img_shape[1], 3)
    shm_shape[0] = shape
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
    np.ndarray(shape, dtype=np.uint8, buffer=shm.buf).fill(0)

    stats = {name: mp.Value("Q", 0) for name in
             ("hand_seen", "hand_ok", "hand_exc", "hand_ns", "cam_ok", "cam_exc", "cam_ns")}

    module = sys.modules[TeleVuerWrapper.__module__]
    original = module.TeleVuer
    module.TeleVuer = build_probe_class(original, shm, stats)
    wrapper = None
    try:
        wrapper = TeleVuerWrapper(
            binocular=True, use_hand_tracking=True, img_shape=shape, img_shm_name=shm.name,
            return_state_data=True, return_hand_rot_data=False,
            cert_file=args.cert, key_file=args.key, webrtc=False)
    finally:
        module.TeleVuer = original

    print(f"televuer from {os.path.dirname(os.path.abspath(televuer.__file__))}")
    print("Server up. Open the Quest browser page, enter XR, and MOVE BOTH HANDS "
          f"for {args.seconds:.0f}s (Ctrl+C to stop early).\n")

    def arr(data, name):
        value = getattr(data, name, None)
        return None if value is None else np.asarray(value, dtype=float)

    names = ("left_hand_pos", "right_hand_pos", "left_arm_pose", "right_arm_pose")
    lo, hi, prev = {}, {}, {}
    nonfinite = set()
    exited = None
    last = {k: 0 for k in ("hand_ok", "cam_ok")}
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            time.sleep(1.0)
            if not wrapper.tvuer.process.is_alive():
                exited = wrapper.tvuer.process.exitcode
                print(f"Vuer child EXITED (exitcode {exited}); see its traceback above.")
                break
            data = wrapper.get_motion_state_data()
            moved = {}
            for name in names:
                value = arr(data, name)
                if value is None:
                    continue
                if not np.isfinite(value).all():
                    nonfinite.add(name)
                    continue
                lo[name] = value if name not in lo else np.minimum(lo[name], value)
                hi[name] = value if name not in hi else np.maximum(hi[name], value)
                moved[name] = float(np.abs(value - prev[name]).max()) if name in prev else 0.0
                prev[name] = value
            now = time.monotonic_ns()
            hand_age = (now - stats["hand_ns"].value) / 1e9 if stats["hand_ns"].value else float("inf")
            cam_age = (now - stats["cam_ns"].value) / 1e9 if stats["cam_ns"].value else float("inf")
            hand_ok, cam_ok = stats["hand_ok"].value, stats["cam_ok"].value
            gate = hand_age < args.gate_age and cam_age < args.gate_age
            print(f"t={time.monotonic() - start:4.0f}s "
                  f"hand ok/exc/seen={hand_ok}/{stats['hand_exc'].value}/{stats['hand_seen'].value} "
                  f"(+{hand_ok - last['hand_ok']}/s, age {hand_age:.2f}s) | "
                  f"head ok/exc={cam_ok}/{stats['cam_exc'].value} (+{cam_ok - last['cam_ok']}/s, age {cam_age:.2f}s) | "
                  f"d(L,R hand)={moved.get('left_hand_pos', float('nan')):.4f},"
                  f"{moved.get('right_hand_pos', float('nan')):.4f} | "
                  f"start gate {'OPEN' if gate else 'CLOSED'}")
            last.update(hand_ok=hand_ok, cam_ok=cam_ok)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        spans = {n: float((hi[n] - lo[n]).max()) for n in lo}
        seen, ok, exc = stats["hand_seen"].value, stats["hand_ok"].value, stats["hand_exc"].value
        cam_ok, cam_exc = stats["cam_ok"].value, stats["cam_exc"].value
        if wrapper is not None:
            proc = wrapper.tvuer.process
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        shm.close()
        shm.unlink()

    print("\n=== verdict ===")
    print(f"hand events seen={seen} accepted={ok} raised={exc}; head accepted={cam_ok} raised={cam_exc}")
    print("value span over run (max-min): " + ", ".join(f"{n}={s:.4f}" for n, s in spans.items()))
    problems = []
    if exited is not None:
        problems.append(f"Vuer child process died (exitcode {exited}).")
    if seen == 0:
        problems.append("No hand events reached the server: the Quest is not streaming hands "
                        "(not in hand-tracking mode / permission / page not in XR / stale TLS page).")
    elif ok == 0:
        problems.append("Every hand event made the fork's on_hand_move raise: the fork cannot parse "
                        "this payload. The traceback and payload layout were printed above ([child]).")
    elif exc:
        problems.append(f"{exc} of {seen} hand events raised (first traceback printed above).")
    if cam_ok == 0:
        problems.append("No accepted head (camera) events: the start gate can never open.")
    if nonfinite:
        problems.append(f"NaN/inf in {sorted(nonfinite)}.")
    if ok:
        for name in ("left_hand_pos", "right_hand_pos"):
            if name in hi and not np.any(hi[name]) and not np.any(lo[name]):
                problems.append(f"{name} stayed all zeros although events were accepted: the handler "
                                "accepts events but does not fill the array.")
            elif spans.get(name, 0) < 1e-4:
                problems.append(f"{name} never changed (span {spans.get(name, 0):.1e}); "
                                "move that hand, or the parser does not update it.")
        for name in ("left_arm_pose", "right_arm_pose"):
            if name in spans and spans[name] < 1e-4:
                problems.append(f"{name} (IK input) never changed: the arm would not move.")
    if problems:
        print("FAIL:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("PASS: Quest hand tracking -> fork handler -> hand/arm targets works. If the robot still "
          "does not move, look after this stage: start gate log, IK, robot_bridge / DDS link.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

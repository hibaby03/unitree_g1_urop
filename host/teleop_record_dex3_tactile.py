#!/usr/bin/env python3
"""G1_29 + Dex3-1 teleoperation with neck forwarding and full episode recording.

This is a self-contained main loop for the active-stereo setup. It uses the
same xr_teleoperate modules as teleop/teleop_hand_and_arm.py (TeleVuerWrapper,
G1_29_ArmIK/Controller, Dex3_1_Controller, ImageClient, EpisodeWriter), and
additionally:

- forwards the raw Quest head pose to PC2 over UDP (see PROTOCOL.md), so the
  ZED neck follows the operator's head during teleoperation and recording;
- records left/right arm state (q, dq) and action (IK q, feed-forward tau);
- records Dex3-1 hand state/action and the Dex3-1 tactile (press sensor)
  pressure/temperature arrays in each item's ``tactiles`` field;
- receives the neck state that PC2's head_pose_receiver sends back (UDP 5006)
  and records encoder yaw/pitch in states["neck"] and the commanded yaw/pitch
  in actions["neck"];
- records per-item timing in states["timing"]: the Host monotonic sample time,
  the ZED frame's PC2 capture time (stamped into the ZMQ JPEG) and its age, and
  the neck command/encoder ages, using a Host/PC2 clock offset measured over
  UDP (see host/frame_timing.py and PROTOCOL.md).

Keys (sshkeyboard, same as xr_teleoperate):
  r: start following the operator   s: start/save an episode   q: quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional, Sequence

try:
    from .frame_timing import (DEFAULT_SYNC_PORT, ClockSyncClient, StampedFrameReader,
                               frame_timing_entry)
    from .head_pose_udp import HeadPoseUdpSender, enabled, install_televuer_hook, send_televuer_pose
    from .neck_state_receiver import NeckStateReceiver, neck_record_entries
    from .xr_legacy import LegacyStereoXR, LegacyHandSession, check_legacy_api, installed_versions
    from .zmq_stereo_vr import resolve_tls
except ImportError:
    from frame_timing import (  # type: ignore[no-redef]
        DEFAULT_SYNC_PORT, ClockSyncClient, StampedFrameReader, frame_timing_entry)
    from head_pose_udp import HeadPoseUdpSender, enabled, install_televuer_hook, send_televuer_pose  # type: ignore[no-redef]
    from neck_state_receiver import (  # type: ignore[no-redef]
        NeckStateReceiver, neck_record_entries)
    from xr_legacy import LegacyStereoXR, LegacyHandSession, check_legacy_api, installed_versions
    from zmq_stereo_vr import resolve_tls


DEX3_NUM_MOTORS = 7
TOPIC_DEX3_LEFT_STATE = "rt/dex3/left/state"
TOPIC_DEX3_RIGHT_STATE = "rt/dex3/right/state"


def positive_rate(value: str) -> float:
    parsed = float(value)
    if not 1.0 <= parsed <= 240.0:
        raise argparse.ArgumentTypeError("must be in 1..240")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1_29 + Dex3-1 teleop with neck forwarding and arm/hand/tactile recording."
    )
    parser.add_argument("--xr-repo", default="~/hckang/xr_teleoperate",
                        help="xr_teleoperate repository root")
    parser.add_argument("--neck-pose-ip", default="192.168.50.9")
    parser.add_argument("--neck-pose-port", type=int, default=5005)
    parser.add_argument("--neck-pose-rate", type=positive_rate, default=60.0)
    parser.add_argument("--neck-state-bind", default="0.0.0.0",
                        help="local address for neck state from PC2")
    parser.add_argument("--neck-state-port", type=int, default=5006,
                        help="UDP port for neck state from PC2")
    parser.add_argument("--frequency", type=positive_rate, default=30.0,
                        help="control and record frequency")
    parser.add_argument("--display-mode", choices=["immersive", "ego", "pass-through"],
                        default="immersive", help="legacy fork supports immersive only")
    parser.add_argument("--screen-y", type=float, default=-0.1)
    parser.add_argument("--cert")
    parser.add_argument("--key")
    parser.add_argument("--check", action="store_true", help="check installed APIs without hardware")
    parser.add_argument("--probe", action="store_true", help="receive camera/DDS/neck only; no motor commands")
    parser.add_argument("--probe-seconds", type=positive_rate, default=5.0)
    parser.add_argument("--smoke-test", action="store_true", help="write and validate a synthetic episode without hardware")
    parser.add_argument("--validate-episode", help="validate an existing episode directory or data.json")
    parser.add_argument("--img-server-ip", default="192.168.50.9")
    parser.add_argument("--clock-sync-port", type=int, default=DEFAULT_SYNC_PORT,
                        help="PC2 clock-sync UDP port (zed_teleimager_server.py)")
    parser.add_argument("--no-clock-sync", action="store_true",
                        help="do not probe the PC2 clock; timing ages are recorded as null")
    parser.add_argument("--network-interface", default=None,
                        help="DDS network interface, e.g. eth0")
    parser.add_argument("--motion", action="store_true",
                        help="robot is in motion (locomotion) mode; skip entering debug mode")
    parser.add_argument("--headless", action="store_true", help="disable rerun visualization")
    parser.add_argument("--record", action="store_true", help="enable episode recording")
    parser.add_argument("--task-dir", default="./utils/data/")
    parser.add_argument("--task-name", default="active_stereo_task")
    parser.add_argument("--task-goal", default="pick up cube.")
    parser.add_argument("--task-desc", default="task description")
    parser.add_argument("--task-steps", default="step1: do this; step2: do that;")
    args = parser.parse_args(argv)
    for name in ("neck_pose_port", "neck_state_port", "clock_sync_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    if args.display_mode != "immersive":
        parser.error("the installed shared-memory fork supports --display-mode immersive only")
    if not math.isfinite(args.screen_y):
        parser.error("--screen-y must be finite")
    # Resolve before run() changes cwd to the robot checkout's teleop directory.
    args.task_dir = str(Path(args.task_dir).expanduser().resolve())
    if args.validate_episode:
        args.validate_episode = str(Path(args.validate_episode).expanduser().resolve())
    return args


# ---------------------------------------------------------------------------
# Dex3-1 tactile
# ---------------------------------------------------------------------------

def press_sensors_to_dict(press_sensor_state: Any) -> dict[str, list]:
    """Convert a HandState_.press_sensor_state sequence into JSON-ready lists.

    Each PressSensorState_ carries 12 pressure and 12 temperature values plus
    a lost-packet counter. The number of sensors is taken from the message.
    """

    pressure: list[list[float]] = []
    temperature: list[list[float]] = []
    lost: list[int] = []
    for sensor in press_sensor_state:
        pressure.append([float(value) for value in sensor.pressure])
        temperature.append([float(value) for value in sensor.temperature])
        lost.append(int(sensor.lost))
    return {"pressure": pressure, "temperature": temperature, "lost": lost}


class Dex3TactileSubscriber:
    """Keep the latest Dex3-1 press-sensor data for both hands.

    Dex3_1_Controller only extracts motor q from rt/dex3/*/state. This adds a
    second reader on the same topics for the press sensors. Callback mode is
    used because unitree_sdk2py's Read() without a timeout blocks until a
    sample arrives, so polling both hands in one loop would let a silent hand
    stall the other.
    """

    def __init__(self, subscriber_factory: Any, message_type: Any) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Optional[dict]] = {"left_ee": None, "right_ee": None}
        self._received_ns: dict[str, Optional[int]] = {"left_ee": None, "right_ee": None}
        self._subscribers = []
        for key, topic in (("left_ee", TOPIC_DEX3_LEFT_STATE), ("right_ee", TOPIC_DEX3_RIGHT_STATE)):
            subscriber = subscriber_factory(topic, message_type)
            # queueLen > 0 runs the handler on the SDK's own reader thread,
            # not inside the DDS listener callback.
            subscriber.Init(self._make_handler(key), 10)
            self._subscribers.append(subscriber)

    def _make_handler(self, key: str):
        def handler(message: Any) -> None:
            try:
                sample = press_sensors_to_dict(message.press_sensor_state)
            except (AttributeError, TypeError, ValueError):
                # An exception here would end the SDK reader thread for good.
                return
            with self._lock:
                self._latest[key] = sample
                self._received_ns[key] = time.monotonic_ns()
        return handler

    def snapshot(self) -> dict[str, Optional[dict]]:
        """Latest tactile data per hand with its age, or None if never received."""

        with self._lock:
            now_ns = time.monotonic_ns()
            result: dict[str, Optional[dict]] = {}
            for key, sample in self._latest.items():
                received = self._received_ns[key]
                if sample is None or received is None:
                    result[key] = None
                else:
                    result[key] = {**sample, "received_host_monotonic_ns": received,
                                   "snapshot_host_monotonic_ns": now_ns,
                                   "age_ms": round((now_ns - received) / 1e6, 3)}
            return result

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.Close()


# ---------------------------------------------------------------------------
# Neck forwarding
# ---------------------------------------------------------------------------

class NeckPoseForwarder:
    """Send TeleVuer's raw OpenXR head matrix to PC2 at a fixed rate."""

    def __init__(self, televuer: Any, ip: str, port: int, rate_hz: float) -> None:
        self._televuer = televuer
        self._sender = HeadPoseUdpSender(ip, port)
        self._period = 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="quest-head-pose-udp", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        deadline = time.perf_counter()
        while not self._stop.is_set():
            try:
                received = self._televuer.head_received_ns.value
                # Stop sending on Quest disconnect so PC2's existing watchdog
                # can distinguish a stationary head from a stale connection.
                if received and time.monotonic_ns() - received < 250_000_000:
                    send_televuer_pose(self._sender, self._televuer)
            except (OSError, ValueError):
                # The shared matrix is all-zero until the first CAMERA_MOVE.
                pass
            deadline += self._period
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                deadline = time.perf_counter()
                continue
            self._stop.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._sender.close()


# ---------------------------------------------------------------------------
# Episode item construction
# ---------------------------------------------------------------------------

def _as_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def split_stereo(head_bgr: Any, combined_width: int) -> dict[str, Any]:
    """ZED left|right frame -> color_0 (left), color_1 (right)."""

    if (combined_width <= 0 or combined_width % 2 or head_bgr.ndim != 3
            or head_bgr.shape[1] != combined_width or head_bgr.shape[2] != 3):
        raise ValueError("expected a BGR left|right image with an even configured width")
    half = combined_width // 2
    return {"color_0": head_bgr[:, :half], "color_1": head_bgr[:, half:]}


def build_states_actions(
    arm_q: Any,
    arm_dq: Any,
    sol_q: Any,
    sol_tauff: Any,
    hand_state: Sequence[float],
    hand_action: Sequence[float],
    neck: Optional[tuple[dict, dict]] = None,
) -> tuple[dict, dict]:
    """Build xr_teleoperate-compatible states/actions for G1_29 + Dex3-1.

    ``neck`` is (states["neck"], actions["neck"]) from NeckStateReceiver.
    """

    arm_q, arm_dq = _as_list(arm_q), _as_list(arm_dq)
    sol_q, sol_tauff = _as_list(sol_q), _as_list(sol_tauff)
    hand_state, hand_action = _as_list(hand_state), _as_list(hand_action)
    if any(len(v) != 14 for v in (arm_q, arm_dq, sol_q, sol_tauff)):
        raise ValueError("G1 arm q/dq/action q/tau must each have 14 values")
    if len(hand_state) != 2 * DEX3_NUM_MOTORS or len(hand_action) != 2 * DEX3_NUM_MOTORS:
        raise ValueError("Dex3-1 state/action must have 14 values")
    if not all(math.isfinite(v) for values in
               (arm_q, arm_dq, sol_q, sol_tauff, hand_state, hand_action) for v in values):
        raise ValueError("joint data contains a non-finite value")

    half = len(arm_q) // 2
    n = DEX3_NUM_MOTORS
    states = {
        "left_arm": {"qpos": arm_q[:half], "qvel": arm_dq[:half], "torque": []},
        "right_arm": {"qpos": arm_q[half:], "qvel": arm_dq[half:], "torque": []},
        "left_ee": {"qpos": hand_state[:n], "qvel": [], "torque": []},
        "right_ee": {"qpos": hand_state[n:], "qvel": [], "torque": []},
        "body": {"qpos": []},
    }
    actions = {
        "left_arm": {"qpos": sol_q[:half], "qvel": [], "torque": sol_tauff[:half]},
        "right_arm": {"qpos": sol_q[half:], "qvel": [], "torque": sol_tauff[half:]},
        "left_ee": {"qpos": hand_action[:n], "qvel": [], "torque": []},
        "right_ee": {"qpos": hand_action[n:], "qvel": [], "torque": []},
        "body": {"qpos": []},
    }
    if neck is not None:
        states["neck"], actions["neck"] = neck
    return states, actions


# ---------------------------------------------------------------------------
# Episode timing summary
# ---------------------------------------------------------------------------

def summarize_ages(ages: Sequence[Optional[float]]) -> str:
    """One log line on how stale the recorded images were."""

    known = sorted(age for age in ages if age is not None)
    missing = len(ages) - len(known)
    if not known:
        return f"image age unknown for all {len(ages)} items (no PC2 stamp or clock sync)"
    median = known[len(known) // 2]
    p95 = known[min(len(known) - 1, int(0.95 * len(known)))]
    text = f"image age median {median:.1f} ms, p95 {p95:.1f} ms, max {known[-1]:.1f} ms"
    if missing:
        text += f", unknown for {missing}/{len(ages)} items"
    return text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

PART_DIMS = (("left_arm", 7), ("right_arm", 7), ("left_ee", 7),
             ("right_ee", 7), ("neck", 2))


def make_writer(writer_class, args, image_shape, synthetic=False):
    """Use the fork's actual constructor and report asynchronous write errors."""
    class CheckedWriter(writer_class):
        error = None

        def _process_item_data(self, item):
            try:
                super()._process_item_data(item)
                for relative in item["colors"].values():
                    path = Path(self.episode_dir) / relative
                    if not path.is_file() or path.stat().st_size == 0:
                        raise OSError(f"image was not written: {path}")
            except Exception as error:
                self.error = error
                self.info.setdefault("recording_errors", []).append(str(error))
                raise

        def _save_episode(self):
            try:
                super()._save_episode()
            except Exception as error:
                self.error = error
                # The legacy close() waits forever on is_available after a
                # failed JSON write. Release it and report the failure.
                self.need_save = False
                self.is_available = True

        def close(self):
            super().close()
            if self.error is not None:
                raise OSError(f"episode save failed: {self.error}") from self.error

    writer = CheckedWriter(
        task_dir=os.path.join(args.task_dir, args.task_name),
        frequency=args.frequency, image_size=[image_shape[1] // 2, image_shape[0]],
        rerun_log=not args.headless and not synthetic,
    )
    writer.text.update(goal=args.task_goal, desc=args.task_desc, steps=args.task_steps)
    writer.info.update(
        recording_source="synthetic" if synthetic else "robot",
        packages=installed_versions(),
        schema="g1_active_stereo_v1",
        qpos_layout=[{"part": part, "size": size} for part, size in PART_DIMS],
        qpos_units="radians", qvel_units="radians/second",
        neck_convention="[yaw right-positive, pitch up-positive], relative to center counts",
        camera_order={"color_0": "ZED left", "color_1": "ZED right"},
        synchronization="latest available observations, sampled on Host; per-source times in states.timing",
        arm_action="IK q target and feed-forward tau passed to ctrl_dual_arm; controller may slew-limit q",
        hand_action="latest asynchronous retargeted command, paired with hand state by controller lock",
    )
    # Names come from this checkout, including the different left/right Dex3
    # index/middle ordering. Never assume both hands have the same joint order.
    from teleop.robot_control.robot_arm import G1_29_JointArmIndex
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Left_JointIndex, Dex3_1_Right_JointIndex
    names = [m.name for m in G1_29_JointArmIndex]
    writer.info["joint_names"].update(
        left_arm=names[:7], right_arm=names[7:], neck=["yaw", "pitch"],
        left_ee=[m.name for m in Dex3_1_Left_JointIndex],
        right_ee=[m.name for m in Dex3_1_Right_JointIndex],
    )
    return writer


def complete_observation(frame, image_age_ms, neck_latest, touch, max_age_ms=500.0):
    """Never label missing/stale sensor values as a complete training sample."""
    packet, receive_age = neck_latest
    return bool(
        frame is not None and frame.bgr is not None
        and image_age_ms is not None and image_age_ms <= max_age_ms
        and packet is not None and packet.present_deg is not None
        and packet.torque_enabled and receive_age is not None and receive_age <= max_age_ms
        and all(touch.get(side) and touch[side].get("pressure")
                and touch[side].get("age_ms", math.inf) <= max_age_ms
                for side in ("left_ee", "right_ee"))
    )


def validate_episode(path) -> dict:
    """Verify every saved stereo image and the 30-DoF state/action layout."""
    import cv2
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "data.json"
    data = json.loads(path.read_text())
    errors = list(data["info"].get("recording_errors", []))
    unknown_age = 0
    previous_ns = -1
    image_size = data["info"]["image"]
    for i, item in enumerate(data["data"]):
        def fail(message):
            if len(errors) < 30:
                errors.append(f"item {i}: {message}")
        for eye in ("color_0", "color_1"):
            relative = item.get("colors", {}).get(eye)
            image = cv2.imread(str(path.parent / relative)) if relative else None
            if image is None or image.shape[:2] != (image_size["height"], image_size["width"]):
                fail(f"missing or incorrect image {eye}")
        for kind in ("states", "actions"):
            for part, size in PART_DIMS:
                q = item.get(kind, {}).get(part, {}).get("qpos", [])
                if len(q) != size or not all(math.isfinite(v) for v in q):
                    fail(f"{kind}.{part}.qpos must have {size} finite values")
        for part in ("left_arm", "right_arm"):
            for kind, field in (("states", "qvel"), ("actions", "torque")):
                values = item.get(kind, {}).get(part, {}).get(field, [])
                if len(values) != 7 or not all(math.isfinite(v) for v in values):
                    fail(f"invalid {kind}.{part}.{field}")
        for side in ("left_ee", "right_ee"):
            tactile = item.get("tactiles", {}).get(side)
            if not tactile or not tactile.get("pressure"):
                fail(f"missing tactile {side}")
            else:
                for field in ("pressure", "temperature"):
                    rows = tactile.get(field, [])
                    if (len(rows) != len(tactile["pressure"])
                            or any(len(row) != 12 or not all(math.isfinite(v) for v in row) for row in rows)):
                        fail(f"invalid tactile {side}.{field}")
        timing = item["states"].get("timing", {})
        now = timing.get("host_monotonic_ns", -1)
        if now <= previous_ns or "host_wall_ns" not in timing:
            fail("missing or non-increasing Host timing")
        previous_ns = now
        if timing.get("image_age_ms") is None:
            unknown_age += 1
    if not data["data"]:
        errors.append("episode is empty")
    return {"path": str(path), "source": data["info"].get("recording_source"),
            "items": len(data["data"]), "qpos_dim": 30,
            "unknown_image_age_items": unknown_age,
            "image_latency_available": not unknown_age and bool(data["data"]),
            "errors": errors, "ok": not errors}

class KeyState:
    def __init__(self) -> None:
        self.start = False
        self.stop = False
        self.record_toggle = False

    def on_press(self, key: str) -> None:
        if key == "r":
            self.start = True
        elif key == "q":
            self.start = False
            self.stop = True
        elif key == "s" and self.start:
            self.record_toggle = True


def smoke_test(args, writer_class) -> int:
    """Exercise the real fork's JPEG/JSON writer; synthetic data is labeled."""
    import numpy as np
    from types import SimpleNamespace
    try:
        from .frame_timing import ClockEstimate, FrameStamp, StampedFrame
    except ImportError:
        from frame_timing import ClockEstimate, FrameStamp, StampedFrame

    shape = (48, 128, 3)
    writer = make_writer(writer_class, args, shape, synthetic=True)
    try:
        if not writer.create_episode():
            raise RuntimeError("cannot create smoke episode")
        for index in range(3):
            bgr = np.zeros(shape, dtype=np.uint8)
            bgr[:, :64] = (0, 0, 200)
            bgr[:, 64:] = (200, 0, 0)
            sample_ns = time.monotonic_ns()
            joint = {"qpos": [0.1, -0.2], "qvel": [], "torque": []}
            states, actions = build_states_actions(
                range(14), range(14), range(14), range(14),
                range(14), range(14), (dict(joint), dict(joint)))
            frame = StampedFrame(bgr, FrameStamp(index, sample_ns - 10_000_000, None))
            states["timing"] = frame_timing_entry(sample_ns, frame, ClockEstimate(0, 0))
            states["timing"]["host_wall_ns"] = time.time_ns()
            sensor = SimpleNamespace(pressure=list(range(12)), temperature=[30.] * 12, lost=0)
            touch = press_sensors_to_dict([sensor])
            writer.add_item(colors=split_stereo(bgr, shape[1]), depths={},
                            states=states, actions=actions,
                            tactiles={"left_ee": dict(touch), "right_ee": dict(touch)})
        writer.save_episode()
    finally:
        writer.close()
    result = validate_episode(writer.json_path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def probe_inputs(args, image_client_class, channel_initialize, subscriber_factory,
                 hand_message_type, low_message_type) -> int:
    """Bounded receive-only probe: no controllers, motion switcher or head UDP."""
    client = clock = neck = tactile = low = None
    low_received = {"host_monotonic_ns": None}
    try:
        client = image_client_class(host=args.img_server_ip, request_bgr=False)
        config = client.get_cam_config()["head_camera"]
        if not config["binocular"] or not config["enable_zmq"]:
            raise RuntimeError("ZED binocular enable_zmq is required")
        reader = StampedFrameReader(client.get_head_frame)
        channel_initialize(0, networkInterface=args.network_interface)
        tactile = Dex3TactileSubscriber(subscriber_factory, hand_message_type)
        low = subscriber_factory("rt/lowstate", low_message_type)
        low.Init(lambda msg: low_received.update(host_monotonic_ns=time.monotonic_ns()), 10)
        neck = NeckStateReceiver(args.neck_state_bind, args.neck_state_port)
        if not args.no_clock_sync:
            clock = ClockSyncClient(args.img_server_ip, args.clock_sync_port)
        deadline = time.monotonic() + args.probe_seconds
        frame = None
        while time.monotonic() < deadline:
            frame = reader.read()
            time.sleep(.05)
        packet, age = neck.latest()
        touch = tactile.snapshot()
        now_ns = time.monotonic_ns()
        shape = list(frame.bgr.shape) if frame is not None and frame.bgr is not None else None
        result = {
            "mode": "receive_only", "camera_config": config, "image_shape": shape,
            "timing": frame_timing_entry(now_ns, frame, clock.estimate() if clock else None, packet),
            "lowstate_received_host_monotonic_ns": low_received["host_monotonic_ns"],
            "tactile_sensor_counts": {side: len(value["pressure"]) if value else 0
                                      for side, value in touch.items()},
            "neck_received": packet is not None, "neck_receive_age_ms": age,
            "neck_encoder_valid": packet is not None and packet.present_deg is not None,
        }
        result["ok"] = bool(shape and low_received["host_monotonic_ns"]
                            and all(value and value["pressure"] for value in touch.values())
                            and result["neck_encoder_valid"])
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    finally:
        for resource in (low, tactile, neck, clock, client):
            if resource is not None:
                resource.Close() if resource is low else resource.close()


def run(args: argparse.Namespace) -> int:
    if args.validate_episode:
        result = validate_episode(args.validate_episode)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    repository = Path(args.xr_repo).expanduser().resolve()
    teleop_dir = repository / "teleop"
    if not (teleop_dir / "teleop_hand_and_arm.py").is_file():
        raise FileNotFoundError(f"xr_teleoperate teleop directory not found: {teleop_dir}")
    sys.path.insert(0, str(repository))
    # Retargeting and IK load ../assets relative to teleop, just as upstream.
    os.chdir(teleop_dir)

    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    logger = logging_mp.getLogger(__name__)
    from multiprocessing import Array, Lock
    from sshkeyboard import listen_keyboard, stop_listening
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_
    from televuer import TeleVuerWrapper
    from teleimager.image_client import ImageClient
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
    from teleop.utils.episode_writer import EpisodeWriter
    from teleop.utils.motion_switcher import MotionSwitcher

    apis = check_legacy_api(TeleVuerWrapper, Dex3_1_Controller, EpisodeWriter, ImageClient)
    if args.check:
        cert, key = resolve_tls(args.cert, args.key)
        print(json.dumps({"packages": installed_versions(), "apis": apis,
                          "tls_cert": str(cert), "tls_key": str(key),
                          "task_dir": args.task_dir, "hardware_initialized": False}, indent=2))
        return 0
    if args.smoke_test:
        return smoke_test(args, EpisodeWriter)
    if args.probe:
        return probe_inputs(args, ImageClient, ChannelFactoryInitialize, ChannelSubscriber,
                            HandState_, LowState_)
    cert, key = resolve_tls(args.cert, args.key)
    if enabled():
        install_televuer_hook(sys.modules[TeleVuerWrapper.__module__].TeleVuer)
    if args.network_interface:
        os.environ["UNITREE_NETWORK_INTERFACE"] = args.network_interface

    keys = KeyState()
    img_client = clock_sync = xr = neck = neck_state = tactile = None
    arm_ctrl = recorder = hand_session = keyboard_thread = None
    record_running = False
    cleanup_errors = []
    try:
        # Start image/XR first. Arm and hand controllers are constructed only
        # after the operator presses r and both hands are tracked.
        img_client = ImageClient(host=args.img_server_ip, request_bgr=False)
        head_frames = StampedFrameReader(img_client.get_head_frame)
        head_config = img_client.get_cam_config()["head_camera"]
        if not head_config["binocular"] or not head_config["enable_zmq"]:
            raise RuntimeError("ZED binocular camera with enable_zmq=true is required")
        shape = tuple(head_config["image_shape"][:2]) + (3,)
        if shape[1] % 2:
            raise ValueError("stereo combined width must be even")
        xr = LegacyStereoXR(TeleVuerWrapper, shape, cert, key,
                            screen_y=args.screen_y, fps=args.frequency)
        neck_state = NeckStateReceiver(args.neck_state_bind, args.neck_state_port)
        if not args.no_clock_sync:
            clock_sync = ClockSyncClient(args.img_server_ip, args.clock_sync_port)
        if args.record:
            recorder = make_writer(EpisodeWriter, args, shape)

        keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": keys.on_press, "until": None, "sequential": False}, daemon=True)
        keyboard_thread.start()
        logger.info("Quest: https://<Host-IP>:8012/?ws=wss://<Host-IP>:8012 ; "
                    "press [r] with both hands tracked to start, [s] record/save, [q] quit.")
        while not keys.stop:
            frame = head_frames.read()
            if frame.bgr is not None:
                xr.render(frame.bgr)
            xr.get_motion_state_data()  # also detect an early Vuer server failure
            if keys.start and xr.tracking_ready() and frame.bgr is not None:
                break
            time.sleep(.033)
        if keys.stop:
            return 0

        ChannelFactoryInitialize(0, networkInterface=args.network_interface)
        if not args.motion:
            status, _ = MotionSwitcher().Enter_Debug_Mode()
            if status != 0:
                raise RuntimeError(f"Enter_Debug_Mode failed: {status}")
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=False)
        left_hand_pos_array, right_hand_pos_array = Array("d", 75), Array("d", 75)
        tele_data = xr.get_motion_state_data()
        left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
        hand_lock = Lock()
        hand_states = Array("d", 14, lock=False)
        hand_actions = Array("d", 14, lock=False)
        hand_session = LegacyHandSession(Dex3_1_Controller,
            left_hand_pos_array, right_hand_pos_array, hand_lock, hand_states, hand_actions)
        tactile = Dex3TactileSubscriber(ChannelSubscriber, HandState_)
        neck = NeckPoseForwarder(xr.tvuer, args.neck_pose_ip,
                                 args.neck_pose_port, args.neck_pose_rate)
        arm_ctrl.speed_gradual_max()
        image_ages = []
        period = 1.0 / args.frequency
        next_tick = time.monotonic()
        last_warning = 0.0
        previous_bgr = None
        image_observed_ns = 0
        while not keys.stop:
            frame = head_frames.read()
            image_read_ns = time.monotonic_ns()
            if frame.bgr is not None and frame.bgr is not previous_bgr:
                previous_bgr = frame.bgr
                image_observed_ns = image_read_ns
            if frame.bgr is not None:
                xr.render(frame.bgr)
            if recorder is not None and recorder.error is not None:
                raise RuntimeError(f"episode writer failed: {recorder.error}")
            if recorder is not None and keys.record_toggle:
                keys.record_toggle = False
                if record_running:
                    record_running = False
                    recorder.save_episode()
                    logger.info(f"Episode timing: {summarize_ages(image_ages)}")
                else:
                    record_running = recorder.create_episode()
                    image_ages = []
                    if not record_running:
                        logger.warning("Writer still saving; press s again when save completes.")

            tele_data = xr.get_motion_state_data()
            tracking = xr.tracking_ready()
            observation_complete = False
            if tracking:
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                # Read a single lowstate buffer for q and dq so they correspond
                # to the same DDS observation in this fork.
                from teleop.robot_control.robot_arm import G1_29_JointArmIndex
                lowstate = arm_ctrl.lowstate_buffer.GetData()
                arm_q = [lowstate.motor_state[j].q for j in G1_29_JointArmIndex]
                arm_dq = [lowstate.motor_state[j].dq for j in G1_29_JointArmIndex]
                arm_read_ns = time.monotonic_ns()
                with hand_lock:
                    hand_state, hand_action = list(hand_states[:]), list(hand_actions[:])
                hand_read_ns = time.monotonic_ns()
                neck_latest = neck_state.latest()
                neck_read_ns = time.monotonic_ns()
                touch = tactile.snapshot()
                sample_ns, wall_ns = time.monotonic_ns(), time.time_ns()
                # Arrays are required by Pinocchio/CasADi in the legacy IK.
                import numpy as np
                sol_q, sol_tauff = arm_ik.solve_ik(
                    tele_data.left_arm_pose, tele_data.right_arm_pose,
                    np.asarray(arm_q), np.asarray(arm_dq))
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                action_ns = time.monotonic_ns()
                if record_running:
                    states, actions = build_states_actions(
                        arm_q, arm_dq, sol_q, sol_tauff, hand_state, hand_action,
                        neck_record_entries(*neck_latest))
                    timing = frame_timing_entry(sample_ns, frame,
                        clock_sync.estimate() if clock_sync else None, neck_latest[0])
                    timing.update(host_wall_ns=wall_ns, image_read_host_monotonic_ns=image_read_ns,
                        image_observed_host_monotonic_ns=image_observed_ns,
                        arm_read_host_monotonic_ns=arm_read_ns,
                        hand_read_host_monotonic_ns=hand_read_ns,
                        neck_read_host_monotonic_ns=neck_read_ns,
                        action_host_monotonic_ns=action_ns,
                        xr_head_received_host_monotonic_ns=xr.tvuer.head_received_ns.value,
                        xr_hand_received_host_monotonic_ns=xr.tvuer.hand_received_ns.value)
                    states["timing"] = timing
                    # Without clock sync we can still bound how long a cached
                    # JPEG has been reused, while capture latency stays null.
                    freshness = timing["image_age_ms"]
                    if freshness is None:
                        freshness = (sample_ns - image_observed_ns) / 1e6
                    observation_complete = complete_observation(frame, freshness, neck_latest, touch)
                    if observation_complete:
                        image_ages.append(timing["image_age_ms"])
                        recorder.add_item(colors=split_stereo(frame.bgr, shape[1]), depths={},
                            states=states, actions=actions, tactiles=touch)
                    if recorder.item_data_queue.qsize() > 2 * args.frequency:
                        raise RuntimeError("episode disk writer is over two seconds behind")
            if record_running and not observation_complete:
                if time.monotonic() - last_warning > 2:
                    logger.warning("Skipping incomplete/stale sample: check Quest tracking, ZED, "
                                   "neck encoder/torque and tactile (freshness limit 500 ms).")
                    last_warning = time.monotonic()
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Interrupted; saving episode and stopping.")
    finally:
        def attempt(label, action):
            try:
                action()
            except Exception as error:
                cleanup_errors.append(f"{label}: {error}")
                logger.error(cleanup_errors[-1])
        if neck is not None:
            attempt("close neck forwarder", neck.close)
        if hand_session is not None:
            attempt("close hand controller", hand_session.close)
        if arm_ctrl is not None:
            attempt("move arms home", arm_ctrl.ctrl_dual_arm_go_home)
        if keyboard_thread is not None:
            attempt("stop keyboard", stop_listening)
        for label, resource in (("tactile", tactile), ("neck state", neck_state),
                                ("clock sync", clock_sync), ("images", img_client),
                                ("XR", xr), ("recorder", recorder)):
            if resource is not None:
                attempt(f"close {label}", resource.close)
        if recorder is not None and hasattr(recorder, "json_path"):
            def validate_saved():
                result = validate_episode(recorder.json_path)
                logger.info("Saved episode validation: " + json.dumps(result))
                if not result["ok"]:
                    cleanup_errors.append("saved episode failed validation")
            attempt("validate saved episode", validate_saved)
            if recorder.error is not None:
                cleanup_errors.append(str(recorder.error))
    return 1 if cleanup_errors else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

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
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional, Sequence

try:
    from .frame_timing import (DEFAULT_SYNC_PORT, ClockSyncClient, StampedFrameReader,
                               frame_timing_entry)
    from .head_pose_udp import HeadPoseUdpSender
    from .neck_state_receiver import NeckStateReceiver, neck_record_entries
except ImportError:
    from frame_timing import (  # type: ignore[no-redef]
        DEFAULT_SYNC_PORT, ClockSyncClient, StampedFrameReader, frame_timing_entry)
    from head_pose_udp import HeadPoseUdpSender  # type: ignore[no-redef]
    from neck_state_receiver import (  # type: ignore[no-redef]
        NeckStateReceiver, neck_record_entries)


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
    parser.add_argument("--neck-pose-ip", default="192.168.123.164")
    parser.add_argument("--neck-pose-port", type=int, default=5005)
    parser.add_argument("--neck-pose-rate", type=positive_rate, default=60.0)
    parser.add_argument("--neck-state-bind", default="0.0.0.0",
                        help="local address for neck state from PC2")
    parser.add_argument("--neck-state-port", type=int, default=5006,
                        help="UDP port for neck state from PC2")
    parser.add_argument("--frequency", type=positive_rate, default=30.0,
                        help="control and record frequency")
    parser.add_argument("--display-mode", choices=["immersive", "ego", "pass-through"],
                        default="immersive")
    parser.add_argument("--img-server-ip", default="192.168.123.164")
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

        now_ns = time.monotonic_ns()
        with self._lock:
            result: dict[str, Optional[dict]] = {}
            for key, sample in self._latest.items():
                received = self._received_ns[key]
                if sample is None or received is None:
                    result[key] = None
                else:
                    result[key] = {**sample, "age_ms": round((now_ns - received) / 1e6, 3)}
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
                self._sender.send_openxr_matrix(self._televuer.head_pose)
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
    if len(arm_q) != len(sol_q) or len(arm_q) % 2:
        raise ValueError(f"unexpected arm sizes: q={len(arm_q)} sol_q={len(sol_q)}")
    if len(hand_state) != 2 * DEX3_NUM_MOTORS or len(hand_action) != 2 * DEX3_NUM_MOTORS:
        raise ValueError("Dex3-1 state/action must have 14 values")

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


def run(args: argparse.Namespace) -> int:
    repository = Path(args.xr_repo).expanduser().resolve()
    teleop_dir = repository / "teleop"
    if not (teleop_dir / "teleop_hand_and_arm.py").is_file():
        raise FileNotFoundError(f"xr_teleoperate teleop directory not found: {teleop_dir}")
    # Same import layout as teleop_hand_and_arm.py; its hand retargeting loads
    # ../assets relative to the teleop directory.
    sys.path.insert(0, str(repository))
    os.chdir(teleop_dir)

    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    logger = logging_mp.getLogger(__name__)

    from multiprocessing import Array, Lock, Value
    from sshkeyboard import listen_keyboard, stop_listening
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    from televuer import TeleVuerWrapper
    from teleimager.image_client import ImageClient
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
    from teleop.utils.episode_writer import EpisodeWriter
    from teleop.utils.motion_switcher import MotionSwitcher

    keys = KeyState()
    img_client = None
    clock_sync = None
    tv_wrapper = None
    neck = None
    neck_state = None
    tactile = None
    arm_ctrl = None
    recorder = None
    keyboard_thread = None
    try:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": keys.on_press, "until": None, "sequential": False},
            daemon=True,
        )
        keyboard_thread.start()

        # StampedFrameReader decodes the JPEG itself so image and stamp match.
        img_client = ImageClient(host=args.img_server_ip, request_bgr=False)
        head_frames = StampedFrameReader(img_client.get_head_frame)
        if not args.no_clock_sync:
            clock_sync = ClockSyncClient(args.img_server_ip, args.clock_sync_port)
        camera_config = img_client.get_cam_config()
        head_config = camera_config["head_camera"]
        if not head_config["binocular"]:
            raise RuntimeError("expected the ZED binocular head camera (cam_config_zed.yaml)")
        combined_width = int(head_config["image_shape"][1])
        xr_need_local_img = not (args.display_mode == "pass-through" or head_config["enable_webrtc"])

        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=True,
            binocular=True,
            img_shape=head_config["image_shape"],
            display_mode=args.display_mode,
            zmq=head_config["enable_zmq"],
            webrtc=head_config["enable_webrtc"],
            webrtc_url=f"https://{args.img_server_ip}:{head_config['webrtc_port']}/offer",
            arm_reference_mode="head_yaw",
        )
        neck = NeckPoseForwarder(tv_wrapper.tvuer, args.neck_pose_ip,
                                 args.neck_pose_port, args.neck_pose_rate)
        neck_state = NeckStateReceiver(args.neck_state_bind, args.neck_state_port)

        if not args.motion:
            status, _ = MotionSwitcher().Enter_Debug_Mode()
            logger.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        xr_motion_data_ready = Value("b", False, lock=True)
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=False)

        left_hand_pos_array = Array("d", 75, lock=True)
        right_hand_pos_array = Array("d", 75, lock=True)
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array("d", 2 * DEX3_NUM_MOTORS, lock=False)
        dual_hand_action_array = Array("d", 2 * DEX3_NUM_MOTORS, lock=False)
        Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                          dual_hand_state_array, dual_hand_action_array,
                          simulation_mode=False, xr_motion_data_ready_in=xr_motion_data_ready)
        tactile = Dex3TactileSubscriber(ChannelSubscriber, HandState_)

        if args.record:
            per_eye_height = int(head_config["image_shape"][0])
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                image_size=[combined_width // 2, per_eye_height],
                rerun_log=not args.headless,
            )

        logger.info("Press [r] to start following, "
                    + ("[s] to start/save an episode, " if args.record else "")
                    + "[q] to quit.")
        while not keys.start and not keys.stop:
            time.sleep(0.033)
            if head_config["enable_zmq"] and xr_need_local_img:
                head_bgr = head_frames.read().bgr
                if head_bgr is not None:
                    tv_wrapper.render_to_xr(head_bgr)

        logger.info("Tracking started.")
        record_running = False
        image_ages: list[Optional[float]] = []
        period = 1.0 / args.frequency
        # Monotonic fixed-rate schedule: immune to wall-clock (NTP) steps, and
        # sleep overshoot does not accumulate into a lower recording rate.
        next_tick = time.monotonic()
        while not keys.stop:
            frame = None
            head_bgr = None
            if head_config["enable_zmq"] and (args.record or xr_need_local_img):
                frame = head_frames.read()
                head_bgr = frame.bgr
                if xr_need_local_img and head_bgr is not None:
                    tv_wrapper.render_to_xr(head_bgr)

            if recorder is not None and keys.record_toggle:
                keys.record_toggle = False
                if not record_running:
                    record_running = recorder.create_episode()
                    image_ages = []
                    if not record_running:
                        logger.error("Failed to create episode; recording not started.")
                    elif clock_sync is not None and clock_sync.estimate() is None:
                        logger.warning("No PC2 clock sync yet; image/neck ages will be null. "
                                       "Is zed_teleimager_server.py running with UDP "
                                       f"{args.clock_sync_port} reachable?")
                else:
                    record_running = False
                    recorder.save_episode()
                    logger.info(f"Episode timing: {summarize_ages(image_ages)}")

            tele_data = tv_wrapper.get_tele_data()
            with left_hand_pos_array.get_lock():
                left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
            with right_hand_pos_array.get_lock():
                right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready

            arm_q = arm_ctrl.get_current_dual_arm_q()
            arm_dq = arm_ctrl.get_current_dual_arm_dq()
            sample_ns = time.monotonic_ns()
            sol_q, sol_tauff = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose,
                                               arm_q, arm_dq)
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            if record_running:
                with dual_hand_data_lock:
                    hand_state = list(dual_hand_state_array[:])
                    hand_action = list(dual_hand_action_array[:])
                neck_latest = neck_state.latest()
                states, actions = build_states_actions(arm_q, arm_dq, sol_q, sol_tauff,
                                                       hand_state, hand_action,
                                                       neck_record_entries(*neck_latest))
                states["timing"] = frame_timing_entry(
                    sample_ns, frame,
                    clock_sync.estimate() if clock_sync is not None else None,
                    neck_latest[0])
                image_ages.append(states["timing"]["image_age_ms"])
                if head_bgr is not None:
                    colors = split_stereo(head_bgr, combined_width)
                else:
                    colors = {}
                    logger.warning("Head image is None!")
                recorder.add_item(colors=colors, depths={}, states=states,
                                  actions=actions, tactiles=tactile.snapshot())

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()  # overran: restart the schedule, no burst

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exiting.")
    finally:
        def attempt(label: str, action) -> None:
            try:
                action()
            except Exception as error:  # keep shutting down the rest
                logger.error(f"Failed to {label}: {error}")

        if arm_ctrl is not None:
            attempt("move arms home", arm_ctrl.ctrl_dual_arm_go_home)
        if keyboard_thread is not None:
            attempt("stop keyboard listener", stop_listening)
        if tactile is not None:
            attempt("close tactile subscriber", tactile.close)
        if neck is not None:
            attempt("close neck forwarder", neck.close)
        if neck_state is not None:
            attempt("close neck state receiver", neck_state.close)
        if clock_sync is not None:
            attempt("close clock sync", clock_sync.close)
        if img_client is not None:
            attempt("close image client", img_client.close)
        if tv_wrapper is not None:
            attempt("close televuer", tv_wrapper.close)
        if recorder is not None:
            attempt("close recorder", recorder.close)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

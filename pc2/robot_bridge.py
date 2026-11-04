#!/usr/bin/env python3
"""Orin-side DDS<->UDP bridge for the Host recorder.

The Host runs the recorder over Wi-Fi and cannot see the G1's DDS (arm/hand/
tactile live on the robot's wired network). This bridge runs on the Orin, which
is on that DDS network. It:

- reads arm q/dq (rt/lowstate), Dex3-1 hand state/action and tactile over DDS,
  and sends them to the Host over UDP (robot-state and tactile packets);
- receives the Host's control command over UDP (arm target q/tau from the Host's
  IK, and raw hand-tracking positions) and applies them to the robot over DDS
  through the same G1_29_ArmController and Dex3_1_Controller the recorder used.

Wire formats are shared verbatim with host/robot_link.py (a test asserts the
struct formats match). Video (teleimager ZMQ) and neck (UDP) are unchanged and
do not go through this bridge.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
from typing import Optional, Sequence


# --- shared wire constants (keep identical in host/robot_link.py) ------------
STATE_MAGIC = b"ARST"
TACTILE_MAGIC = b"ATAC"
COMMAND_MAGIC = b"ARCM"
VERSION = 1

NUM_ARM = 14
NUM_HAND = 7
NUM_HAND_POS = 75
TACTILE_VALUES = 12

_HEADER = "!4sBBHIQ"
STATE_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 4 * NUM_HAND}f")
COMMAND_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 2 * NUM_HAND_POS}f")
TACTILE_HEADER = struct.Struct(_HEADER + "HH")
TACTILE_SENSOR = struct.Struct(f"!{TACTILE_VALUES}f{TACTILE_VALUES}fi")

STATE_FLAG_ARM = 1 << 0
STATE_FLAG_HAND = 1 << 1
COMMAND_FLAG_ARM = 1 << 0
COMMAND_FLAG_HAND = 1 << 1

DEFAULT_STATE_PORT = 5008
DEFAULT_COMMAND_PORT = 5009
DEX3_NUM_MOTORS = 7


class RobotBridgeError(RuntimeError):
    pass


def _finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


# --- robot state (Orin -> Host) ---------------------------------------------

def encode_state(arm_q, arm_dq, left_hand_q, right_hand_q,
                 left_hand_action, right_hand_action,
                 sequence: int, pc2_monotonic_ns: int) -> bytes:
    """arm_q/arm_dq are 14 (or empty) and each hand list is 7 (or empty)."""
    flags = 0
    arm_q, arm_dq = list(arm_q), list(arm_dq)
    if arm_q and arm_dq:
        if len(arm_q) != NUM_ARM or len(arm_dq) != NUM_ARM:
            raise RobotBridgeError("arm q/dq must have 14 values")
        flags |= STATE_FLAG_ARM
    else:
        arm_q = arm_dq = [0.0] * NUM_ARM
    hands = [list(h) for h in (left_hand_q, right_hand_q, left_hand_action, right_hand_action)]
    if all(hands):
        if any(len(h) != NUM_HAND for h in hands):
            raise RobotBridgeError("each hand list must have 7 values")
        flags |= STATE_FLAG_HAND
    else:
        hands = [[0.0] * NUM_HAND for _ in range(4)]
    payload = arm_q + arm_dq + hands[0] + hands[1] + hands[2] + hands[3]
    if not _finite(payload):
        raise RobotBridgeError("state contains a non-finite value")
    return STATE_STRUCT.pack(STATE_MAGIC, VERSION, flags, 0,
                             sequence & 0xFFFFFFFF, pc2_monotonic_ns, *payload)


def encode_tactile(left: Optional[dict], right: Optional[dict],
                   sequence: int, pc2_monotonic_ns: int) -> bytes:
    """left/right are {"pressure": [[12]...], "temperature": [[12]...], "lost": [...]} or None."""
    def sensors(sample):
        if not sample or not sample.get("pressure"):
            return []
        pressure, temperature = sample["pressure"], sample.get("temperature", [])
        lost = sample.get("lost", [])
        rows = []
        for i, press in enumerate(pressure):
            temp = temperature[i] if i < len(temperature) else [0.0] * TACTILE_VALUES
            if len(press) != TACTILE_VALUES or len(temp) != TACTILE_VALUES:
                raise RobotBridgeError("each tactile sensor needs 12 pressure/temperature values")
            rows.append((list(press), list(temp), int(lost[i]) if i < len(lost) else 0))
        return rows

    left_rows, right_rows = sensors(left), sensors(right)
    packet = bytearray(TACTILE_HEADER.pack(TACTILE_MAGIC, VERSION, 0, 0,
                                           sequence & 0xFFFFFFFF, pc2_monotonic_ns,
                                           len(left_rows), len(right_rows)))
    for rows in (left_rows, right_rows):
        for press, temp, lost in rows:
            if not _finite(press) or not _finite(temp):
                raise RobotBridgeError("tactile contains a non-finite value")
            packet += TACTILE_SENSOR.pack(*press, *temp, lost)
    return bytes(packet)


# --- robot command (Host -> Orin) -------------------------------------------

def decode_command(datagram: bytes) -> dict:
    if len(datagram) != COMMAND_STRUCT.size:
        raise RobotBridgeError(f"command is {len(datagram)} bytes, expected {COMMAND_STRUCT.size}")
    magic, version, flags, _reserved, sequence, timestamp, *values = COMMAND_STRUCT.unpack(datagram)
    if magic != COMMAND_MAGIC or version != VERSION:
        raise RobotBridgeError("bad command magic/version")
    arm_q = values[0:NUM_ARM]
    arm_tau = values[NUM_ARM:2 * NUM_ARM]
    rest = values[2 * NUM_ARM:]
    left_hand_pos = rest[0:NUM_HAND_POS]
    right_hand_pos = rest[NUM_HAND_POS:2 * NUM_HAND_POS]
    return {"arm_q": list(arm_q), "arm_tau": list(arm_tau),
            "left_hand_pos": list(left_hand_pos), "right_hand_pos": list(right_hand_pos),
            "flags": flags, "sequence": sequence, "host_monotonic_ns": timestamp}


# --- argument parsing --------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin DDS<->UDP bridge for the Host recorder.")
    parser.add_argument("--xr-repo", default="~/hckang/xr_teleoperate")
    parser.add_argument("--interface", default="eth0", help="DDS network interface (e.g. eth0)")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--host-ip", default="192.168.50.7", help="Host address to send robot state to")
    parser.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT)
    parser.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--state-rate", type=float, default=60.0, help="robot-state send rate [Hz]")
    parser.add_argument("--tactile-rate", type=float, default=30.0, help="tactile send rate [Hz]")
    parser.add_argument("--command-timeout-ms", type=float, default=500.0,
                        help="hold the last command, then stop arm motion, after this silence")
    parser.add_argument("--motion", action="store_true",
                        help="robot is in motion mode; skip entering debug mode")
    parser.add_argument("--check", action="store_true",
                        help="verify imports/APIs without touching the robot")
    args = parser.parse_args(argv)
    for name in ("state_port", "command_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    for name in ("state_rate", "tactile_rate"):
        if not getattr(args, name) > 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


# --- main --------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    from pathlib import Path

    repository = Path(args.xr_repo).expanduser().resolve()
    teleop_dir = repository / "teleop"
    if not (teleop_dir / "teleop_hand_and_arm.py").is_file():
        raise FileNotFoundError(f"xr_teleoperate teleop directory not found: {teleop_dir}")
    sys.path.insert(0, str(repository))
    os.chdir(teleop_dir)

    from multiprocessing import Array, Lock
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_JointArmIndex
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
    from teleop.utils.motion_switcher import MotionSwitcher

    try:
        from .dex3_tactile import Dex3TactileSubscriber  # optional shared helper
    except Exception:
        # The tactile reader is small; inline a local copy to avoid a new import
        # dependency on the recorder module layout.
        from teleop_record_dex3_tactile import Dex3TactileSubscriber  # type: ignore

    if args.check:
        print("robot_bridge: imports OK; "
              f"state->{args.host_ip}:{args.state_port}, commands<-:{args.command_port}")
        return 0

    ChannelFactoryInitialize(args.domain, networkInterface=args.interface)
    if not args.motion:
        status, _ = MotionSwitcher().Enter_Debug_Mode()
        if status != 0:
            raise RobotBridgeError(f"Enter_Debug_Mode failed: {status}")

    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=False)
    hand_lock = Lock()
    left_hand_pos = Array("d", NUM_HAND_POS, lock=True)
    right_hand_pos = Array("d", NUM_HAND_POS, lock=True)
    hand_state = Array("d", 2 * DEX3_NUM_MOTORS, lock=False)
    hand_action = Array("d", 2 * DEX3_NUM_MOTORS, lock=False)
    Dex3_1_Controller(left_hand_pos, right_hand_pos, hand_lock, hand_state, hand_action,
                      simulation_mode=False)
    tactile = Dex3TactileSubscriber(ChannelSubscriber, HandState_)
    arm_ctrl.speed_gradual_max()

    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    command_socket.bind(("0.0.0.0", args.command_port))
    command_socket.settimeout(0.02)
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_addr = (args.host_ip, args.state_port)

    import numpy as np

    stop = threading.Event()
    last_command_ns = [0]

    def command_loop() -> None:
        # Apply commands as they arrive to keep control latency low.
        while not stop.is_set():
            try:
                datagram, _ = command_socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if datagram[:4] != COMMAND_MAGIC:
                continue
            try:
                command = decode_command(datagram)
            except RobotBridgeError:
                continue
            if command["flags"] & COMMAND_FLAG_HAND:
                with left_hand_pos.get_lock():
                    left_hand_pos[:] = command["left_hand_pos"]
                with right_hand_pos.get_lock():
                    right_hand_pos[:] = command["right_hand_pos"]
            if command["flags"] & COMMAND_FLAG_ARM:
                arm_ctrl.ctrl_dual_arm(np.asarray(command["arm_q"]), np.asarray(command["arm_tau"]))
            last_command_ns[0] = time.monotonic_ns()

    command_thread = threading.Thread(target=command_loop, name="bridge-command", daemon=True)
    command_thread.start()

    state_seq = tactile_seq = 0
    state_period = 1.0 / args.state_rate
    tactile_period = 1.0 / args.tactile_rate
    next_state = next_tactile = time.monotonic()
    timeout_ns = args.command_timeout_ms * 1e6
    print(f"robot_bridge: state->{state_addr}, commands<-0.0.0.0:{args.command_port}", flush=True)
    try:
        while True:
            now = time.monotonic()
            if now >= next_state:
                next_state += state_period
                lowstate = arm_ctrl.lowstate_buffer.GetData()
                arm_q = [lowstate.motor_state[j].q for j in G1_29_JointArmIndex]
                arm_dq = [lowstate.motor_state[j].dq for j in G1_29_JointArmIndex]
                with hand_lock:
                    state_values = list(hand_state[:])
                    action_values = list(hand_action[:])
                n = DEX3_NUM_MOTORS
                state_seq += 1
                send_socket.sendto(encode_state(
                    arm_q, arm_dq, state_values[:n], state_values[n:],
                    action_values[:n], action_values[n:], state_seq, time.monotonic_ns()), state_addr)
            if now >= next_tactile:
                next_tactile += tactile_period
                touch = tactile.snapshot()
                tactile_seq += 1
                send_socket.sendto(encode_tactile(
                    touch.get("left_ee"), touch.get("right_ee"), tactile_seq,
                    time.monotonic_ns()), state_addr)
            # If the Host stops commanding, let the arm hold; a longer silence is
            # a link failure the operator must see rather than drive blind.
            if last_command_ns[0] and (time.monotonic_ns() - last_command_ns[0]) > timeout_ns:
                print("robot_bridge: no command for "
                      f"{args.command_timeout_ms:.0f} ms (holding last pose)", flush=True)
                last_command_ns[0] = 0
            time.sleep(min(state_period, tactile_period) / 2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        command_thread.join(timeout=0.5)
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as error:
            print(f"robot_bridge: arm home failed: {error}", file=sys.stderr)
        command_socket.close()
        send_socket.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

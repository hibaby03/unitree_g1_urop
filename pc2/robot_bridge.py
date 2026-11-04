#!/usr/bin/env python3
"""Orin DDS<->UDP bridge for the Host recorder (unitree_sdk2py only).

The Host runs the recorder over Wi-Fi and cannot see the G1's DDS. This bridge
runs on the Orin (on the robot's DDS network) and:

- reads arm q/dq (rt/lowstate via the vendored G1_29 arm controller), Dex3-1
  hand q and tactile (rt/dex3/*/state) and sends them to the Host over UDP;
- receives the Host's control command over UDP (arm target q/tau from the Host
  IK, and already-retargeted Dex3-1 joint q) and applies it over DDS through the
  vendored arm controller and a raw Dex3 command publisher.

It needs only unitree_sdk2py + numpy + logging_mp — not xr_teleoperate, pinocchio
or dex_retargeting (the Host owns IK and hand retargeting). Wire formats are
shared verbatim with host/robot_link.py and checked for equality in tests.
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
TACTILE_VALUES = 12

_HEADER = "!4sBBHIQ"
STATE_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 4 * NUM_HAND}f")
COMMAND_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 2 * NUM_HAND}f")
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
    return {"arm_q": list(values[0:NUM_ARM]),
            "arm_tau": list(values[NUM_ARM:2 * NUM_ARM]),
            "left_hand_q": list(values[2 * NUM_ARM:2 * NUM_ARM + NUM_HAND]),
            "right_hand_q": list(values[2 * NUM_ARM + NUM_HAND:2 * NUM_ARM + 2 * NUM_HAND]),
            "flags": flags, "sequence": sequence, "host_monotonic_ns": timestamp}


# --- argument parsing --------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin DDS<->UDP bridge (unitree_sdk2py only).")
    parser.add_argument("--interface", default="eth0", help="DDS network interface (e.g. eth0)")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--host-ip", default="192.168.50.7", help="Host address to send robot state to")
    parser.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT)
    parser.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--state-rate", type=float, default=90.0, help="robot-state send rate [Hz]")
    parser.add_argument("--tactile-rate", type=float, default=30.0, help="tactile send rate [Hz]")
    parser.add_argument("--command-timeout-ms", type=float, default=500.0,
                        help="log when Host commands stop; the arm then holds its last pose")
    parser.add_argument("--motion", action="store_true",
                        help="robot is in motion mode; skip entering debug mode")
    parser.add_argument("--no-arm", action="store_true",
                        help="relay state/tactile and hand only; do not command the arm")
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


def enter_debug_mode() -> None:
    """Release any running locomotion mode so arm_sdk/lowcmd takes effect."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    client = MotionSwitcherClient()
    client.SetTimeout(1.0)
    client.Init()
    _status, result = client.CheckMode()
    while result and result.get("name"):
        client.ReleaseMode()
        _status, result = client.CheckMode()
        time.sleep(1)


# --- main --------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    try:
        from .g1_arm_control import G1_29_ArmController, G1_29_JointArmIndex
        from .dex3_control import Dex3CommandPublisher, Dex3StateReader, MAX_DELTA_Q_PER_CYCLE
    except ImportError:
        from g1_arm_control import G1_29_ArmController, G1_29_JointArmIndex  # type: ignore
        from dex3_control import Dex3CommandPublisher, Dex3StateReader, MAX_DELTA_Q_PER_CYCLE  # type: ignore

    if args.check:
        print(f"robot_bridge: imports OK; state->{args.host_ip}:{args.state_port}, "
              f"commands<-:{args.command_port}, interface {args.interface}")
        return 0

    import numpy as np
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    os.environ["UNITREE_NETWORK_INTERFACE"] = args.interface
    ChannelFactoryInitialize(args.domain, args.interface)
    if not args.motion:
        enter_debug_mode()

    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=False)
    hand_pub = Dex3CommandPublisher()
    hand_reader = Dex3StateReader()
    arm_ctrl.speed_gradual_max()

    command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    command_socket.bind(("0.0.0.0", args.command_port))
    command_socket.settimeout(0.02)
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_addr = (args.host_ip, args.state_port)

    stop = threading.Event()
    last_command_ns = [0]

    def command_loop() -> None:
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
                # Slew-limit hand q against the measured state, like Dex3_1_Controller.
                left_now, right_now = hand_reader.joints()
                left_q, right_q = command["left_hand_q"], command["right_hand_q"]
                if left_now is not None:
                    left_q = np.clip(left_q, np.subtract(left_now, MAX_DELTA_Q_PER_CYCLE),
                                     np.add(left_now, MAX_DELTA_Q_PER_CYCLE))
                if right_now is not None:
                    right_q = np.clip(right_q, np.subtract(right_now, MAX_DELTA_Q_PER_CYCLE),
                                      np.add(right_now, MAX_DELTA_Q_PER_CYCLE))
                hand_pub.send(left_q, right_q)
            if not args.no_arm and command["flags"] & COMMAND_FLAG_ARM:
                arm_ctrl.ctrl_dual_arm(np.asarray(command["arm_q"]), np.asarray(command["arm_tau"]))
            last_command_ns[0] = time.monotonic_ns()

    command_thread = threading.Thread(target=command_loop, name="bridge-command", daemon=True)
    command_thread.start()

    state_seq = tactile_seq = 0
    state_period = 1.0 / args.state_rate
    tactile_period = 1.0 / args.tactile_rate
    next_state = next_tactile = time.monotonic()
    timeout_ns = args.command_timeout_ms * 1e6
    print(f"robot_bridge: state->{state_addr}, commands<-0.0.0.0:{args.command_port}, "
          f"arm {'OFF' if args.no_arm else 'ON'}", flush=True)
    try:
        while True:
            now = time.monotonic()
            if now >= next_state:
                next_state += state_period
                arm_q = list(arm_ctrl.get_current_dual_arm_q())
                arm_dq = list(arm_ctrl.get_current_dual_arm_dq())
                left_q, right_q = hand_reader.joints()
                have_hand = left_q is not None and right_q is not None
                state_seq += 1
                send_socket.sendto(encode_state(
                    arm_q, arm_dq,
                    left_q or [], right_q or [],
                    # No separate retargeted action is read back; the measured q
                    # doubles as action here (Host records its own sent command).
                    left_q or [], right_q or [],
                    state_seq, time.monotonic_ns()) if have_hand else encode_state(
                    arm_q, arm_dq, [], [], [], [], state_seq, time.monotonic_ns()), state_addr)
            if now >= next_tactile:
                next_tactile += tactile_period
                left_t, right_t = hand_reader.tactile()
                tactile_seq += 1
                send_socket.sendto(encode_tactile(left_t, right_t, tactile_seq,
                                                  time.monotonic_ns()), state_addr)
            if last_command_ns[0] and (time.monotonic_ns() - last_command_ns[0]) > timeout_ns:
                print(f"robot_bridge: no command for {args.command_timeout_ms:.0f} ms "
                      "(arm holding last pose)", flush=True)
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
        hand_reader.close()
        command_socket.close()
        send_socket.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Receive Meta Quest OpenXR head poses and produce 2-DoF targets."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Optional, Sequence

try:
    from .active_camera_protocol import (
        PACKET_SIZE,
        PacketError,
        TwoDofPoseMapper,
        decode_packet,
        is_newer_sequence,
    )
    from .dynamixel_neck import (
        DynamixelNeck,
        DynamixelNeckError,
        NeckConfiguration,
    )
    from .neck_state_sender import NeckState, NeckStateUdpSender
except ImportError:
    from active_camera_protocol import (  # type: ignore[no-redef]
        PACKET_SIZE,
        PacketError,
        TwoDofPoseMapper,
        decode_packet,
        is_newer_sequence,
    )
    from dynamixel_neck import (  # type: ignore[no-redef]
        DynamixelNeck,
        DynamixelNeckError,
        NeckConfiguration,
    )
    from neck_state_sender import NeckState, NeckStateUdpSender  # type: ignore[no-redef]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def smoothing_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def direction_sign(value: str) -> int:
    parsed = int(value)
    if parsed not in (-1, 1):
        raise argparse.ArgumentTypeError("must be -1 or 1")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive Quest head pose UDP packets and output yaw/pitch JSON."
    )
    parser.add_argument("--bind", default="0.0.0.0", help="local address to bind")
    parser.add_argument("--port", type=int, default=5005, help="UDP port")
    parser.add_argument("--yaw-limit-deg", type=positive_float, default=80.0)
    parser.add_argument("--pitch-down-limit-deg", type=positive_float, default=35.0)
    parser.add_argument("--pitch-up-limit-deg", type=positive_float, default=45.0)
    parser.add_argument(
        "--smoothing",
        type=smoothing_float,
        default=1.0,
        help="EMA alpha: 1 is no smoothing; smaller values are smoother",
    )
    parser.add_argument(
        "--print-rate",
        type=positive_float,
        default=20.0,
        help="maximum JSON lines printed per second",
    )
    parser.add_argument(
        "--timeout-ms",
        type=positive_float,
        default=250.0,
        help="report tracking stale after this interval",
    )
    feedback = parser.add_argument_group("neck state feedback to the Host")
    feedback.add_argument(
        "--neck-state-port",
        type=int,
        default=5006,
        help="UDP port on the Host for yaw/pitch command and encoder state",
    )
    feedback.add_argument(
        "--neck-state-host",
        default=None,
        help="Host address for neck state; default is the head-pose sender",
    )
    feedback.add_argument(
        "--no-neck-state",
        action="store_true",
        help="do not send neck state back to the Host",
    )
    motor = parser.add_argument_group("2XL430 motor output")
    motor.add_argument(
        "--enable-motor",
        action="store_true",
        help="actually enable torque and write goals; omitted means monitor-only",
    )
    motor.add_argument("--dxl-device", default="/dev/ttyUSB0")
    motor.add_argument("--dxl-baudrate", type=int, default=1_000_000)
    motor.add_argument("--yaw-id", type=int, default=1)
    motor.add_argument("--pitch-id", type=int, default=2)
    motor.add_argument("--yaw-center", type=int, default=2048)
    motor.add_argument("--pitch-center", type=int, default=2048)
    motor.add_argument("--yaw-sign", type=direction_sign, default=1)
    motor.add_argument("--pitch-sign", type=direction_sign, default=1)
    motor.add_argument("--profile-velocity", type=int, default=180)
    motor.add_argument("--profile-acceleration", type=int, default=60)
    motor.add_argument(
        "--motor-timeout-ms",
        type=positive_float,
        default=1000.0,
        help="torque off and exit after this long without a valid tracked pose",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")

    mapper = TwoDofPoseMapper(
        yaw_limit_deg=args.yaw_limit_deg,
        pitch_down_limit_deg=args.pitch_down_limit_deg,
        pitch_up_limit_deg=args.pitch_up_limit_deg,
        smoothing=args.smoothing,
    )

    neck: Optional[DynamixelNeck] = None
    if args.enable_motor:
        configuration = NeckConfiguration(
            device=args.dxl_device,
            baudrate=args.dxl_baudrate,
            yaw_id=args.yaw_id,
            pitch_id=args.pitch_id,
            yaw_center=args.yaw_center,
            pitch_center=args.pitch_center,
            yaw_sign=args.yaw_sign,
            pitch_sign=args.pitch_sign,
            profile_velocity=args.profile_velocity,
            profile_acceleration=args.profile_acceleration,
        )
        neck = DynamixelNeck(configuration)
        neck.connect()

    state_sender: Optional[NeckStateUdpSender] = None
    if not args.no_neck_state:
        state_sender = NeckStateUdpSender(args.neck_state_port)
    last_state_error: Optional[str] = None

    receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receive_socket.bind((args.bind, args.port))
    receive_socket.settimeout(min(0.05, args.timeout_ms / 1000.0))

    print(
        json.dumps(
            {
                "event": "listening",
                "bind": args.bind,
                "port": args.port,
                "packet_size": PACKET_SIZE,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    previous_sequence: Optional[int] = None
    last_valid_monotonic: Optional[float] = None
    last_print_monotonic = 0.0
    stale_reported = False
    print_period = 1.0 / args.print_rate
    timeout_seconds = args.timeout_ms / 1000.0
    motor_watchdog_start = time.monotonic() if neck is not None else None

    try:
        while True:
            now = time.monotonic()
            if (
                last_valid_monotonic is not None
                and not stale_reported
                and now - last_valid_monotonic >= timeout_seconds
            ):
                print(
                    json.dumps(
                        {"event": "stale", "timeout_ms": args.timeout_ms},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                stale_reported = True

            if neck is not None:
                watchdog_reference = (
                    last_valid_monotonic
                    if last_valid_monotonic is not None
                    else motor_watchdog_start
                )
                assert watchdog_reference is not None
                if now - watchdog_reference >= args.motor_timeout_ms / 1000.0:
                    print(
                        json.dumps(
                            {
                                "event": "motor_timeout",
                                "timeout_ms": args.motor_timeout_ms,
                                "action": "torque_off_and_exit",
                            },
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    return 3

            try:
                datagram, source = receive_socket.recvfrom(256)
            except socket.timeout:
                continue

            try:
                packet = decode_packet(datagram)
            except PacketError as error:
                print(
                    json.dumps(
                        {
                            "event": "packet_rejected",
                            "source": source[0],
                            "reason": str(error),
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if not is_newer_sequence(packet.sequence, previous_sequence):
                continue
            previous_sequence = packet.sequence

            target = mapper.update(packet)
            if target is None:
                continue

            sample_pc2_time_ns = time.monotonic_ns()
            now_monotonic = sample_pc2_time_ns / 1_000_000_000.0
            last_valid_monotonic = now_monotonic
            stale_reported = False

            motor_positions = None
            motor_command_pc2_time_ns = None
            if neck is not None:
                motor_command_pc2_time_ns = time.monotonic_ns()
                motor_positions = neck.write_angles(
                    target.yaw_deg,
                    target.pitch_deg,
                )

            if state_sender is not None:
                present_deg = present_position = present_time_ns = None
                state_error = None
                if neck is not None:
                    try:
                        present_time_ns = time.monotonic_ns()
                        present_deg, present_position = neck.read_present_angles()
                    except DynamixelNeckError as error:
                        present_time_ns = None
                        state_error = f"encoder read failed: {error}"
                try:
                    state_sender.send(
                        args.neck_state_host or source[0],
                        NeckState(
                            pose_sequence=target.sequence,
                            command_pc2_monotonic_ns=(
                                motor_command_pc2_time_ns or sample_pc2_time_ns
                            ),
                            command_deg=(target.yaw_deg, target.pitch_deg),
                            goal_position=motor_positions,
                            present_pc2_monotonic_ns=present_time_ns,
                            present_deg=present_deg,
                            present_position=present_position,
                            torque_enabled=neck is not None and neck.torque_enabled,
                        ),
                    )
                except OSError as error:
                    state_error = f"send failed: {error}"
                # Report each distinct failure once instead of at the pose rate.
                if state_error != last_state_error and state_error is not None:
                    print(
                        json.dumps(
                            {"event": "neck_state_error", "reason": state_error},
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                last_state_error = state_error

            if now_monotonic - last_print_monotonic < print_period:
                continue
            last_print_monotonic = now_monotonic

            clock_delta_ms = (time.time_ns() - target.sender_time_ns) / 1_000_000.0
            output = {
                "sequence": target.sequence,
                "yaw_deg": round(target.yaw_deg, 4),
                "pitch_deg": round(target.pitch_deg, 4),
                "tracked": True,
                "source": source[0],
                "pc2_monotonic_ns": sample_pc2_time_ns,
            }
            # This is only a latency estimate when Quest and Jetson wall clocks
            # are synchronized. Keep implausible values out of normal output.
            if -1000.0 <= clock_delta_ms <= 10_000.0:
                output["clock_delta_ms"] = round(clock_delta_ms, 3)
            if motor_positions is not None:
                output["motor_command_pc2_monotonic_ns"] = motor_command_pc2_time_ns
                output["yaw_position"] = motor_positions[0]
                output["pitch_position"] = motor_positions[1]

            print(json.dumps(output, separators=(",", ":")), flush=True)

    except KeyboardInterrupt:
        print('{"event":"stopped"}', flush=True)
        return 0
    finally:
        receive_socket.close()
        if state_sender is not None:
            state_sender.close()
        if neck is not None:
            neck.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, DynamixelNeckError) as error:
        print(f"receiver error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

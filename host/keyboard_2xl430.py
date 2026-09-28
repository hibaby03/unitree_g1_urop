#!/usr/bin/env python3
"""Windows keyboard teleoperation for a 2-axis ROBOTIS 2XL430.

The program starts in a *torque-off* state.  It reads each motor's present
position before torque is enabled, so the first command holds the current
position instead of jumping to an assumed centre.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

try:
    import dynamixel_sdk as dxl
except ImportError as error:
    dxl = None  # type: ignore[assignment]
    DXL_IMPORT_ERROR = error
else:
    DXL_IMPORT_ERROR = None


PROTOCOL_VERSION = 2.0
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

POSITION_MODE = 3
TORQUE_OFF = 0
TORQUE_ON = 1
POSITION_MIN = 0
POSITION_MAX = 4095
COUNTS_PER_DEGREE = 4096.0 / 360.0
STD_INPUT_HANDLE = -10


@dataclass
class Config:
    port: str
    baudrate: int
    yaw_id: int
    pitch_id: int
    step_deg: float
    limit_deg: float
    velocity: int
    acceleration: int
    rate_hz: float


class MotorBus:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.port = dxl.PortHandler(config.port)
        self.packet = dxl.PacketHandler(PROTOCOL_VERSION)
        self.sync_write = dxl.GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, 4
        )
        self.connected = False
        self.torque_enabled = False
        self.home: dict[int, int] = {}
        self.goal: dict[int, int] = {}

    @property
    def ids(self) -> tuple[int, int]:
        return self.config.yaw_id, self.config.pitch_id

    def open(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Cannot open {self.config.port}. Check Device Manager COM port.")
        self.connected = True
        if not self.port.setBaudRate(self.config.baudrate):
            raise RuntimeError(f"Cannot set baudrate to {self.config.baudrate}.")

    def _check(self, motor_id: int, address: int, result: int, error: int) -> None:
        if result != dxl.COMM_SUCCESS:
            raise RuntimeError(
                f"ID {motor_id}, addr {address}: {self.packet.getTxRxResult(result)}"
            )
        if error:
            raise RuntimeError(
                f"ID {motor_id}, addr {address}: {self.packet.getRxPacketError(error)}"
            )

    def write1(self, motor_id: int, address: int, value: int) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, motor_id, address, value)
        self._check(motor_id, address, result, error)

    def write4(self, motor_id: int, address: int, value: int) -> None:
        result, error = self.packet.write4ByteTxRx(self.port, motor_id, address, value)
        self._check(motor_id, address, result, error)

    def read_position(self, motor_id: int) -> int:
        value, result, error = self.packet.read4ByteTxRx(
            self.port, motor_id, ADDR_PRESENT_POSITION
        )
        self._check(motor_id, ADDR_PRESENT_POSITION, result, error)
        return int(value)

    def enable_torque_at_current_position(self) -> None:
        # Set the goal first while torque is off; this prevents a jump on enable.
        for motor_id in self.ids:
            current = self.read_position(motor_id)
            self.home[motor_id] = current
            self.goal[motor_id] = current
            self.write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            self.write1(motor_id, ADDR_OPERATING_MODE, POSITION_MODE)
            self.write4(motor_id, ADDR_PROFILE_ACCELERATION, self.config.acceleration)
            self.write4(motor_id, ADDR_PROFILE_VELOCITY, self.config.velocity)
            self.write4(motor_id, ADDR_GOAL_POSITION, current)
        for motor_id in self.ids:
            self.write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        self.torque_enabled = True

    def move_both(self, yaw_delta_counts: int, pitch_delta_counts: int) -> None:
        if not self.torque_enabled:
            return
        max_delta = round(self.config.limit_deg * COUNTS_PER_DEGREE)
        deltas = {
            self.config.yaw_id: yaw_delta_counts,
            self.config.pitch_id: pitch_delta_counts,
        }
        for motor_id in self.ids:
            lower = max(POSITION_MIN, self.home[motor_id] - max_delta)
            upper = min(POSITION_MAX, self.home[motor_id] + max_delta)
            self.goal[motor_id] = max(
                lower, min(upper, self.goal[motor_id] + deltas[motor_id])
            )
        self.sync_write.clearParam()
        try:
            for motor_id in self.ids:
                data = self.goal[motor_id].to_bytes(4, byteorder="little", signed=False)
                if not self.sync_write.addParam(motor_id, data):
                    raise RuntimeError(f"Cannot add ID {motor_id} to Sync Write")
            result = self.sync_write.txPacket()
            if result != dxl.COMM_SUCCESS:
                raise RuntimeError(f"Sync Write: {self.packet.getTxRxResult(result)}")
        finally:
            self.sync_write.clearParam()

    def torque_off(self) -> None:
        if not self.connected:
            return
        for motor_id in self.ids:
            try:
                self.write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except RuntimeError as error:
                print(f"Warning while disabling ID {motor_id}: {error}", file=sys.stderr)
        self.torque_enabled = False

    def close(self) -> None:
        self.torque_off()
        if self.connected:
            self.port.closePort()
        self.connected = False


def print_status(bus: MotorBus) -> None:
    if not bus.torque_enabled:
        print("Torque: OFF")
        return
    yaw, pitch = bus.ids
    print(
        f"Goal counts: yaw(ID {yaw})={bus.goal[yaw]}, "
        f"pitch(ID {pitch})={bus.goal[pitch]}"
    )


def flush_console_input() -> None:
    """Discard key events accumulated while polling GetAsyncKeyState."""

    console_input = ctypes.windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
    if console_input:
        ctypes.windll.kernel32.FlushConsoleInputBuffer(console_input)


def print_help(config: Config) -> None:
    print(
        "\nControls (hold one or two direction keys)\n"
        "  W / S : pitch + / -     (W+D, W+A, S+D, S+A: simultaneous diagonal)\n"
        "  A / D : yaw   - / +\n"
        "  T     : torque ON at the current position (safe hold)\n"
        "  X     : torque OFF (motors become free)\n"
        "  R     : set current goals as the new software centre\n"
        "  P     : print present positions\n"
        "  H     : show this help\n"
        "  Q or Esc : torque OFF and exit\n"
        f"\nSpeed: {config.step_deg * config.rate_hz:g} deg/s; software travel limit: +/- {config.limit_deg:g} deg\n"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Keyboard control for a 2XL430 on Windows")
    parser.add_argument("--port", default="COM3", help="U2D2 COM port (default: COM3)")
    # 2XL430-W250 factory default: baud-rate register value 1 = 57,600 bps.
    parser.add_argument("--baudrate", type=int, default=57_600)
    parser.add_argument("--yaw-id", type=int, default=5)
    parser.add_argument("--pitch-id", type=int, default=6)
    parser.add_argument("--step-deg", type=float, default=2.0)
    parser.add_argument("--limit-deg", type=float, default=20.0,
                        help="limit relative to the position at T (default: 20)")
    parser.add_argument("--velocity", type=int, default=60, help="Profile Velocity (default: 60)")
    parser.add_argument("--acceleration", type=int, default=20, help="Profile Acceleration (default: 20)")
    parser.add_argument("--rate-hz", type=float, default=10.0,
                        help="keyboard update rate; lower is gentler (default: 10)")
    args = parser.parse_args(argv)
    if args.yaw_id == args.pitch_id:
        parser.error("yaw-id and pitch-id must differ")
    if args.baudrate <= 0 or args.step_deg <= 0 or args.limit_deg <= 0 or args.rate_hz <= 0:
        parser.error("baudrate, step-deg, limit-deg, and rate-hz must be positive")
    return Config(**vars(args))


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    if dxl is None:
        print(
            "dynamixel-sdk is missing. Install it with: python -m pip install dynamixel-sdk",
            file=sys.stderr,
        )
        return 2
    step_counts = max(1, round(config.step_deg * COUNTS_PER_DEGREE))
    bus = MotorBus(config)
    command_keys = {ord(key) for key in "TXRPHQ"} | {0x1B}
    previously_down: set[int] = set()

    def key_down(virtual_key: int) -> bool:
        return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    try:
        bus.open()
        print(f"Connected: {config.port} @ {config.baudrate}. Torque is OFF.")
        print_help(config)
        while True:
            down = {key for key in command_keys if key_down(key)}
            pressed = down - previously_down
            previously_down = down
            if ord("Q") in pressed or 0x1B in pressed:
                return 0
            if ord("H") in pressed:
                print_help(config)
            elif ord("T") in pressed:
                bus.enable_torque_at_current_position()
                print("Torque ON; goals initialized from present positions.")
                print_status(bus)
            elif ord("X") in pressed:
                bus.torque_off()
                print("Torque OFF.")
            elif ord("R") in pressed:
                if bus.torque_enabled:
                    bus.home = dict(bus.goal)
                    print("Software centre reset to current goals.")
                else:
                    print("Torque is OFF. Press T first.")
            elif ord("P") in pressed:
                for motor_id in bus.ids:
                    print(f"ID {motor_id} present position: {bus.read_position(motor_id)}")
            yaw_delta = step_counts * (key_down(ord("D")) - key_down(ord("A")))
            pitch_delta = step_counts * (key_down(ord("W")) - key_down(ord("S")))
            if yaw_delta or pitch_delta:
                bus.move_both(yaw_delta, pitch_delta)
            time.sleep(1.0 / config.rate_hz)
    except (OSError, RuntimeError) as error:
        print(f"Motor error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        flush_console_input()
        bus.close()
        print("Closed port; torque OFF.")


if __name__ == "__main__":
    raise SystemExit(main())

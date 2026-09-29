"""Encode PC2 neck command/encoder state as fixed-size UDP packets for the Host.

See PROTOCOL.md, "Neck state packet". The Host decoder lives in
host/neck_state_receiver.py; both sides define the same struct.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
from typing import Optional, Tuple


MAGIC = b"ANCK"
VERSION = 1
PACKET_STRUCT = struct.Struct("!4sBBHIQQ2f2f2i2i")
PACKET_SIZE = PACKET_STRUCT.size

FLAG_COMMAND_VALID = 1 << 0
FLAG_PRESENT_VALID = 1 << 1
FLAG_TORQUE_ENABLED = 1 << 2


@dataclass(frozen=True)
class NeckState:
    pose_sequence: int
    command_pc2_monotonic_ns: int
    command_deg: Tuple[float, float]
    goal_position: Optional[Tuple[int, int]] = None
    present_pc2_monotonic_ns: Optional[int] = None
    present_deg: Optional[Tuple[float, float]] = None
    present_position: Optional[Tuple[int, int]] = None
    torque_enabled: bool = False


def build_neck_state_packet(state: NeckState) -> bytes:
    flags = FLAG_COMMAND_VALID
    if state.torque_enabled:
        flags |= FLAG_TORQUE_ENABLED

    present_deg = (0.0, 0.0)
    present_position = (0, 0)
    present_time = 0
    if (
        state.present_deg is not None
        and state.present_position is not None
        and state.present_pc2_monotonic_ns is not None
    ):
        flags |= FLAG_PRESENT_VALID
        present_deg = state.present_deg
        present_position = state.present_position
        present_time = state.present_pc2_monotonic_ns

    values = (*state.command_deg, *present_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("neck state contains a non-finite angle")

    return PACKET_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        0,
        state.pose_sequence & 0xFFFFFFFF,
        state.command_pc2_monotonic_ns,
        present_time,
        *state.command_deg,
        *present_deg,
        *(state.goal_position or (0, 0)),
        *present_position,
    )


class NeckStateUdpSender:
    """Send one neck-state datagram per command to the Host."""

    def __init__(self, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be in 1..65535")
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, host: str, state: NeckState) -> None:
        self._socket.sendto(build_neck_state_packet(state), (host, self.port))

    def close(self) -> None:
        self._socket.close()

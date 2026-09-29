"""Receive PC2 neck command/encoder state packets on the Host.

See PROTOCOL.md, "Neck state packet". The PC2 encoder lives in
pc2/neck_state_sender.py; both sides define the same struct.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import threading
import time
from typing import Optional, Tuple


MAGIC = b"ANCK"
VERSION = 1
PACKET_STRUCT = struct.Struct("!4sBBHIQQ2f2f2i2i")
PACKET_SIZE = PACKET_STRUCT.size

FLAG_COMMAND_VALID = 1 << 0
FLAG_PRESENT_VALID = 1 << 1
FLAG_TORQUE_ENABLED = 1 << 2


class NeckStatePacketError(ValueError):
    """Raised when a datagram is not a valid neck-state packet."""


@dataclass(frozen=True)
class NeckStatePacket:
    pose_sequence: int
    command_pc2_monotonic_ns: int
    command_deg: Tuple[float, float]
    goal_position: Optional[Tuple[int, int]]
    present_pc2_monotonic_ns: Optional[int]
    present_deg: Optional[Tuple[float, float]]
    present_position: Optional[Tuple[int, int]]
    torque_enabled: bool


def decode_neck_state_packet(datagram: bytes) -> NeckStatePacket:
    if len(datagram) != PACKET_SIZE:
        raise NeckStatePacketError(f"wrong packet size: {len(datagram)} != {PACKET_SIZE}")
    (
        magic, version, flags, reserved, pose_sequence,
        command_time, present_time,
        command_yaw, command_pitch, present_yaw, present_pitch,
        goal_yaw, goal_pitch, present_yaw_position, present_pitch_position,
    ) = PACKET_STRUCT.unpack(datagram)
    if magic != MAGIC:
        raise NeckStatePacketError("wrong magic")
    if version != VERSION:
        raise NeckStatePacketError(f"unsupported version: {version}")
    if reserved != 0:
        raise NeckStatePacketError("reserved field must be zero")
    if not flags & FLAG_COMMAND_VALID:
        raise NeckStatePacketError("command is not valid")
    if not all(math.isfinite(v) for v in (command_yaw, command_pitch, present_yaw, present_pitch)):
        raise NeckStatePacketError("angle is not finite")

    torque_enabled = bool(flags & FLAG_TORQUE_ENABLED)
    present_valid = bool(flags & FLAG_PRESENT_VALID)
    return NeckStatePacket(
        pose_sequence=pose_sequence,
        command_pc2_monotonic_ns=command_time,
        command_deg=(command_yaw, command_pitch),
        goal_position=(goal_yaw, goal_pitch) if torque_enabled else None,
        present_pc2_monotonic_ns=present_time if present_valid else None,
        present_deg=(present_yaw, present_pitch) if present_valid else None,
        present_position=(present_yaw_position, present_pitch_position) if present_valid else None,
        torque_enabled=torque_enabled,
    )


def neck_record_entries(
    packet: Optional[NeckStatePacket],
    age_ms: Optional[float],
) -> tuple[dict, dict]:
    """Return (states["neck"], actions["neck"]) for EpisodeWriter.

    qpos is [yaw, pitch] in radians like the other joints: yaw is right
    positive and pitch is up positive, relative to the neck center counts.
    Empty lists mean no data (never received, or encoder not read).
    """

    if packet is None:
        empty = {"qpos": [], "qvel": [], "torque": []}
        return dict(empty), dict(empty)

    state = {
        "qpos": [math.radians(v) for v in packet.present_deg] if packet.present_deg else [],
        "qvel": [],
        "torque": [],
        "position_counts": list(packet.present_position) if packet.present_position else [],
        "pc2_monotonic_ns": packet.present_pc2_monotonic_ns,
        "age_ms": age_ms,
    }
    action = {
        "qpos": [math.radians(v) for v in packet.command_deg],
        "qvel": [],
        "torque": [],
        "position_counts": list(packet.goal_position) if packet.goal_position else [],
        "pc2_monotonic_ns": packet.command_pc2_monotonic_ns,
        "pose_sequence": packet.pose_sequence,
        "torque_enabled": packet.torque_enabled,
        "age_ms": age_ms,
    }
    return state, action


class NeckStateReceiver:
    """Background UDP listener that keeps the newest neck-state packet."""

    def __init__(self, bind: str = "0.0.0.0", port: int = 5006) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in 0..65535 (0 picks a free port)")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((bind, port))
        self._socket.settimeout(0.05)
        self.port = self._socket.getsockname()[1]
        self._lock = threading.Lock()
        self._latest: Optional[NeckStatePacket] = None
        self._received_ns: Optional[int] = None
        self.rejected = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="neck-state-udp", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                datagram, _ = self._socket.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            try:
                packet = decode_neck_state_packet(datagram)
            except NeckStatePacketError:
                self.rejected += 1
                continue
            with self._lock:
                # PC2 sends from a single loop; UDP reordering on the local
                # link is rare, but never let an older command replace a newer one.
                if (self._latest is None
                        or packet.command_pc2_monotonic_ns >= self._latest.command_pc2_monotonic_ns):
                    self._latest = packet
                    self._received_ns = time.monotonic_ns()

    def latest(self) -> tuple[Optional[NeckStatePacket], Optional[float]]:
        """Newest packet and its Host-side age in milliseconds."""

        with self._lock:
            if self._latest is None or self._received_ns is None:
                return None, None
            return self._latest, round((time.monotonic_ns() - self._received_ns) / 1e6, 3)

    def record_entries(self) -> tuple[dict, dict]:
        return neck_record_entries(*self.latest())

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close()

"""Host side of the Orin DDS<->UDP bridge.

The recorder runs on the Host over Wi-Fi, but the G1 publishes arm/hand/tactile
on DDS on the robot's wired network, which the Host cannot see. pc2/robot_bridge.py
runs on the Orin (on the robot DDS network), relays robot state to the Host over
UDP and applies the Host's control commands to the robot over DDS. This module is
the Host end: it decodes the state/tactile packets and encodes command packets.

Wire formats (network byte order), shared verbatim with pc2/robot_bridge.py and
checked for equality in tests:

- Robot state  (Orin -> Host): magic ``ARST`` + arm q/dq and hand q/action.
- Tactile      (Orin -> Host): magic ``ATAC`` + variable Dex3-1 press sensors.
- Robot command (Host -> Orin): magic ``ARCM`` + arm target q/tau and hand pos.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import threading
import time
from typing import Optional


# --- shared wire constants (keep identical in pc2/robot_bridge.py) -----------
STATE_MAGIC = b"ARST"
TACTILE_MAGIC = b"ATAC"
COMMAND_MAGIC = b"ARCM"
VERSION = 1

NUM_ARM = 14          # 7 left + 7 right
NUM_HAND = 7          # per hand
NUM_HAND_POS = 75     # 25 landmarks * 3, one hand of Quest hand tracking
TACTILE_VALUES = 12   # pressure or temperature values per press sensor

# header: magic(4s) version(B) flags(B) reserved(H) sequence(I) timestamp_ns(Q)
_HEADER = "!4sBBHIQ"
STATE_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 4 * NUM_HAND}f")
COMMAND_STRUCT = struct.Struct(_HEADER + f"{NUM_ARM + NUM_ARM + 2 * NUM_HAND_POS}f")
TACTILE_HEADER = struct.Struct(_HEADER + "HH")  # + left_count, right_count
TACTILE_SENSOR = struct.Struct(f"!{TACTILE_VALUES}f{TACTILE_VALUES}fi")  # pressure, temp, lost

STATE_FLAG_ARM = 1 << 0
STATE_FLAG_HAND = 1 << 1
COMMAND_FLAG_ARM = 1 << 0
COMMAND_FLAG_HAND = 1 << 1

DEFAULT_STATE_PORT = 5008
DEFAULT_COMMAND_PORT = 5009


class RobotLinkError(RuntimeError):
    """Raised when a packet cannot be decoded."""


@dataclass(frozen=True)
class RobotState:
    arm_q: list[float]       # 14, empty if arm not valid
    arm_dq: list[float]      # 14
    left_hand_q: list[float]  # 7, empty if hand not valid
    right_hand_q: list[float]
    left_hand_action: list[float]
    right_hand_action: list[float]
    pc2_monotonic_ns: int
    sequence: int


def _finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


# --- robot state (Orin -> Host) ---------------------------------------------

def decode_state(datagram: bytes) -> RobotState:
    if len(datagram) != STATE_STRUCT.size:
        raise RobotLinkError(f"state packet is {len(datagram)} bytes, expected {STATE_STRUCT.size}")
    magic, version, flags, _reserved, sequence, timestamp, *values = STATE_STRUCT.unpack(datagram)
    if magic != STATE_MAGIC or version != VERSION:
        raise RobotLinkError("bad state magic/version")
    arm_q = values[0:NUM_ARM]
    arm_dq = values[NUM_ARM:2 * NUM_ARM]
    rest = values[2 * NUM_ARM:]
    lh_q, rh_q = rest[0:NUM_HAND], rest[NUM_HAND:2 * NUM_HAND]
    lh_a, rh_a = rest[2 * NUM_HAND:3 * NUM_HAND], rest[3 * NUM_HAND:4 * NUM_HAND]
    arm_valid = bool(flags & STATE_FLAG_ARM)
    hand_valid = bool(flags & STATE_FLAG_HAND)
    return RobotState(
        arm_q=list(arm_q) if arm_valid else [],
        arm_dq=list(arm_dq) if arm_valid else [],
        left_hand_q=list(lh_q) if hand_valid else [],
        right_hand_q=list(rh_q) if hand_valid else [],
        left_hand_action=list(lh_a) if hand_valid else [],
        right_hand_action=list(rh_a) if hand_valid else [],
        pc2_monotonic_ns=timestamp, sequence=sequence)


# --- tactile (Orin -> Host) -------------------------------------------------

def decode_tactile(datagram: bytes) -> dict:
    """Return {"left_ee": {...}|None, "right_ee": {...}|None} without age_ms.

    Matches Dex3TactileSubscriber.snapshot() apart from age_ms, which the Host
    stamps on arrival (see RobotLink)."""

    if len(datagram) < TACTILE_HEADER.size:
        raise RobotLinkError("tactile packet too short")
    magic, version, _flags, _reserved, _seq, _ts, left_count, right_count = \
        TACTILE_HEADER.unpack_from(datagram)
    if magic != TACTILE_MAGIC or version != VERSION:
        raise RobotLinkError("bad tactile magic/version")
    offset = TACTILE_HEADER.size
    result: dict = {}
    for key, count in (("left_ee", left_count), ("right_ee", right_count)):
        if count == 0:
            result[key] = None
            continue
        pressure, temperature, lost = [], [], []
        for _ in range(count):
            if offset + TACTILE_SENSOR.size > len(datagram):
                raise RobotLinkError("tactile packet truncated")
            values = TACTILE_SENSOR.unpack_from(datagram, offset)
            offset += TACTILE_SENSOR.size
            pressure.append(list(values[:TACTILE_VALUES]))
            temperature.append(list(values[TACTILE_VALUES:2 * TACTILE_VALUES]))
            lost.append(int(values[-1]))
        result[key] = {"pressure": pressure, "temperature": temperature, "lost": lost}
    return result


# --- robot command (Host -> Orin) -------------------------------------------

def encode_command(arm_target_q, arm_tauff, left_hand_pos, right_hand_pos,
                   sequence: int, host_monotonic_ns: int) -> bytes:
    arm_target_q = list(arm_target_q)
    arm_tauff = list(arm_tauff)
    left_hand_pos = list(left_hand_pos)
    right_hand_pos = list(right_hand_pos)
    if len(arm_target_q) != NUM_ARM or len(arm_tauff) != NUM_ARM:
        raise RobotLinkError("arm command must have 14 q and 14 tau")
    if len(left_hand_pos) != NUM_HAND_POS or len(right_hand_pos) != NUM_HAND_POS:
        raise RobotLinkError("hand positions must have 75 values each")
    payload = arm_target_q + arm_tauff + left_hand_pos + right_hand_pos
    if not _finite(payload):
        raise RobotLinkError("command contains a non-finite value")
    flags = COMMAND_FLAG_ARM | COMMAND_FLAG_HAND
    return COMMAND_STRUCT.pack(COMMAND_MAGIC, VERSION, flags, 0,
                               sequence & 0xFFFFFFFF, host_monotonic_ns, *payload)


# --- Host-side link ----------------------------------------------------------

class RobotLink:
    """Receive robot state/tactile from the Orin bridge; send control commands."""

    def __init__(self, bridge_ip: str, state_port: int = DEFAULT_STATE_PORT,
                 command_port: int = DEFAULT_COMMAND_PORT, state_bind: str = "0.0.0.0") -> None:
        self._recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv.bind((state_bind, state_port))
        self._recv.settimeout(0.05)
        self.state_port = self._recv.getsockname()[1]
        self._send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._command_addr = (bridge_ip, command_port)
        self._lock = threading.Lock()
        self._state: Optional[RobotState] = None
        self._state_ns: Optional[int] = None
        self._tactile: dict = {"left_ee": None, "right_ee": None}
        self._tactile_ns: Optional[int] = None
        self._command_sequence = 0
        self.rejected = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="robot-link-udp", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                datagram, _ = self._recv.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            now = time.monotonic_ns()
            try:
                if datagram[:4] == STATE_MAGIC:
                    state = decode_state(datagram)
                    with self._lock:
                        if self._state is None or state.sequence >= self._state.sequence:
                            self._state, self._state_ns = state, now
                elif datagram[:4] == TACTILE_MAGIC:
                    tactile = decode_tactile(datagram)
                    with self._lock:
                        self._tactile, self._tactile_ns = tactile, now
                else:
                    self.rejected += 1
            except RobotLinkError:
                self.rejected += 1

    def latest_state(self) -> tuple[Optional[RobotState], Optional[float]]:
        """Newest robot state and its Host-side age in milliseconds."""
        with self._lock:
            if self._state is None or self._state_ns is None:
                return None, None
            return self._state, round((time.monotonic_ns() - self._state_ns) / 1e6, 3)

    def tactile_snapshot(self) -> dict:
        """Latest tactile per hand with age_ms, matching Dex3TactileSubscriber."""
        now = time.monotonic_ns()
        with self._lock:
            result: dict = {}
            for key, sample in self._tactile.items():
                if sample is None or self._tactile_ns is None:
                    result[key] = None
                else:
                    result[key] = {**sample, "age_ms": round((now - self._tactile_ns) / 1e6, 3)}
            return result

    def send_command(self, arm_target_q, arm_tauff, left_hand_pos, right_hand_pos) -> None:
        self._command_sequence = (self._command_sequence + 1) & 0xFFFFFFFF
        datagram = encode_command(arm_target_q, arm_tauff, left_hand_pos, right_hand_pos,
                                  self._command_sequence, time.monotonic_ns())
        self._send.sendto(datagram, self._command_addr)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._recv.close()
        self._send.close()

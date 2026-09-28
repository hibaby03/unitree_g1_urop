"""Encode raw TeleVuer/OpenXR head matrices as fixed-size UDP packets."""

from __future__ import annotations

import math
import socket
import struct
import time
from typing import Any, Optional, Tuple


MAGIC = b"ACAM"
VERSION = 1
PACKET_STRUCT = struct.Struct("!4sBBHIQ3f4f")

FLAG_ORIENTATION_VALID = 1 << 0
FLAG_POSITION_VALID = 1 << 1
FLAG_TRACKED = 1 << 2

Quaternion = Tuple[float, float, float, float]


Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]
Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


def _validated_pose_matrix(matrix: Any) -> Matrix4:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("head pose must be an iterable 4x4 matrix") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("head pose must be 4x4")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("head pose contains a non-finite value")
    if any(abs(rows[3][index] - expected) > 1e-4 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("head pose has an invalid homogeneous row")

    rotation = tuple(tuple(row[column] for column in range(3)) for row in rows[:3])
    for left_column in range(3):
        for right_column in range(3):
            dot = sum(
                rotation[row][left_column] * rotation[row][right_column]
                for row in range(3)
            )
            expected = 1.0 if left_column == right_column else 0.0
            if not math.isclose(dot, expected, abs_tol=2e-3):
                raise ValueError("head rotation is not orthonormal")

    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=2e-3):
        raise ValueError("head rotation determinant is not +1")
    return rows  # type: ignore[return-value]


def rotation_matrix_to_quaternion(rotation: Any) -> Quaternion:
    """Return an x/y/z/w quaternion from a proper 3x3 rotation matrix."""

    try:
        matrix = tuple(tuple(float(value) for value in row) for row in rotation)
    except (TypeError, ValueError) as error:
        raise ValueError("rotation must be an iterable 3x3 matrix") from error
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("rotation must be 3x3")

    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("rotation produced a zero quaternion")
    return qx / norm, qy / norm, qz / norm, qw / norm


def build_packet_from_openxr_matrix(
    matrix: Any,
    sequence: int,
    sender_time_ns: Optional[int] = None,
) -> bytes:
    """Build protocol-v1 data from TeleVuer's raw OpenXR 4x4 matrix."""

    pose = _validated_pose_matrix(matrix)
    position = tuple(pose[row][3] for row in range(3))
    orientation = rotation_matrix_to_quaternion(
        tuple(tuple(pose[row][column] for column in range(3)) for row in range(3))
    )
    flags = FLAG_ORIENTATION_VALID | FLAG_POSITION_VALID | FLAG_TRACKED
    timestamp = time.time_ns() if sender_time_ns is None else sender_time_ns

    return PACKET_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        0,
        sequence & 0xFFFFFFFF,
        timestamp,
        *position,
        *orientation,
    )


class HeadPoseUdpSender:
    """Small allocation-light UDP sender used by the TeleVuer wrapper."""

    def __init__(self, destination_ip: str, port: int = 5005) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be in 1..65535")
        self.destination = (destination_ip, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0
        self._closed = False

    def send_openxr_matrix(self, matrix: Any) -> None:
        if self._closed:
            raise RuntimeError("sender is closed")
        datagram = build_packet_from_openxr_matrix(matrix, self._sequence)
        self._socket.sendto(datagram, self.destination)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF

    def close(self) -> None:
        if not self._closed:
            self._socket.close()
            self._closed = True

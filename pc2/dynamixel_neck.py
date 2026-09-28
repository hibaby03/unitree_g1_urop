"""2XL430 yaw/pitch driver using the ROBOTIS DYNAMIXEL SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116

LEN_GOAL_POSITION = 4
POSITION_CONTROL_MODE = 3
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

PULSES_PER_REVOLUTION = 4096.0
DEGREES_PER_REVOLUTION = 360.0
MIN_POSITION = 0
MAX_POSITION = 4095


class DynamixelNeckError(RuntimeError):
    """Raised when the neck bus cannot be configured or commanded safely."""


def angle_to_position(
    angle_deg: float,
    center_position: int,
    direction_sign: int,
) -> int:
    """Convert a signed joint angle to a bounded 2XL430 position count."""

    if direction_sign not in (-1, 1):
        raise ValueError("direction_sign must be -1 or 1")
    if not MIN_POSITION <= center_position <= MAX_POSITION:
        raise ValueError("center_position must be in 0..4095")

    pulses = angle_deg * (PULSES_PER_REVOLUTION / DEGREES_PER_REVOLUTION)
    position = int(round(center_position + direction_sign * pulses))
    return max(MIN_POSITION, min(MAX_POSITION, position))


@dataclass(frozen=True)
class NeckConfiguration:
    device: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000
    yaw_id: int = 1
    pitch_id: int = 2
    yaw_center: int = 2048
    pitch_center: int = 2048
    yaw_sign: int = 1
    pitch_sign: int = 1
    profile_velocity: int = 180
    profile_acceleration: int = 60


class DynamixelNeck:
    """Owns the serial port and sends both neck goals in one Sync Write."""

    def __init__(
        self,
        configuration: NeckConfiguration,
        sdk_module: Optional[Any] = None,
    ) -> None:
        self.configuration = configuration
        if configuration.yaw_id == configuration.pitch_id:
            raise ValueError("yaw_id and pitch_id must be different")
        if not 0 <= configuration.yaw_id <= 252:
            raise ValueError("yaw_id must be in 0..252")
        if not 0 <= configuration.pitch_id <= 252:
            raise ValueError("pitch_id must be in 0..252")
        if configuration.baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if configuration.profile_velocity < 0:
            raise ValueError("profile_velocity cannot be negative")
        if configuration.profile_acceleration < 0:
            raise ValueError("profile_acceleration cannot be negative")

        if sdk_module is None:
            try:
                import dynamixel_sdk as sdk_module  # type: ignore[no-redef]
            except ImportError as error:
                raise DynamixelNeckError(
                    "dynamixel_sdk is not installed; run pip3 install dynamixel-sdk"
                ) from error

        self._sdk = sdk_module
        self._port = sdk_module.PortHandler(configuration.device)
        self._packet = sdk_module.PacketHandler(PROTOCOL_VERSION)
        self._sync_write = sdk_module.GroupSyncWrite(
            self._port,
            self._packet,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self._connected = False
        self._torque_enabled = False

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def connect(self) -> None:
        if self._connected:
            return
        if not self._port.openPort():
            raise DynamixelNeckError(
                f"could not open DYNAMIXEL port {self.configuration.device}"
            )
        self._connected = True

        if not self._port.setBaudRate(self.configuration.baudrate):
            self.close()
            raise DynamixelNeckError(
                f"could not set baudrate {self.configuration.baudrate}"
            )

        try:
            for motor_id in self._motor_ids:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                self._write1(motor_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
                self._write4(
                    motor_id,
                    ADDR_PROFILE_ACCELERATION,
                    self.configuration.profile_acceleration,
                )
                self._write4(
                    motor_id,
                    ADDR_PROFILE_VELOCITY,
                    self.configuration.profile_velocity,
                )

            for motor_id in self._motor_ids:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            self._torque_enabled = True
        except Exception:
            self.close()
            raise

    def write_angles(self, yaw_deg: float, pitch_deg: float) -> Tuple[int, int]:
        if not self._connected or not self._torque_enabled:
            raise DynamixelNeckError("neck is not connected with torque enabled")

        yaw_position = angle_to_position(
            yaw_deg,
            self.configuration.yaw_center,
            self.configuration.yaw_sign,
        )
        pitch_position = angle_to_position(
            pitch_deg,
            self.configuration.pitch_center,
            self.configuration.pitch_sign,
        )

        self._sync_write.clearParam()
        try:
            if not self._sync_write.addParam(
                self.configuration.yaw_id,
                yaw_position.to_bytes(4, byteorder="little", signed=False),
            ):
                raise DynamixelNeckError("could not add yaw goal to Sync Write")
            if not self._sync_write.addParam(
                self.configuration.pitch_id,
                pitch_position.to_bytes(4, byteorder="little", signed=False),
            ):
                raise DynamixelNeckError("could not add pitch goal to Sync Write")

            result = self._sync_write.txPacket()
            if result != self._sdk.COMM_SUCCESS:
                detail = self._packet.getTxRxResult(result)
                raise DynamixelNeckError(f"Sync Write failed: {detail}")
        finally:
            self._sync_write.clearParam()

        return yaw_position, pitch_position

    def torque_off(self) -> None:
        if not self._connected:
            self._torque_enabled = False
            return
        for motor_id in self._motor_ids:
            try:
                self._write1(motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            except DynamixelNeckError:
                # Attempt both axes even if one device is no longer responding.
                pass
        self._torque_enabled = False

    def close(self) -> None:
        if self._connected:
            self.torque_off()
            self._port.closePort()
        self._connected = False

    @property
    def _motor_ids(self) -> Tuple[int, int]:
        return self.configuration.yaw_id, self.configuration.pitch_id

    def _write1(self, motor_id: int, address: int, value: int) -> None:
        result, device_error = self._packet.write1ByteTxRx(
            self._port,
            motor_id,
            address,
            value,
        )
        self._check_result(motor_id, address, result, device_error)

    def _write4(self, motor_id: int, address: int, value: int) -> None:
        result, device_error = self._packet.write4ByteTxRx(
            self._port,
            motor_id,
            address,
            value,
        )
        self._check_result(motor_id, address, result, device_error)

    def _check_result(
        self,
        motor_id: int,
        address: int,
        result: int,
        device_error: int,
    ) -> None:
        if result != self._sdk.COMM_SUCCESS:
            detail = self._packet.getTxRxResult(result)
            raise DynamixelNeckError(
                f"motor {motor_id} write at {address} failed: {detail}"
            )
        if device_error:
            detail = self._packet.getRxPacketError(device_error)
            raise DynamixelNeckError(
                f"motor {motor_id} rejected write at {address}: {detail}"
            )

    def __enter__(self) -> "DynamixelNeck":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


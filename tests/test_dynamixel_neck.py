import unittest

from pc2.dynamixel_neck import (
    ADDR_GOAL_POSITION,
    LEN_GOAL_POSITION,
    DynamixelNeck,
    NeckConfiguration,
    angle_to_position,
)


class FakePort:
    def __init__(self, device):
        self.device = device
        self.baudrate = None
        self.closed = False

    def openPort(self):
        return True

    def setBaudRate(self, baudrate):
        self.baudrate = baudrate
        return True

    def closePort(self):
        self.closed = True


class FakePacket:
    def __init__(self):
        self.writes = []

    def write1ByteTxRx(self, _port, motor_id, address, value):
        self.writes.append((1, motor_id, address, value))
        return 0, 0

    def write4ByteTxRx(self, _port, motor_id, address, value):
        self.writes.append((4, motor_id, address, value))
        return 0, 0

    @staticmethod
    def getTxRxResult(result):
        return f"communication result {result}"

    @staticmethod
    def getRxPacketError(error):
        return f"device error {error}"


class FakeSyncWrite:
    def __init__(self, _port, _packet, address, length):
        self.address = address
        self.length = length
        self.parameters = {}
        self.last_transmission = None

    def addParam(self, motor_id, data):
        self.parameters[motor_id] = bytes(data)
        return True

    def txPacket(self):
        self.last_transmission = dict(self.parameters)
        return 0

    def clearParam(self):
        self.parameters.clear()


class FakeSdk:
    COMM_SUCCESS = 0

    def __init__(self):
        self.port = None
        self.packet = FakePacket()
        self.sync_write = None

    def PortHandler(self, device):
        self.port = FakePort(device)
        return self.port

    def PacketHandler(self, _protocol_version):
        return self.packet

    def GroupSyncWrite(self, port, packet, address, length):
        self.sync_write = FakeSyncWrite(port, packet, address, length)
        return self.sync_write


class AngleToPositionTests(unittest.TestCase):
    def test_zero_angle_is_center(self):
        self.assertEqual(angle_to_position(0.0, 2048, 1), 2048)

    def test_quarter_turn_uses_1024_counts(self):
        self.assertEqual(angle_to_position(90.0, 2048, 1), 3072)
        self.assertEqual(angle_to_position(90.0, 2048, -1), 1024)

    def test_position_is_bounded(self):
        self.assertEqual(angle_to_position(1000.0, 2048, 1), 4095)
        self.assertEqual(angle_to_position(-1000.0, 2048, 1), 0)

    def test_direction_sign_is_validated(self):
        with self.assertRaises(ValueError):
            angle_to_position(0.0, 2048, 0)


class DynamixelNeckTests(unittest.TestCase):
    def test_connect_and_sync_write_both_axes(self):
        sdk = FakeSdk()
        neck = DynamixelNeck(
            NeckConfiguration(yaw_sign=1, pitch_sign=-1),
            sdk_module=sdk,
        )

        neck.connect()
        yaw_position, pitch_position = neck.write_angles(90.0, 90.0)

        self.assertEqual(sdk.port.baudrate, 1_000_000)
        self.assertEqual(sdk.sync_write.address, ADDR_GOAL_POSITION)
        self.assertEqual(sdk.sync_write.length, LEN_GOAL_POSITION)
        self.assertEqual((yaw_position, pitch_position), (3072, 1024))
        self.assertEqual(
            sdk.sync_write.last_transmission,
            {
                1: (3072).to_bytes(4, "little"),
                2: (1024).to_bytes(4, "little"),
            },
        )

        neck.close()
        self.assertTrue(sdk.port.closed)
        self.assertFalse(neck.torque_enabled)


if __name__ == "__main__":
    unittest.main()

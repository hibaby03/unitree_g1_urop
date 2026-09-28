import math
import struct
import unittest

from pc2.active_camera_protocol import (
    FLAG_ORIENTATION_VALID,
    FLAG_TRACKED,
    MAGIC,
    PACKET_SIZE,
    PACKET_STRUCT,
    PacketError,
    TwoDofPoseMapper,
    decode_packet,
    is_newer_sequence,
    quaternion_multiply,
    quaternion_to_yaw_pitch,
)


def axis_angle(axis, angle_deg):
    angle_rad = math.radians(angle_deg)
    half = angle_rad / 2.0
    scale = math.sin(half)
    return (
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(half),
    )


def packet_bytes(sequence, orientation, flags=None):
    if flags is None:
        flags = FLAG_ORIENTATION_VALID | FLAG_TRACKED
    return PACKET_STRUCT.pack(
        MAGIC,
        1,
        flags,
        0,
        sequence,
        1_700_000_000_000_000_000,
        0.0,
        1.6,
        0.0,
        *orientation,
    )


class PacketTests(unittest.TestCase):
    def test_packet_size_is_fixed(self):
        self.assertEqual(PACKET_SIZE, 48)

    def test_decode_valid_packet(self):
        packet = decode_packet(packet_bytes(42, (0.0, 0.0, 0.0, 1.0)))
        self.assertEqual(packet.sequence, 42)
        self.assertTrue(packet.orientation_valid)
        self.assertTrue(packet.tracked)

    def test_rejects_bad_magic(self):
        data = bytearray(packet_bytes(1, (0.0, 0.0, 0.0, 1.0)))
        data[0:4] = b"NOPE"
        with self.assertRaises(PacketError):
            decode_packet(bytes(data))

    def test_rejects_bad_size(self):
        with self.assertRaises(PacketError):
            decode_packet(b"short")

    def test_rejects_non_unit_quaternion(self):
        with self.assertRaises(PacketError):
            decode_packet(packet_bytes(1, (0.0, 0.0, 0.0, 2.0)))


class SequenceTests(unittest.TestCase):
    def test_normal_and_wrapped_sequence(self):
        self.assertTrue(is_newer_sequence(5, 4))
        self.assertFalse(is_newer_sequence(4, 4))
        self.assertFalse(is_newer_sequence(3, 4))
        self.assertTrue(is_newer_sequence(0, 0xFFFFFFFF))


class AngleTests(unittest.TestCase):
    def assert_angle(self, actual, expected):
        self.assertAlmostEqual(math.degrees(actual), expected, places=4)

    def test_identity_is_zero(self):
        yaw, pitch = quaternion_to_yaw_pitch((0.0, 0.0, 0.0, 1.0))
        self.assert_angle(yaw, 0.0)
        self.assert_angle(pitch, 0.0)

    def test_look_right_is_positive_yaw(self):
        # In OpenXR coordinates, looking right is a negative rotation about +Y.
        yaw, pitch = quaternion_to_yaw_pitch(axis_angle((0.0, 1.0, 0.0), -30.0))
        self.assert_angle(yaw, 30.0)
        self.assert_angle(pitch, 0.0)

    def test_look_up_is_positive_pitch(self):
        yaw, pitch = quaternion_to_yaw_pitch(axis_angle((1.0, 0.0, 0.0), 25.0))
        self.assert_angle(yaw, 0.0)
        self.assert_angle(pitch, 25.0)

    def test_roll_does_not_change_yaw_pitch(self):
        yaw, pitch = quaternion_to_yaw_pitch(axis_angle((0.0, 0.0, 1.0), 40.0))
        self.assert_angle(yaw, 0.0)
        self.assert_angle(pitch, 0.0)

    def test_mapper_uses_first_packet_as_neutral_and_limits_output(self):
        mapper = TwoDofPoseMapper(yaw_limit_deg=20.0, smoothing=1.0)
        neutral = decode_packet(packet_bytes(1, (0.0, 0.0, 0.0, 1.0)))
        target = mapper.update(neutral)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.yaw_deg, 0.0)

        right_45 = decode_packet(
            packet_bytes(2, axis_angle((0.0, 1.0, 0.0), -45.0))
        )
        target = mapper.update(right_45)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.yaw_deg, 20.0)

    def test_combined_yaw_pitch_stays_close(self):
        yaw_q = axis_angle((0.0, 1.0, 0.0), -20.0)
        pitch_q = axis_angle((1.0, 0.0, 0.0), 15.0)
        combined = quaternion_multiply(yaw_q, pitch_q)
        yaw, pitch = quaternion_to_yaw_pitch(combined)
        self.assert_angle(yaw, 20.0)
        self.assert_angle(pitch, 15.0)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from host.head_pose_udp import build_packet_from_openxr_matrix
from pc2.active_camera_protocol import decode_packet, quaternion_to_yaw_pitch


class HeadPoseSenderTests(unittest.TestCase):
    def test_identity_matrix_encodes_identity_quaternion_and_position(self):
        pose = [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 1.6],
            [0.0, 0.0, 1.0, -0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]

        packet = decode_packet(
            build_packet_from_openxr_matrix(pose, 42, sender_time_ns=123)
        )

        self.assertEqual(packet.sequence, 42)
        self.assertEqual(packet.sender_time_ns, 123)
        self.assertTrue(packet.tracked)
        self.assertTrue(packet.position_valid)
        self.assertTrue(packet.orientation_valid)
        self.assertAlmostEqual(packet.position[0], 0.1, places=6)
        self.assertAlmostEqual(packet.position[1], 1.6, places=6)
        self.assertAlmostEqual(packet.position[2], -0.2, places=6)
        self.assertEqual(packet.orientation, (0.0, 0.0, 0.0, 1.0))

    def test_openxr_look_right_becomes_positive_yaw(self):
        angle = math.radians(-30.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        pose = [
            [cosine, 0.0, sine, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sine, 0.0, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        packet = decode_packet(build_packet_from_openxr_matrix(pose, 1))
        yaw, pitch = quaternion_to_yaw_pitch(packet.orientation)

        self.assertAlmostEqual(math.degrees(yaw), 30.0, places=4)
        self.assertAlmostEqual(math.degrees(pitch), 0.0, places=4)

    def test_rejects_uninitialized_shared_matrix(self):
        with self.assertRaises(ValueError):
            build_packet_from_openxr_matrix([[0.0] * 4 for _ in range(4)], 1)


if __name__ == "__main__":
    unittest.main()

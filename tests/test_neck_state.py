import math
import socket
import time
import unittest

from host import neck_state_receiver as host_side
from host.neck_state_receiver import (
    NeckStatePacketError,
    NeckStateReceiver,
    decode_neck_state_packet,
    neck_record_entries,
)
from host.teleop_record_dex3_tactile import build_states_actions
from pc2 import neck_state_sender as pc2_side
from pc2.neck_state_sender import NeckState, build_neck_state_packet


MOTOR_STATE = NeckState(
    pose_sequence=42,
    command_pc2_monotonic_ns=1_000_000_000,
    command_deg=(10.0, -5.0),
    goal_position=(2162, 1991),
    present_pc2_monotonic_ns=1_000_500_000,
    present_deg=(9.5, -4.5),
    present_position=(2156, 1997),
    torque_enabled=True,
)


class NeckStatePacketTests(unittest.TestCase):
    def test_both_sides_share_the_wire_format(self):
        self.assertEqual(pc2_side.PACKET_STRUCT.format, host_side.PACKET_STRUCT.format)
        self.assertEqual(pc2_side.MAGIC, host_side.MAGIC)
        self.assertEqual(pc2_side.VERSION, host_side.VERSION)
        self.assertEqual(pc2_side.PACKET_SIZE, 60)

    def test_round_trip_with_encoder(self):
        packet = decode_neck_state_packet(build_neck_state_packet(MOTOR_STATE))

        self.assertEqual(packet.pose_sequence, 42)
        self.assertEqual(packet.command_pc2_monotonic_ns, 1_000_000_000)
        self.assertEqual(packet.command_deg, (10.0, -5.0))
        self.assertEqual(packet.goal_position, (2162, 1991))
        self.assertEqual(packet.present_pc2_monotonic_ns, 1_000_500_000)
        self.assertEqual(packet.present_deg, (9.5, -4.5))
        self.assertEqual(packet.present_position, (2156, 1997))
        self.assertTrue(packet.torque_enabled)

    def test_monitor_only_has_no_encoder_or_goal(self):
        state = NeckState(pose_sequence=1, command_pc2_monotonic_ns=5, command_deg=(1.0, 2.0))
        packet = decode_neck_state_packet(build_neck_state_packet(state))

        self.assertFalse(packet.torque_enabled)
        self.assertIsNone(packet.goal_position)
        self.assertIsNone(packet.present_deg)
        self.assertIsNone(packet.present_pc2_monotonic_ns)

    def test_rejects_bad_packets(self):
        valid = build_neck_state_packet(MOTOR_STATE)
        with self.assertRaises(NeckStatePacketError):
            decode_neck_state_packet(valid[:-1])
        with self.assertRaises(NeckStatePacketError):
            decode_neck_state_packet(b"ACAM" + valid[4:])
        with self.assertRaises(ValueError):
            build_neck_state_packet(NeckState(0, 0, (math.nan, 0.0)))


class NeckRecordEntryTests(unittest.TestCase):
    def test_entries_are_radians_with_timestamps(self):
        packet = decode_neck_state_packet(build_neck_state_packet(MOTOR_STATE))
        state, action = neck_record_entries(packet, 3.0)

        self.assertAlmostEqual(state["qpos"][0], math.radians(9.5), places=6)
        self.assertAlmostEqual(action["qpos"][1], math.radians(-5.0), places=6)
        self.assertEqual(state["position_counts"], [2156, 1997])
        self.assertEqual(action["position_counts"], [2162, 1991])
        self.assertEqual(state["pc2_monotonic_ns"], 1_000_500_000)
        self.assertEqual(action["pose_sequence"], 42)
        self.assertEqual(action["age_ms"], 3.0)

    def test_no_packet_gives_empty_entries(self):
        state, action = neck_record_entries(None, None)
        self.assertEqual(state["qpos"], [])
        self.assertEqual(action["qpos"], [])

    def test_neck_is_added_to_states_and_actions(self):
        neck = neck_record_entries(None, None)
        states, actions = build_states_actions([0.0] * 14, [0.0] * 14, [0.0] * 14,
                                               [0.0] * 14, [0.0] * 14, [0.0] * 14, neck)
        self.assertIn("neck", states)
        self.assertIn("neck", actions)


class NeckStateReceiverTests(unittest.TestCase):
    def test_receives_over_loopback_and_ignores_older_commands(self):
        receiver = NeckStateReceiver("127.0.0.1", 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.assertEqual(receiver.latest(), (None, None))
            older = NeckState(pose_sequence=41, command_pc2_monotonic_ns=1,
                              command_deg=(0.0, 0.0))
            sender.sendto(b"junk", ("127.0.0.1", receiver.port))
            sender.sendto(build_neck_state_packet(MOTOR_STATE), ("127.0.0.1", receiver.port))
            sender.sendto(build_neck_state_packet(older), ("127.0.0.1", receiver.port))

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and receiver.rejected < 1:
                time.sleep(0.005)
            time.sleep(0.05)
            packet, age_ms = receiver.latest()
        finally:
            sender.close()
            receiver.close()

        self.assertEqual(receiver.rejected, 1)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.pose_sequence, 42)
        self.assertGreaterEqual(age_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

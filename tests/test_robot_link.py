import socket
import time
import unittest

from host import robot_link as host
from pc2 import robot_bridge as pc2


ARM_Q = [0.1 * i for i in range(14)]
ARM_DQ = [0.01 * i for i in range(14)]
LH_Q = [0.2 * i for i in range(7)]
RH_Q = [0.3 * i for i in range(7)]
LH_A = [0.4 * i for i in range(7)]
RH_A = [0.5 * i for i in range(7)]


def make_tactile(offset):
    return {"pressure": [[offset + i for i in range(12)], [offset + 100 + i for i in range(12)]],
            "temperature": [[30.0 + i for i in range(12)], [31.0 + i for i in range(12)]],
            "lost": [0, 2]}


class WireFormatMatchTests(unittest.TestCase):
    def test_both_sides_share_the_struct_formats(self):
        for name in ("STATE_STRUCT", "COMMAND_STRUCT", "TACTILE_HEADER", "TACTILE_SENSOR"):
            self.assertEqual(getattr(host, name).format, getattr(pc2, name).format, name)
        for name in ("STATE_MAGIC", "TACTILE_MAGIC", "COMMAND_MAGIC", "VERSION",
                     "NUM_ARM", "NUM_HAND", "TACTILE_VALUES",
                     "DEFAULT_STATE_PORT", "DEFAULT_COMMAND_PORT"):
            self.assertEqual(getattr(host, name), getattr(pc2, name), name)


class StateRoundTripTests(unittest.TestCase):
    def test_full_state(self):
        packet = pc2.encode_state(ARM_Q, ARM_DQ, LH_Q, RH_Q, LH_A, RH_A, 7, 123)
        state = host.decode_state(packet)
        self.assertEqual(state.sequence, 7)
        self.assertEqual(state.pc2_monotonic_ns, 123)
        for got, want in ((state.arm_q, ARM_Q), (state.arm_dq, ARM_DQ),
                          (state.left_hand_q, LH_Q), (state.right_hand_q, RH_Q),
                          (state.left_hand_action, LH_A), (state.right_hand_action, RH_A)):
            self.assertEqual([round(v, 5) for v in got], [round(v, 5) for v in want])

    def test_arm_only(self):
        packet = pc2.encode_state(ARM_Q, ARM_DQ, [], [], [], [], 1, 5)
        state = host.decode_state(packet)
        self.assertTrue(state.arm_q)
        self.assertEqual(state.left_hand_q, [])

    def test_rejects_wrong_sizes_and_nonfinite(self):
        with self.assertRaises(pc2.RobotBridgeError):
            pc2.encode_state(ARM_Q[:5], ARM_DQ, LH_Q, RH_Q, LH_A, RH_A, 1, 1)
        with self.assertRaises(pc2.RobotBridgeError):
            pc2.encode_state([float("nan")] * 14, ARM_DQ, [], [], [], [], 1, 1)

    def test_host_rejects_bad_state(self):
        with self.assertRaises(host.RobotLinkError):
            host.decode_state(b"XXXX" + b"\x00" * (host.STATE_STRUCT.size - 4))
        with self.assertRaises(host.RobotLinkError):
            host.decode_state(b"short")


class TactileRoundTripTests(unittest.TestCase):
    def test_both_hands(self):
        packet = pc2.encode_tactile(make_tactile(0), make_tactile(1000), 3, 9)
        decoded = host.decode_tactile(packet)
        self.assertEqual(decoded["left_ee"]["pressure"][0][0], 0)
        self.assertEqual(decoded["right_ee"]["pressure"][1][0], 1100)
        self.assertEqual(decoded["left_ee"]["lost"], [0, 2])

    def test_missing_hand_is_none(self):
        packet = pc2.encode_tactile(make_tactile(0), None, 1, 1)
        decoded = host.decode_tactile(packet)
        self.assertIsNotNone(decoded["left_ee"])
        self.assertIsNone(decoded["right_ee"])

    def test_truncated_packet_raises(self):
        packet = pc2.encode_tactile(make_tactile(0), make_tactile(0), 1, 1)
        with self.assertRaises(host.RobotLinkError):
            host.decode_tactile(packet[:-10])


class CommandRoundTripTests(unittest.TestCase):
    def test_full_command(self):
        packet = host.encode_command(ARM_Q, ARM_DQ, LH_Q, RH_Q, 4, 11)
        command = pc2.decode_command(packet)
        self.assertEqual(command["sequence"], 4)
        self.assertEqual(command["host_monotonic_ns"], 11)
        self.assertEqual([round(v, 5) for v in command["arm_q"]], [round(v, 5) for v in ARM_Q])
        self.assertEqual([round(v, 5) for v in command["arm_tau"]], [round(v, 5) for v in ARM_DQ])
        self.assertEqual([round(v, 5) for v in command["left_hand_q"]], [round(v, 5) for v in LH_Q])
        self.assertEqual(len(command["right_hand_q"]), 7)

    def test_rejects_wrong_sizes(self):
        with self.assertRaises(host.RobotLinkError):
            host.encode_command(ARM_Q[:3], ARM_DQ, LH_Q, RH_Q, 1, 1)
        with self.assertRaises(host.RobotLinkError):
            host.encode_command(ARM_Q, ARM_DQ, LH_Q[:3], RH_Q, 1, 1)


class RobotLinkLoopbackTests(unittest.TestCase):
    def test_receive_state_tactile_and_send_command(self):
        # Bind the link, then have a fake "bridge" socket send to it.
        link = host.RobotLink("127.0.0.1", state_port=0, command_port=0)
        self.addCleanup(link.close)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as bridge:
            bridge.sendto(pc2.encode_state(ARM_Q, ARM_DQ, LH_Q, RH_Q, LH_A, RH_A, 1, 1),
                          ("127.0.0.1", link.state_port))
            bridge.sendto(pc2.encode_tactile(make_tactile(0), make_tactile(5), 1, 1),
                          ("127.0.0.1", link.state_port))
            deadline = time.monotonic() + 2
            while link.latest_state()[0] is None and time.monotonic() < deadline:
                time.sleep(0.01)
        state, age = link.latest_state()
        self.assertIsNotNone(state)
        self.assertLess(age, 2000)
        self.assertEqual(len(state.arm_q), 14)
        touch = link.tactile_snapshot()
        self.assertIsNotNone(touch["left_ee"])
        self.assertIn("age_ms", touch["left_ee"])

    def test_send_command_reaches_a_bound_socket(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as bridge:
            bridge.bind(("127.0.0.1", 0))
            bridge.settimeout(2)
            port = bridge.getsockname()[1]
            link = host.RobotLink("127.0.0.1", state_port=0, command_port=port)
            self.addCleanup(link.close)
            link.send_command(ARM_Q, ARM_DQ, LH_Q, RH_Q)
            datagram, _ = bridge.recvfrom(4096)
        command = pc2.decode_command(datagram)
        self.assertEqual(len(command["arm_q"]), 14)
        self.assertEqual(len(command["left_hand_q"]), 7)


if __name__ == "__main__":
    unittest.main()

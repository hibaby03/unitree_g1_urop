import io
import json
import math
import socket
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from pc2 import head_pose_receiver as receiver
from pc2.active_camera_protocol import PACKET_STRUCT
from pc2.dynamixel_neck import DynamixelNeckError


def pose(sequence, yaw_deg=0.0, flags=5):
    half = math.radians(-yaw_deg) / 2
    return PACKET_STRUCT.pack(b"ACAM", 1, flags, 0, sequence, 1,
                              0, 0, 0, 0, math.sin(half), 0, math.cos(half))


class RecoveryRampTests(unittest.TestCase):
    def test_recovery_starts_at_held_goal_and_limits_each_axis(self):
        ramp = receiver.RecoveryRamp(30)
        self.assertEqual(ramp.apply(10, -5, 0), (10, -5))
        ramp.hold()
        self.assertEqual(ramp.apply(-20, 25, 100), (10, -5))
        self.assertEqual(ramp.apply(-20, 25, 100.05),
                         (10 - 30 * (100.05 - 100), -5 + 30 * (100.05 - 100)))
        previous = ramp.last_command
        command = ramp.apply(-20, 25, 200)
        self.assertAlmostEqual(command[0], previous[0] - 3)
        self.assertAlmostEqual(command[1], previous[1] + 3)

    def test_repeated_loss_reanchors_at_latest_command_and_catches_up(self):
        ramp = receiver.RecoveryRamp(30)
        ramp.apply(0, 0, 0)
        ramp.hold()
        ramp.apply(9, 0, 10)
        ramp.apply(9, 0, 10.1)
        held = ramp.last_command
        ramp.hold()
        self.assertEqual(ramp.apply(9, 0, 20), held)
        for i in range(1, 6):
            ramp.apply(9, 0, 20 + i * .1)
        self.assertEqual(ramp.last_command, (9, 0))
        self.assertFalse(ramp.recovering)


class ReceiverTimeoutTests(unittest.TestCase):
    def run_receiver(self, events, write_error=None):
        now = [0.0]
        motor = Mock()
        motor.torque_enabled = True
        motor.write_angles.return_value = (2048, 2048)
        motor.read_present_angles.return_value = ((0, 0), (2048, 2048))
        if write_error:
            motor.write_angles.side_effect = write_error
        state_sender = Mock()
        sock = Mock()
        holds = []

        def recv(_size):
            instant, message = next(events)
            now[0] = instant
            # Holding must not close the motor or issue new movement commands.
            if isinstance(message, socket.timeout):
                motor.close.assert_not_called()
                motor.torque_off.assert_not_called()
                holds.append(motor.write_angles.call_count)
            if isinstance(message, BaseException):
                raise message
            return message, ("127.0.0.1", 12345)

        sock.recvfrom.side_effect = recv
        output, errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(receiver, "enabled", return_value=False))
            stack.enter_context(patch.object(receiver.socket, "socket", return_value=sock))
            stack.enter_context(patch.object(receiver, "DynamixelNeck", return_value=motor))
            stack.enter_context(patch.object(receiver, "NeckStateUdpSender", return_value=state_sender))
            stack.enter_context(patch.object(receiver.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(receiver.time, "monotonic_ns",
                                            side_effect=lambda: int(now[0] * 1e9)))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(errors))
            args = receiver.parse_args([])
            if write_error:
                with self.assertRaises(DynamixelNeckError):
                    receiver.run(args)
            else:
                self.assertEqual(receiver.run(args), 0)
        motor.close.assert_called_once()  # Ctrl+C / motor failure still cleans up.
        sock.close.assert_called_once()
        records = [json.loads(line) for line in (output.getvalue() + errors.getvalue()).splitlines()]
        return motor, state_sender, records, holds

    def test_holds_through_silence_and_invalid_packets_then_resumes_slowly(self):
        events = iter([
            (0, pose(1)),
            (.3, socket.timeout()),
            (1.1, socket.timeout()),
            (5, socket.timeout()),
            (5.1, b"invalid"),
            (5.2, pose(1)),  # duplicate cannot renew tracking
            (5.3, pose(2, 60, flags=0)),  # untracked cannot renew it either
            (5.4, pose(3, 60)),
            (5.5, pose(4, 60)),
            (6.6, socket.timeout()),
            (7, socket.timeout()),
            (8, pose(5, -60)),
            (8.1, pose(6, -60)),
            (8.2, KeyboardInterrupt()),
        ])
        motor, feedback, records, holds = self.run_receiver(events)
        self.assertEqual(holds[:3], [1, 1, 1])
        commands = [call.args for call in motor.write_angles.call_args_list]
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0], (0, 0))
        self.assertEqual(commands[1], (0, 0))
        self.assertAlmostEqual(commands[2][0], 3)
        self.assertEqual(commands[3], commands[2])
        self.assertAlmostEqual(commands[4][0], 0)
        sent = [call.args[1].command_deg for call in feedback.send.call_args_list]
        self.assertEqual(sent, commands)  # record actual ramped commands, not raw targets
        timeouts = [r for r in records if r.get("event") == "motor_timeout"]
        resumed = [r for r in records if r.get("event") == "tracking_resumed"]
        self.assertEqual(len(timeouts), 2)
        self.assertTrue(all(r["action"] == "hold_last_goal" for r in timeouts))
        self.assertEqual(len(resumed), 2)
        self.assertTrue(any(r.get("event") == "stale" for r in records))

    def test_initial_wait_does_not_start_watchdog(self):
        _, _, records, _ = self.run_receiver(iter([
            (5, socket.timeout()), (20, socket.timeout()), (21, KeyboardInterrupt())]))
        self.assertFalse(any(r.get("event") == "motor_timeout" for r in records))

    def test_motor_errors_still_propagate_and_close(self):
        self.run_receiver(iter([(0, pose(1))]), DynamixelNeckError("motor failure"))

    def test_rejects_nonfinite_resume_speed(self):
        for value in ("nan", "inf", "-1", "0"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    receiver.parse_args(["--resume-speed-deg-s", value])


if __name__ == "__main__":
    unittest.main()

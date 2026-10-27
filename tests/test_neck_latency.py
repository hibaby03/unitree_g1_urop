import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from host.head_pose_udp import HeadPoseUdpSender, build_packet_from_openxr_matrix
from host.analyze_neck_latency import analyze, read_log
from host.neck_clock import ClockProbe, REPLY, sample
from host.neck_trace import SharedPose, TraceWriter, install_televuer_hook, send_televuer_pose
from pc2 import head_pose_receiver as receiver
from pc2.active_camera_protocol import decode_packet
from pc2.dynamixel_neck import DynamixelNeckError

IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class TimingTests(unittest.TestCase):
    def test_real_clock_probe_loopback_records_four_timestamps(self):
        from pc2.frame_timing import ClockSyncServer
        with tempfile.TemporaryDirectory() as d:
            writer = TraceWriter("host", d)
            server = ClockSyncServer("127.0.0.1", 0)
            client = ClockProbe("127.0.0.1", writer, server.port)
            try:
                deadline = time.monotonic() + 2
                records = []
                while time.monotonic() < deadline:
                    time.sleep(.05)
                    records, _ = read_log(writer.path)
                    if any(r.get("event") == "clock_sample" for r in records):
                        break
            finally:
                client.close()
                server.close()
                writer.close()
            records, _ = read_log(writer.path)
            clocks = [r for r in records if r.get("event") == "clock_sample"]
            self.assertTrue(clocks)
            for r in clocks:
                self.assertLessEqual(abs(r["offset_ns"]), r["uncertainty_ns"] + 1)

    def test_neck_only_preserves_pose_id_when_resending(self):
        from host.run_teleop_with_neck import send_fresh_head_pose
        tracker = SharedPose()
        now = time.monotonic_ns()
        tracker.update(IDENTITY, now, time.time_ns())
        sender = Mock()
        self.assertTrue(send_fresh_head_pose(None, sender, trace_pose=tracker))
        self.assertTrue(send_fresh_head_pose(None, sender, trace_pose=tracker))
        self.assertEqual([c.kwargs["pose_stamp"]["pose_id"] for c in sender.send_openxr_matrix.call_args_list], [1, 1])
        tracker.update(IDENTITY, now - 1_000_000_000, time.time_ns())
        self.assertFalse(send_fresh_head_pose(None, sender, trace_pose=tracker))

    def test_send_failure_is_recorded_and_does_not_advance_sequence(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NECK_TRACE_DIR": d}), \
                patch("host.head_pose_udp.ClockProbe"), patch("host.head_pose_udp.socket.socket") as factory:
            sender = HeadPoseUdpSender("127.0.0.1")
            factory.return_value.sendto.side_effect = OSError("test send failure")
            with self.assertRaises(OSError):
                sender.send_openxr_matrix(IDENTITY)
            self.assertEqual(sender._sequence, 0)
            sender.close()
            records, _ = read_log(sender._trace.path)
            record = next(r for r in records if r.get("event") == "host_packet")
            self.assertEqual(record["send_error"], "test send failure")
            self.assertGreaterEqual(record["send_end_host_ns"], record["send_start_host_ns"])

    def test_offset_sign_asymmetry_and_invalid_samples(self):
        # PC2 is ahead by 1000 ns; symmetric 20 ns transit and 5 ns server work.
        s = sample(100, 1120, 1125, 145)
        self.assertEqual(s["offset_ns"], 1000)
        self.assertEqual(s["rtt_ns"], 40)
        self.assertEqual(s["uncertainty_ns"], 20)
        asymmetric = sample(100, 1130, 1135, 145)
        self.assertEqual(asymmetric["offset_ns"], 1010)
        for values in ((10, 20, 19, 30), (30, 20, 21, 10), (10, 20, 100, 30)):
            with self.assertRaises(ValueError):
                sample(*values)

    def test_snapshot_preserves_integer_ns_and_identical_event_ids(self):
        tracker = SharedPose()
        stamp = 2**54 + 17  # would lose precision in a double
        self.assertIsNone(tracker.snapshot())
        tracker.update(IDENTITY, stamp, stamp + 1)
        a = tracker.snapshot()
        tracker.update(IDENTITY, stamp + 3, stamp + 4)
        b = tracker.snapshot()
        self.assertEqual(a[1]["pose_rx_host_ns"], stamp)
        self.assertEqual([a[1]["pose_id"], b[1]["pose_id"]], [1, 2])
        self.assertEqual(a[0], b[0])

    def test_televuer_hook_keeps_original_callback_and_atomic_pose(self):
        class Viewer:
            def __init__(self):
                self.calls = 0
            async def on_cam_move(self, event, session):
                self.calls += 1
        with patch.dict(os.environ, {"NECK_TRACE_DIR": "unused"}), \
                patch("multiprocessing.get_start_method", return_value="fork"):
            install_televuer_hook(Viewer)
            viewer = Viewer()
            event = SimpleNamespace(value={"camera": {"matrix": [v for row in IDENTITY for v in row]}})
            asyncio.run(viewer.on_cam_move(event, None))
            asyncio.run(viewer.on_cam_move(event, None))
            sender = Mock()
            send_televuer_pose(sender, viewer)
            self.assertEqual(viewer.calls, 2)
            self.assertEqual(sender.send_openxr_matrix.call_args.kwargs["pose_stamp"]["pose_id"], 2)
            self.assertEqual(sender.send_openxr_matrix.call_args.args[0], IDENTITY)

    def test_writer_flush_and_drop_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TraceWriter("test", d)
            writer.emit({"event": "example", "integer": 2**60 + 1})
            with patch.object(writer._queue, "put_nowait", side_effect=__import__("queue").Full):
                writer.emit({"event": "dropped"})
            writer.close()
            records, bad = read_log(writer.path)
            self.assertEqual(bad, 0)
            self.assertEqual(records[1]["integer"], 2**60 + 1)
            self.assertEqual(records[-1]["dropped"], 1)

    def test_probe_ignores_wrong_sequence_and_echo(self):
        writer = Mock()
        sock = Mock()
        probe = object.__new__(ClockProbe)
        import threading
        probe.writer, probe.sock, probe.stop = writer, sock, threading.Event()
        request = []
        sock.send.side_effect = lambda data: request.append(data)
        def recv(_):
            from host.neck_clock import REQUEST
            _, _, _, _, seq, t0 = REQUEST.unpack(request[0])
            count = sock.recv.call_count
            if count == 1:
                return REPLY.pack(b"ACLK", 1, 0, 0, seq + 1, t0, t0 + 10, t0 + 11)
            if count == 2:
                return REPLY.pack(b"ACLK", 1, 0, 0, seq, t0 + 1, t0 + 10, t0 + 11)
            probe.stop.set()
            return REPLY.pack(b"ACLK", 1, 0, 0, seq, t0, t0 + 10, t0 + 11)
        sock.recv.side_effect = recv
        probe._loop()
        writer.emit.assert_called_once()
        self.assertEqual(sock.recv.call_count, 3)

    def test_complete_mock_motor_pipeline_and_analysis(self):
        for encoder_error in (False, True):
            with self.subTest(encoder_error=encoder_error), tempfile.TemporaryDirectory() as d, ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"NECK_TRACE_DIR": d}))
                stack.enter_context(patch("host.head_pose_udp.ClockProbe"))
                send_socket = Mock()
                stack.enter_context(patch("host.head_pose_udp.socket.socket", return_value=send_socket))
                sender = HeadPoseUdpSender("127.0.0.1")
                now = time.monotonic_ns()
                sender._trace.emit(sample(now - 5000, now - 4000, now - 3000, now - 2000))
                pose = {"pose_id": 7, "pose_rx_host_ns": now, "pose_rx_host_wall_ns": time.time_ns()}
                sender.send_openxr_matrix(IDENTITY, pose_stamp=pose)
                sender.send_openxr_matrix(IDENTITY, pose_stamp=pose)
                packets = [c.args[0] for c in send_socket.sendto.call_args_list]
                sender.close()
                self.assertEqual([len(p) for p in packets], [48, 48])
                self.assertEqual([decode_packet(p).sequence for p in packets], [0, 1])
                receive_socket = Mock()
                # Duplicate packet exercises early continue; all packets must be logged.
                receive_socket.recvfrom.side_effect = [
                    (packets[0], ("127.0.0.1", 40000)),
                    (packets[1], ("127.0.0.1", 40000)),
                    (packets[1], ("127.0.0.1", 40000)), KeyboardInterrupt]
                stack.enter_context(patch.object(receiver.socket, "socket", return_value=receive_socket))
                motor = stack.enter_context(patch.object(receiver, "DynamixelNeck")).return_value
                motor.torque_enabled = True
                motor.write_angles.return_value = (2048, 2048)
                motor.read_present_angles.return_value = ((0.0, 0.0), (2048, 2048))
                if encoder_error:
                    motor.read_present_angles.side_effect = DynamixelNeckError("simulated timeout")
                stack.enter_context(patch.object(receiver, "NeckStateUdpSender"))
                clock_server = stack.enter_context(patch.object(receiver, "ClockSyncServer")).return_value
                clock_server.port = 5010
                stack.enter_context(patch("builtins.print"))
                self.assertEqual(receiver.run(receiver.parse_args(["--enable-motor"])), 0)
                motor.close.assert_called_once()
                host, _ = read_log(next(Path(d).glob("host-*.jsonl")))
                pc2, _ = read_log(next(Path(d).glob("pc2-*.jsonl")))
                rows, summary = analyze(host, pc2)
                self.assertEqual(len(rows), 3)
                self.assertEqual(summary["repeated_pose_packets"], 1)
                self.assertEqual(summary["matched_host_packets"], 2)
                self.assertEqual(rows[2]["status"], "old_sequence")
                for row in rows[:2]:
                    timestamps = [row[k] for k in ("rx_pc2_ns", "target_ready_pc2_ns", "write_start_pc2_ns",
                                  "write_end_pc2_ns", "read_start_pc2_ns", "read_end_pc2_ns", "loop_end_pc2_ns")]
                    self.assertEqual(timestamps, sorted(timestamps))
                    self.assertEqual(row["pose_id"], 7)
                    self.assertEqual(row["clock_status"], "estimated")
                self.assertEqual(summary["encoder_error_rows"], 2 if encoder_error else 0)

    def test_analyzer_keeps_local_times_without_clock_and_rejects_stale(self):
        h = {"event": "host_packet", "packet_uid": "x", "pose_id": 1,
             "send_start_host_ns": 20_000_000_000, "send_end_host_ns": 20_000_100_000}
        p = {"event": "pc2_packet", "packet_uid": "x", "rx_pc2_ns": 21_000_000_000,
             "target_ready_pc2_ns": 21_001_000_000, "loop_end_pc2_ns": 21_002_000_000}
        stale = sample(100, 1120, 1125, 145)
        for clocks in ([], [stale]):
            rows, _ = analyze([h, *clocks], [p])
            self.assertEqual(rows[0]["pc2_service_ms"], 2)
            self.assertEqual(rows[0]["clock_status"], "unavailable_or_stale")
            self.assertNotIn("send_to_pc2_dequeue_ms", rows[0])


if __name__ == "__main__":
    unittest.main()

"""Offline checks for isolation, stereo scene configuration, and resource cleanup.

TeleVuer/network/process boundaries are faked; these do not verify Quest rendering.
"""

import asyncio
from contextlib import ExitStack
import importlib
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from host import run_teleop_with_neck as launcher


class VideoOnlyTests(unittest.TestCase):
    def test_video_branch_bypasses_full_teleop_and_neck_validation(self):
        with patch.object(launcher, "run_video_only", return_value=0) as video, \
                patch.object(launcher.runpy, "run_path") as full, \
                patch.object(launcher.os, "chdir") as chdir:
            self.assertEqual(launcher.main(["--video-only", "--neck-pose-port", "0"]), 0)
        video.assert_called_once()
        full.assert_not_called()
        chdir.assert_not_called()

    def test_launcher_import_does_not_import_udp_sender(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "head_pose_udp":
                self.fail("video-only launcher imported the UDP sender")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            importlib.reload(launcher)

    def test_video_rejects_unrecognized_teleop_options(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            launcher.parse_launcher_args(["--video-only", "--xr-mode", "hand"])

    def test_full_teleop_arguments_still_pass_through(self):
        args, remaining = launcher.parse_launcher_args(["--xr-mode", "hand"])
        self.assertFalse(args.video_only)
        self.assertEqual(remaining, ["--xr-mode", "hand"])

    def _environment(self, stack, process, fail_init=False):
        instances = []
        package = ModuleType("televuer")
        schemas = ModuleType("vuer.schemas")
        schemas.WebRTCStereoVideoPlane = Mock(side_effect=lambda **kw: kw)

        class FakeTeleVuer:
            def __init__(self, binocular, use_hand_tracking, img_shape, img_shm_name,
                         cert_file=None, key_file=None, webrtc=False):
                self.process = process
                self.settings = (binocular, use_hand_tracking, img_shape, webrtc)
                self.shm_name = img_shm_name
                instances.append(self)
                if fail_init:
                    raise RuntimeError("initialization failed")

            async def main_image_webrtc(self, session, fps=60):
                raise AssertionError("original video method must not run")

            async def on_cam_move(self, event, session, fps=60):
                pass

        package.TeleVuer = FakeTeleVuer
        stack.enter_context(patch.dict(sys.modules, {
            "televuer": package, "vuer": ModuleType("vuer"), "vuer.schemas": schemas,
        }))
        stack.enter_context(patch("multiprocessing.get_start_method", return_value="fork"))
        stack.enter_context(patch("ssl.SSLContext.load_cert_chain"))
        stack.enter_context(patch("builtins.print"))
        original_path = sys.path[:]
        stack.callback(lambda: sys.path.__setitem__(slice(None), original_path))
        repo = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        (repo / "teleop" / "televuer" / "src").mkdir(parents=True)
        args, _ = launcher.parse_launcher_args([
            "--video-only", "--xr-repo", str(repo),
            "--video-cert", __file__, "--video-key", __file__,
        ])
        return args, instances, schemas

    def test_unexpected_server_exit_cleans_memory_and_scene_is_stereo_only(self):
        process = Mock(pid=123, exitcode=1)
        process.is_alive.return_value = False
        previous_handler = signal.getsignal(signal.SIGTERM)
        with ExitStack() as stack:
            args, instances, schemas = self._environment(stack, process)
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                launcher.run_video_only(args)
            viewer = instances[0]
            self.assertEqual(viewer.settings, (True, False, (720, 2560, 3), True))
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=viewer.shm_name)

            async def check_scene():
                session = Mock()
                with patch("asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
                    with self.assertRaises(asyncio.CancelledError):
                        await viewer.main_image_webrtc(session)
                session.upsert.assert_called_once()
                node = session.upsert.call_args.args[0]
                self.assertEqual(node["src"], "https://192.168.123.164:60001/offer")
                self.assertEqual(node["aspect"], 1280 / 720)
                self.assertEqual(node["iceServer"], {})
                self.assertNotIn("layers", node)
                self.assertEqual(session.upsert.call_args.kwargs, {"to": "bgChildren"})
                schemas.WebRTCStereoVideoPlane.assert_called_once()

            asyncio.run(check_scene())
        process.close.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_interrupt_stops_server_and_releases_memory(self):
        process = Mock(pid=123)
        process.is_alive.side_effect = [True, True, False, False]
        process.join.side_effect = [KeyboardInterrupt, None]
        with ExitStack() as stack:
            args, instances, _ = self._environment(stack, process)
            self.assertEqual(launcher.run_video_only(args), 0)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=instances[0].shm_name)
        process.terminate.assert_called_once()
        process.close.assert_called_once()

    def test_partial_initialization_also_cleans_process_and_memory(self):
        process = Mock()
        process.is_alive.return_value = False
        with ExitStack() as stack:
            args, instances, _ = self._environment(stack, process, fail_init=True)
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                launcher.run_video_only(args)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=instances[0].shm_name)
        process.close.assert_called_once()


class NeckOnlyTests(unittest.TestCase):
    _environment = VideoOnlyTests._environment

    def test_none_bypasses_robot_entrypoint(self):
        with patch.object(launcher, "run_video_only", return_value=0) as viewer, \
                patch.object(launcher.runpy, "run_path") as full, \
                patch.object(launcher.os, "chdir") as chdir:
            self.assertEqual(launcher.main(["--input-mode=none"]), 0)
        self.assertEqual(viewer.call_args.args[0].input_mode, "none")
        full.assert_not_called()
        chdir.assert_not_called()

    def test_argument_conflicts_and_port_are_rejected(self):
        for arguments in (
            ["--input-mode=none", "--video-only"],
            ["--input-mode=none", "--record"],
            ["--input-mode=none", "--arm=G1_29"],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr"), self.assertRaises(SystemExit):
                launcher.parse_launcher_args(arguments)
        with patch.object(launcher, "run_video_only") as viewer, self.assertRaises(ValueError):
            launcher.main(["--input-mode=none", "--neck-pose-port=0"])
        viewer.assert_not_called()

    def test_hand_and_controller_are_forwarded(self):
        for mode in ("hand", "controller"):
            args, remaining = launcher.parse_launcher_args([f"--input-mode={mode}", "--record"])
            self.assertEqual(remaining, ["--input-mode", mode, "--record"])
            self.assertFalse(args.video_only)

    def test_stale_or_uninitialized_pose_does_not_renew_watchdog(self):
        shared = multiprocessing.Array("d", 17, lock=True)
        sender = Mock()
        with patch.object(launcher.time, "monotonic", return_value=10.0):
            self.assertFalse(launcher.send_fresh_head_pose(shared, sender))
            shared[16] = 9.0
            self.assertFalse(launcher.send_fresh_head_pose(shared, sender))
        sender.send_openxr_matrix.assert_not_called()

    def test_none_forwards_camera_events_and_cleans_up_on_interrupt(self):
        from host import head_pose_udp
        import builtins
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.startswith(("unitree_sdk2py", "teleop.robot_control", "teleimager")):
                self.fail(f"neck-only mode imported robot/recording dependency: {name}")
            return original_import(name, *args, **kwargs)

        process = Mock(pid=123)
        process.is_alive.side_effect = [True, True, True, False, False]
        with ExitStack() as stack:
            args, instances, _ = self._environment(stack, process)
            args.video_only = False
            args.input_mode = "none"
            sender = stack.enter_context(patch.object(head_pose_udp, "HeadPoseUdpSender")).return_value
            stack.enter_context(patch("builtins.__import__", side_effect=guarded))

            def join(timeout):
                if not getattr(join, "called", False):
                    join.called = True
                    # Column-major pose with translation; unchanged poses must
                    # still refresh when a new CAMERA_MOVE arrives.
                    event = SimpleNamespace(value={"camera": {"matrix": [
                        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1,
                    ]}})
                    asyncio.run(instances[0].on_cam_move(event, None))
                else:
                    process.join.side_effect = None
                    raise KeyboardInterrupt

            process.join.side_effect = join
            self.assertEqual(launcher.run_video_only(args), 0)
            sender.send_openxr_matrix.assert_called_once_with([
                [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1],
            ])
            sender.close.assert_called_once()
            sample = list(instances[0].neck_pose_shared[:])
            bad_event = SimpleNamespace(value={"camera": {"matrix": [0] * 16}})
            asyncio.run(instances[0].on_cam_move(bad_event, None))
            self.assertEqual(list(instances[0].neck_pose_shared[:]), sample)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=instances[0].shm_name)
        process.terminate.assert_called_once()
        process.close.assert_called_once()

    def test_sender_creation_failure_stops_video_and_releases_memory(self):
        from host import head_pose_udp
        process = Mock(pid=123)
        process.is_alive.return_value = False
        with ExitStack() as stack:
            args, instances, _ = self._environment(stack, process)
            args.input_mode = "none"
            stack.enter_context(patch.object(head_pose_udp, "HeadPoseUdpSender", side_effect=OSError("socket failed")))
            with self.assertRaisesRegex(OSError, "socket failed"):
                launcher.run_video_only(args)
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=instances[0].shm_name)
        process.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

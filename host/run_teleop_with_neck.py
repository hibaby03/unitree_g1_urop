#!/usr/bin/env python3
"""Run full teleop, video only, or stereo video with head-driven neck control.

This launcher replaces TeleVuerWrapper only inside the current Python process.
It does not start a second Vuer server and does not modify xr_teleoperate files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from typing import Optional, Sequence


def positive_rate(value: str) -> float:
    parsed = float(value)
    if not 1.0 <= parsed <= 240.0:
        raise argparse.ArgumentTypeError("must be in 1..240")
    return parsed


def parse_launcher_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--xr-repo",
        default="~/hckang/xr_teleoperate",
        help="xr_teleoperate repository root",
    )
    parser.add_argument("--neck-pose-ip", default="192.168.123.164")
    parser.add_argument("--neck-pose-port", type=int, default=5005)
    parser.add_argument("--neck-pose-rate", type=positive_rate, default=60.0)
    parser.add_argument(
        "--input-mode", choices=("hand", "controller", "none"), default=None,
        help="none: stereo video and neck only; hand/controller: full robot teleop",
    )
    parser.add_argument(
        "--video-only", action="store_true",
        help="Display PC2 stereo video without robot control, pose UDP, or recording",
    )
    parser.add_argument(
        "--video-offer-url", default="https://192.168.123.164:60001/offer",
        help="PC2 HTTPS WebRTC offer endpoint (video-only or input-mode=none)",
    )
    parser.add_argument("--video-cert", help="Host TLS certificate; or XR_TELEOP_CERT")
    parser.add_argument("--video-key", help="Host TLS private key; or XR_TELEOP_KEY")
    args, remaining = parser.parse_known_args(argv)
    if args.video_only and args.input_mode is not None:
        parser.error("--video-only and --input-mode cannot be combined")
    if args.video_only or args.input_mode == "none":
        if remaining in (["--help"], ["-h"]):
            parser.print_help()
            parser.exit()
        if remaining:
            parser.error("video/neck-only mode does not accept teleop arguments: " + " ".join(remaining))
    elif args.input_mode is not None:
        remaining = ["--input-mode", args.input_mode, *remaining]
    return args, remaining


def send_fresh_head_pose(pose_shared, sender, max_age=0.25) -> bool:
    """Do not renew the PC2 watchdog using a cached pose after XR disconnects."""
    with pose_shared.get_lock():
        sample = list(pose_shared[:])
    if sample[16] <= 0 or time.monotonic() - sample[16] > max_age:
        return False
    matrix = [sample[offset:offset + 4] for offset in range(0, 16, 4)]
    sender.send_openxr_matrix(matrix)
    return True


def _stop_video_process(process) -> None:
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError("Video server process did not stop")
    process.close()


def run_video_only(args: argparse.Namespace) -> int:
    """Use the installed legacy TeleVuer API, with a process-local scene override.

    Vuer 0.0.40's WebRTCStereoVideoPlane crops the source with repeat=(0.5, 1)
    and offsets (0, 0)/(0.5, 0), on XR eye layers 1/2 respectively. Its aspect
    is therefore the per-eye aspect, not the combined SBS frame aspect.
    """
    import asyncio
    import inspect
    import multiprocessing
    from multiprocessing import shared_memory
    import signal
    import ssl
    from urllib.parse import urlsplit

    neck_only = args.input_mode == "none"
    label = "neck-only" if neck_only else "video-only"
    if neck_only:
        try:
            from .head_pose_udp import HeadPoseUdpSender, _validated_pose_matrix
        except ImportError:
            from head_pose_udp import HeadPoseUdpSender, _validated_pose_matrix

    endpoint = urlsplit(args.video_offer_url)
    if (endpoint.scheme != "https" or not endpoint.hostname
            or endpoint.username is not None or endpoint.password is not None
            or endpoint.fragment or not endpoint.path or endpoint.port == 0):
        raise ValueError("--video-offer-url must be an HTTPS endpoint without credentials or fragment")

    cert = args.video_cert or os.environ.get("XR_TELEOP_CERT")
    key = args.video_key or os.environ.get("XR_TELEOP_KEY")
    if not cert or not key:
        raise ValueError("Set --video-cert and --video-key (or XR_TELEOP_CERT and XR_TELEOP_KEY)")
    cert_path, key_path = Path(cert).expanduser().resolve(), Path(key).expanduser().resolve()
    for path in (cert_path, key_path):
        if not path.is_file():
            raise FileNotFoundError(f"TLS file not found: {path}")
    # Validate the certificate/key pair without printing either file's contents.
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(cert_path), str(key_path))

    # The installed TeleVuer starts a bound-method Process in its constructor.
    # Its process-local subclass requires the Host's Linux/Python 3.10 fork mode.
    if multiprocessing.get_start_method() != "fork":
        raise RuntimeError("This video-only adapter requires Linux multiprocessing start method 'fork'")

    repository = Path(args.xr_repo).expanduser().resolve()
    televuer_src = repository / "teleop" / "televuer" / "src"
    if not televuer_src.is_dir():
        raise FileNotFoundError(f"TeleVuer source directory not found: {televuer_src}")
    sys.path.insert(0, str(televuer_src))
    from televuer import TeleVuer
    from vuer.schemas import WebRTCStereoVideoPlane

    required = {"binocular", "use_hand_tracking", "img_shape", "img_shm_name",
                "cert_file", "key_file", "webrtc"}
    if (not required.issubset(inspect.signature(TeleVuer.__init__).parameters)
            or not hasattr(TeleVuer, "main_image_webrtc")):
        raise RuntimeError("Installed TeleVuer API differs from the inspected Host version")
    if neck_only and not hasattr(TeleVuer, "on_cam_move"):
        raise RuntimeError("Neck-only mode requires TeleVuer.on_cam_move")

    class VideoOnlyTeleVuer(TeleVuer):
        def __init__(self, **kwargs):
            # Allocate before TeleVuer forks its server. Both processes share
            # the validated pose and receipt time under one lock.
            if neck_only:
                self.neck_pose_shared = multiprocessing.Array("d", 17, lock=True)
            try:
                super().__init__(**kwargs)
            except BaseException:
                _stop_video_process(getattr(self, "process", None))
                raise

        async def on_cam_move(self, event, session, fps=60):
            await super().on_cam_move(event, session)
            if not neck_only:
                return
            try:
                flat = event.value["camera"]["matrix"]
                if len(flat) != 16:
                    return
                # Vuer CAMERA_MOVE matrices use column-major OpenXR order.
                matrix = _validated_pose_matrix([
                    [flat[column * 4 + row] for column in range(4)]
                    for row in range(4)
                ])
            except (KeyError, TypeError, ValueError, IndexError):
                return
            with self.neck_pose_shared.get_lock():
                self.neck_pose_shared[:] = [
                    value for row in matrix for value in row
                ] + [time.monotonic()]

        async def main_image_webrtc(self, session, fps=60):
            # Do not mount Hands or MotionControllers, or override eye layers.
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=args.video_offer_url,
                    iceServer={},
                    key="webrtc",
                    aspect=1280 / 720,
                    height=7,
                ),
                to="bgChildren",
            )
            while True:
                await asyncio.sleep(1)

    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    image_shape = (720, 2560, 3)
    shm = None
    viewer = None
    sender = None
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        # Legacy TeleVuer attaches this even in WebRTC mode. No video is relayed
        # through it: the Quest negotiates directly with PC2's /offer endpoint.
        shm = shared_memory.SharedMemory(create=True, size=720 * 2560 * 3)
        viewer = VideoOnlyTeleVuer(
            binocular=True, use_hand_tracking=False,
            img_shape=image_shape, img_shm_name=shm.name,
            cert_file=str(cert_path), key_file=str(key_path), webrtc=True,
        )
        print(f"[{label}] TeleVuer process started: PID {viewer.process.pid}", flush=True)
        print(f"[{label}] Quest video source: {args.video_offer_url}", flush=True)
        if neck_only:
            sender = HeadPoseUdpSender(args.neck_pose_ip, args.neck_pose_port)
            print(f"[{label}] Head pose UDP: {args.neck_pose_ip}:{args.neck_pose_port} "
                  f"at up to {args.neck_pose_rate:g} Hz. Enter VR before starting the PC2 motor receiver. "
                  "No arm/hand control or recording. Ctrl+C to stop.", flush=True)
        else:
            print("[video-only] No robot control, pose UDP, or recording. Ctrl+C to stop.", flush=True)
        period = 1.0 / args.neck_pose_rate if neck_only else 0.5
        last_send_error = None
        while viewer.process.is_alive():
            loop_start = time.monotonic()
            if sender is not None:
                try:
                    if send_fresh_head_pose(viewer.neck_pose_shared, sender):
                        last_send_error = None
                except (OSError, ValueError) as error:
                    if str(error) != last_send_error:
                        print(f"[{label}] Head pose send failed: {error}", file=sys.stderr, flush=True)
                    last_send_error = str(error)
            viewer.process.join(timeout=max(0.0, period - (time.monotonic() - loop_start)))
        raise RuntimeError(f"TeleVuer server exited unexpectedly (code {viewer.process.exitcode})")
    except KeyboardInterrupt:
        print(f"\n[{label}] Stopping.", flush=True)
        return 0
    finally:
        try:
            try:
                if sender is not None:
                    sender.close()
            finally:
                if viewer is not None:
                    _stop_video_process(viewer.process)
        finally:
            try:
                if shm is not None:
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: Optional[Sequence[str]] = None) -> int:
    launcher_args, teleop_args = parse_launcher_args(argv)
    if launcher_args.video_only:
        return run_video_only(launcher_args)

    if not 1 <= launcher_args.neck_pose_port <= 65535:
        raise ValueError("--neck-pose-port must be in 1..65535")
    if launcher_args.input_mode == "none":
        return run_video_only(launcher_args)

    repository = Path(launcher_args.xr_repo).expanduser().resolve()
    teleop_script = repository / "teleop" / "teleop_hand_and_arm.py"
    if not teleop_script.is_file():
        raise FileNotFoundError(f"teleop script not found: {teleop_script}")

    sys.path.insert(0, str(repository))
    import televuer  # Imported after the selected repository is on sys.path.
    from head_pose_udp import HeadPoseUdpSender

    original_wrapper = televuer.TeleVuerWrapper
    destination_ip = launcher_args.neck_pose_ip
    destination_port = launcher_args.neck_pose_port
    pose_period = 1.0 / launcher_args.neck_pose_rate

    class NeckForwardingTeleVuerWrapper(original_wrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._neck_sender = HeadPoseUdpSender(destination_ip, destination_port)
            self._neck_stop = threading.Event()
            self._neck_thread = threading.Thread(
                target=self._neck_pose_loop,
                name="quest-head-pose-udp",
                daemon=True,
            )
            self._neck_thread.start()

        def _neck_pose_loop(self) -> None:
            deadline = time.perf_counter()
            while not self._neck_stop.is_set():
                try:
                    # TeleVuer exposes the raw OpenXR matrix here. Reading it
                    # directly avoids waiting for the arm IK/control loop.
                    self._neck_sender.send_openxr_matrix(self.tvuer.head_pose)
                except (OSError, ValueError):
                    # The shared matrix is all-zero until the first CAMERA_MOVE.
                    pass

                deadline += pose_period
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    deadline = time.perf_counter()
                    continue
                self._neck_stop.wait(remaining)

        def close(self) -> None:
            if hasattr(self, "_neck_stop"):
                self._neck_stop.set()
            if hasattr(self, "_neck_thread"):
                self._neck_thread.join(timeout=0.5)
            if hasattr(self, "_neck_sender"):
                self._neck_sender.close()
            super().close()

    televuer.TeleVuerWrapper = NeckForwardingTeleVuerWrapper
    sys.argv = [str(teleop_script), *teleop_args]
    os.chdir(teleop_script.parent)
    runpy.run_path(str(teleop_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

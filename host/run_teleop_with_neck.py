#!/usr/bin/env python3
"""Run full teleop with neck forwarding, or a video-only Quest viewer.

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
        "--video-only", action="store_true",
        help="Display PC2 stereo video without robot control, pose UDP, or recording",
    )
    parser.add_argument(
        "--video-offer-url", default="https://192.168.123.164:60001/offer",
        help="PC2 HTTPS WebRTC offer endpoint (video-only)",
    )
    parser.add_argument("--video-cert", help="Host TLS certificate; or XR_TELEOP_CERT")
    parser.add_argument("--video-key", help="Host TLS private key; or XR_TELEOP_KEY")
    args, remaining = parser.parse_known_args(argv)
    if args.video_only:
        if remaining in (["--help"], ["-h"]):
            parser.print_help()
            parser.exit()
        if remaining:
            parser.error("--video-only does not accept teleop arguments: " + " ".join(remaining))
    return args, remaining


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

    class VideoOnlyTeleVuer(TeleVuer):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except BaseException:
                _stop_video_process(getattr(self, "process", None))
                raise

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
        print(f"[video-only] TeleVuer process started: PID {viewer.process.pid}", flush=True)
        print(f"[video-only] Quest video source: {args.video_offer_url}", flush=True)
        print("[video-only] No robot control, pose UDP, or recording. Ctrl+C to stop.", flush=True)
        while viewer.process.is_alive():
            viewer.process.join(timeout=0.5)
        raise RuntimeError(f"TeleVuer server exited unexpectedly (code {viewer.process.exitcode})")
    except KeyboardInterrupt:
        print("\n[video-only] Stopping.", flush=True)
        return 0
    finally:
        try:
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

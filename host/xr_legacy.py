"""Adapters for the inspected shared-memory xr_teleoperate fork.

All changes live in the Host recorder; the robot checkout is left untouched.
"""
from __future__ import annotations

import asyncio
import inspect
import importlib.metadata
import multiprocessing as mp
import signal
import sys
import time

try:
    from .head_pose_udp import _validated_pose_matrix
    from .recorder_diagnostics import PoseEventDiagnostics
except ImportError:
    from head_pose_udp import _validated_pose_matrix
    from recorder_diagnostics import PoseEventDiagnostics


def installed_versions() -> dict:
    result = {}
    for name in ("televuer", "teleimager", "vuer", "params-proto", "waterbear",
                 "logging_mp", "unitree_sdk2py", "rerun-sdk", "numpy", "opencv-python"):
        try:
            distribution = importlib.metadata.distribution(name)
            result[name] = {"version": distribution.version,
                            "direct_url": distribution.read_text("direct_url.json")}
        except importlib.metadata.PackageNotFoundError:
            result[name] = {"version": None}
    return result


def check_legacy_api(wrapper, hand_controller, writer, image_client=None) -> dict:
    required = {
        wrapper: {"binocular", "use_hand_tracking", "img_shape", "img_shm_name",
                  "return_state_data", "return_hand_rot_data"},
        hand_controller: {"left_hand_array_in", "right_hand_array_in",
                          "dual_hand_data_lock", "dual_hand_state_array_out",
                          "dual_hand_action_array_out"},
        writer: {"task_dir", "frequency", "image_size", "rerun_log"},
    }
    if image_client is not None:
        required[image_client] = {"host", "request_bgr"}
        for name in ("get_cam_config", "get_head_frame", "close"):
            if not callable(getattr(image_client, name, None)):
                raise RuntimeError(f"ImageClient.{name} is required")
    result = {}
    for cls, names in required.items():
        signature = inspect.signature(cls)
        if not names.issubset(signature.parameters):
            raise RuntimeError(f"Unsupported {cls.__name__} API: {signature}")
        result[cls.__name__] = {
            "source": inspect.getfile(cls), "signature": str(signature),
        }
    if not hasattr(wrapper, "get_motion_state_data"):
        raise RuntimeError("TeleVuerWrapper.get_motion_state_data is required")
    return result


def stop_process(process) -> None:
    if process is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


class LegacyStereoXR:
    """Display teleimager BGR frames through the old wrapper's shared memory."""

    def __init__(self, wrapper, shape, cert, key, screen_y=-0.1, fps=30.0):
        import numpy as np
        from multiprocessing import shared_memory
        from vuer.schemas import Hands, ImageBackground

        if mp.get_start_method() != "fork":
            raise RuntimeError("This xr_teleoperate fork requires multiprocessing fork")
        module = sys.modules[wrapper.__module__]
        original = module.TeleVuer

        class RecorderTeleVuer(original):
            def __init__(self, *args, **kwargs):
                # Allocate before the original constructor forks the Vuer server.
                self.head_received_ns = mp.Value("Q", 0)
                self.hand_received_ns = mp.Value("Q", 0)
                self.pose_event_diagnostics = PoseEventDiagnostics()
                # The legacy TeleVuer stores only an ndarray backed by a
                # constructor-local SharedMemory attachment. Its destructor
                # unmaps that buffer; rendering it can segfault. Retain the
                # owner's live mapping in both the parent and Vuer child.
                self._recorder_shm = display_shm
                super().__init__(*args, **kwargs)
                self._use_retained_image()

            def _use_retained_image(self):
                self.img_array = np.ndarray(tuple(shape), dtype=np.uint8,
                                            buffer=self._recorder_shm.buf)

            def vuer_run(self):
                # vuer 0.0.60 uses get_event_loop(). A parent that has already
                # called asyncio.run() leaves no current loop for the fork.
                self._use_retained_image()
                # Handle terminal Ctrl+C in the parent, which flushes the
                # episode before terminating this display-only child.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    super().vuer_run()
                finally:
                    loop.close()

            async def on_cam_move(self, event, session, fps=60):
                try:
                    _validated_pose_matrix(np.asarray(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
                except (KeyError, TypeError, ValueError) as error:
                    self.pose_event_diagnostics.update("head", f"camera.matrix: {type(error).__name__}: {error}")
                    return
                await super().on_cam_move(event, session, fps)
                self.head_received_ns.value = time.monotonic_ns()
                self.pose_event_diagnostics.update("head")

            async def on_hand_move(self, event, session, fps=60):
                stage = "hand payload"
                try:
                    for side in ("left", "right"):
                        stage = f"{side} hand payload"
                        values = np.asarray(event.value[side], dtype=float)
                        if values.shape != (400,) or not np.isfinite(values).all():
                            reason = (f"expected shape=(400,), got shape={values.shape}"
                                      if values.shape != (400,) else "contains NaN or infinity")
                            self.pose_event_diagnostics.update("hand", f"{stage}: {reason}; remaining checks not run")
                            return
                        stage = f"{side} wrist pose"
                        _validated_pose_matrix(values[:16].reshape(4, 4, order="F"))
                        stage = f"{side}State"
                        if not isinstance(event.value[side + "State"], dict):
                            self.pose_event_diagnostics.update("hand", f"{stage}: expected dict; remaining checks not run")
                            return
                except (KeyError, TypeError, ValueError) as error:
                    self.pose_event_diagnostics.update("hand", f"{stage}: {type(error).__name__}: {error}; remaining checks not run")
                    return
                await super().on_hand_move(event, session, fps)
                self.hand_received_ns.value = time.monotonic_ns()
                self.pose_event_diagnostics.update("hand")

            async def main_image_binocular(self, session, fps=60):
                import cv2
                session.upsert(Hands(stream=True, key="hands", hideLeft=True,
                                     hideRight=True), to="bgChildren")
                while True:
                    rgb = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
                    eyes = (rgb[:, :self.img_width], rgb[:, self.img_width:])
                    try:
                        session.upsert([
                            ImageBackground(
                                eye, aspect=self.img_width / self.img_height,
                                height=1, position=[0, screen_y, 0],
                                distanceToCamera=1, layers=index + 1,
                                format="jpeg", quality=90,
                                key=f"background-{'left' if index == 0 else 'right'}",
                                interpolate=True,
                            ) for index, eye in enumerate(eyes)
                        ], to="bgChildren")
                    except AssertionError:
                        return
                    await asyncio.sleep(1.0 / fps)

        self.wrapper = None
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self.image = np.ndarray(shape, dtype=np.uint8, buffer=self.shm.buf)
        self.image.fill(0)
        display_shm = self.shm
        module.TeleVuer = RecorderTeleVuer
        try:
            self.wrapper = wrapper(
                binocular=True, use_hand_tracking=True, img_shape=tuple(shape),
                img_shm_name=self.shm.name, return_state_data=True,
                return_hand_rot_data=False, cert_file=str(cert), key_file=str(key),
                webrtc=False,
            )
        except BaseException:
            self.close()
            raise
        finally:
            module.TeleVuer = original

    @property
    def tvuer(self):
        return self.wrapper.tvuer

    def render(self, bgr) -> None:
        import numpy as np
        if bgr.shape != self.image.shape:
            raise ValueError(f"ZED frame {bgr.shape} != configured {self.image.shape}")
        np.copyto(self.image, bgr)

    def tracking_ready(self, max_age_s=0.5) -> bool:
        now = time.monotonic_ns()
        return all(0 < stamp.value <= now
                   and now - stamp.value < max_age_s * 1e9
                   for stamp in (self.tvuer.head_received_ns,
                                 self.tvuer.hand_received_ns))

    def get_motion_state_data(self):
        if not self.tvuer.process.is_alive():
            raise RuntimeError(f"Vuer exited: {self.tvuer.process.exitcode}")
        return self.wrapper.get_motion_state_data()

    def close(self) -> None:
        if self.wrapper is not None:
            stop_process(self.wrapper.tvuer.process)
            self.wrapper = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


class LegacyHandSession:
    """Retain and stop the legacy controller's otherwise unexposed child."""

    def __init__(self, controller, *args):
        stop = mp.Event()

        class ManagedController(controller):
            @property
            def running(self):
                return not stop.is_set()

            @running.setter
            def running(self, value):
                if not value:
                    stop.set()

        self.stop = stop
        self.controller = None
        before = {p.pid for p in mp.active_children()}
        try:
            self.controller = ManagedController(*args)
        finally:
            self.processes = [p for p in mp.active_children() if p.pid not in before]
            if self.controller is None:
                self.close()

    def close(self):
        self.stop.set()
        for process in self.processes:
            process.join(timeout=1)
            stop_process(process)
        if self.controller is not None:
            for name in ("LeftHandState_subscriber", "RightHandState_subscriber"):
                getattr(self.controller, name).Close()

#!/usr/bin/env python3
"""Run a trained ACT policy (training/train_act.py) on G1_29 + Dex3-1 + ZED neck.

The policy replaces the operator of teleop_record_dex3_tactile.py:

- observation: ZED left/right from teleimager ZMQ, arm q from rt/lowstate,
  Dex3-1 q from rt/dex3/*/state, neck encoder yaw/pitch from PC2 (UDP 5006);
- action: arm q -> G1_29_ArmController (with gravity feed-forward from
  G1_29_ArmIK's model), Dex3-1 q -> rt/dex3/*/cmd directly (no retargeting),
  neck yaw/pitch -> head-pose packets to PC2 (UDP 5005).

The neck command is encoded as an OpenXR orientation. The first packets carry
the recenter flag, so PC2's head_pose_receiver takes the identity as neutral
and yaw/pitch map to motor center exactly as in the recorded episodes. No
PC2 change is needed; run zed_teleimager_server.py and
head_pose_receiver.py --enable-motor as for recording.

Every command is rate-limited per control step relative to the previous
command, which also ramps the robot smoothly into the first policy output.

Keys (sshkeyboard):
  r: start / resume the policy   p: pause (hold the last command)   q: quit
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import time
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

try:
    from .head_pose_udp import FLAG_ORIENTATION_VALID, FLAG_TRACKED, MAGIC, PACKET_STRUCT, VERSION
    from .neck_state_receiver import NeckStateReceiver
except ImportError:
    from head_pose_udp import (  # type: ignore[no-redef]
        FLAG_ORIENTATION_VALID, FLAG_TRACKED, MAGIC, PACKET_STRUCT, VERSION)
    from neck_state_receiver import NeckStateReceiver  # type: ignore[no-redef]


DEX3_NUM_MOTORS = 7
G1_29_NUM_ARM_JOINTS = 14
G1_29_ARM_MOTOR_OFFSET = 15  # left arm 15..21, right arm 22..28 in rt/lowstate
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_DEX3_LEFT_STATE = "rt/dex3/left/state"
TOPIC_DEX3_RIGHT_STATE = "rt/dex3/right/state"
TOPIC_DEX3_LEFT_CMD = "rt/dex3/left/cmd"
TOPIC_DEX3_RIGHT_CMD = "rt/dex3/right/cmd"
DEX3_KP = 1.5  # same gains as Dex3_1_Controller
DEX3_KD = 0.2
FLAG_RECENTER = 1 << 3
DEFAULT_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "training" / "train_act.py"


def positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained ACT policy on G1_29 + Dex3-1.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="policy_best.ckpt / policy_last.ckpt from train_act.py")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT,
                        help="train_act.py that defines the ACT model")
    parser.add_argument("--xr-repo", default="~/hckang/xr_teleoperate",
                        help="xr_teleoperate repository root")
    parser.add_argument("--img-server-ip", default="192.168.123.164")
    parser.add_argument("--network-interface", default=None,
                        help="DDS network interface, e.g. eth0")
    parser.add_argument("--motion", action="store_true",
                        help="robot is in motion (locomotion) mode; skip entering debug mode")
    parser.add_argument("--neck-pose-ip", default="192.168.123.164")
    parser.add_argument("--neck-pose-port", type=int, default=5005)
    parser.add_argument("--neck-pose-rate", type=positive_float, default=60.0)
    parser.add_argument("--neck-state-bind", default="0.0.0.0")
    parser.add_argument("--neck-state-port", type=int, default=5006)
    parser.add_argument("--frequency", type=positive_float, default=30.0,
                        help="control frequency; must match the recording frequency")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--no-temporal-agg", action="store_true",
                        help="execute chunks open-loop instead of temporal ensembling")
    parser.add_argument("--ensemble-k", type=float, default=0.01,
                        help="temporal ensembling weight exp(-k*i), oldest chunk first")
    parser.add_argument("--query-every", type=int, default=None,
                        help="with --no-temporal-agg: steps per policy query (default chunk size)")
    parser.add_argument("--max-arm-step", type=positive_float, default=0.05,
                        help="max arm joint change per step [rad]")
    parser.add_argument("--max-hand-step", type=positive_float, default=0.1,
                        help="max Dex3-1 joint change per step [rad]")
    parser.add_argument("--max-neck-step-deg", type=positive_float, default=3.0,
                        help="max neck yaw/pitch change per step [deg]")
    parser.add_argument("--no-tauff", action="store_true",
                        help="send zero arm feed-forward torque instead of gravity compensation")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="read sensors and run the policy, but send no robot/neck commands")
    args = parser.parse_args(argv)
    for name in ("neck_pose_port", "neck_state_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be in 1..65535")
    return args


# ---------------------------------------------------------------------------
# Pure helpers (unit tested without hardware)
# ---------------------------------------------------------------------------

def yaw_pitch_to_quaternion(yaw_rad: float, pitch_rad: float) -> tuple[float, float, float, float]:
    """OpenXR x/y/z/w orientation whose forward vector has this yaw/pitch.

    Inverse of pc2.active_camera_protocol.quaternion_to_yaw_pitch for a neutral
    of identity: yaw is right positive (rotation about -Y), pitch is up
    positive (rotation about +X), q = q_y(-yaw) * q_x(pitch).
    """

    sy, cy = math.sin(-yaw_rad / 2.0), math.cos(-yaw_rad / 2.0)
    sp, cp = math.sin(pitch_rad / 2.0), math.cos(pitch_rad / 2.0)
    return (cy * sp, sy * cp, -sy * sp, cy * cp)


def build_neck_packet(yaw_rad: float, pitch_rad: float, sequence: int,
                      recenter: bool = False, sender_time_ns: Optional[int] = None) -> bytes:
    flags = FLAG_ORIENTATION_VALID | FLAG_TRACKED | (FLAG_RECENTER if recenter else 0)
    timestamp = time.time_ns() if sender_time_ns is None else sender_time_ns
    if recenter:
        orientation = (0.0, 0.0, 0.0, 1.0)
    else:
        orientation = yaw_pitch_to_quaternion(yaw_rad, pitch_rad)
    return PACKET_STRUCT.pack(MAGIC, VERSION, flags, 0, sequence & 0xFFFFFFFF, timestamp,
                              0.0, 0.0, 0.0, *orientation)


def clip_step(target: np.ndarray, previous: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    """Move from previous toward target by at most max_step per element."""

    return previous + np.clip(target - previous, -max_step, max_step)


class TemporalEnsembler:
    """ACT temporal ensembling: weighted mean of every chunk covering step t."""

    def __init__(self, k: float) -> None:
        self.k = k
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def add(self, t: int, chunk: np.ndarray) -> None:
        self._chunks.append((t, chunk))

    def get(self, t: int) -> np.ndarray:
        while self._chunks and t - self._chunks[0][0] >= len(self._chunks[0][1]):
            self._chunks.popleft()
        rows = [chunk[t - t0] for t0, chunk in self._chunks if 0 <= t - t0 < len(chunk)]
        if not rows:
            raise RuntimeError(f"no chunk covers step {t}")
        weights = np.exp(-self.k * np.arange(len(rows)))  # oldest first, as in ACT
        return (np.stack(rows) * (weights / weights.sum())[:, None]).sum(axis=0)


def split_stereo_rgb(head_bgr: np.ndarray) -> list[np.ndarray]:
    """ZED left|right BGR frame -> [left RGB, right RGB] (color_0, color_1)."""

    half = head_bgr.shape[1] // 2
    return [head_bgr[:, :half, ::-1], head_bgr[:, half:, ::-1]]


def group_slices(joint_groups: Sequence[Sequence[Any]]) -> dict[str, slice]:
    slices, start = {}, 0
    for key, dof in joint_groups:
        slices[key] = slice(start, start + int(dof))
        start += int(dof)
    return slices


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def load_train_module(train_script: Path) -> Any:
    """Import train_act.py from a file path (host/ may be copied without training/)."""

    if not train_script.is_file():
        raise FileNotFoundError(f"{train_script} not found; pass --train-script")
    spec = importlib.util.spec_from_file_location("train_act", train_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {train_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_act"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


class ACTRunner:
    """Load a train_act.py checkpoint and predict denormalized action chunks."""

    def __init__(self, checkpoint: Path, train_script: Path, device: str) -> None:
        import torch
        from PIL import Image

        module = load_train_module(train_script)

        self._torch, self._image = torch, Image
        self.device = module.pick_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = dict(ckpt["config"], pretrained_backbone=False)  # weights come from ckpt
        self.model = module.ACT(module.ACTConfig(**config))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        stats = ckpt["norm_stats"]
        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.joint_groups = [tuple(g) for g in ckpt["joint_groups"]]
        self.camera_keys = list(ckpt["camera_keys"])
        self.image_size = tuple(ckpt["image_size"])  # (H, W)
        self.chunk_size = config["chunk_size"]

    def _image_tensor(self, rgb: np.ndarray):
        # Same resize as ACTDataset._load_image.
        h, w = self.image_size
        img = self._image.fromarray(np.ascontiguousarray(rgb)).resize((w, h), self._image.BILINEAR)
        return self._torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def predict(self, images_rgb: Sequence[np.ndarray], qpos: np.ndarray) -> np.ndarray:
        torch = self._torch
        images = torch.stack([self._image_tensor(img) for img in images_rgb]).unsqueeze(0)
        qpos_n = torch.from_numpy(((qpos - self.qpos_mean) / self.qpos_std).astype(np.float32))
        with torch.inference_mode():
            pred, _, _ = self.model(images.to(self.device), qpos_n.unsqueeze(0).to(self.device))
        return pred[0].float().cpu().numpy() * self.action_std + self.action_mean


# ---------------------------------------------------------------------------
# Robot I/O
# ---------------------------------------------------------------------------

class LatestJointReader:
    """Callback DDS subscriber that keeps motor_state[i].q for the given indices."""

    def __init__(self, subscriber_factory: Any, topic: str, message_type: Any,
                 indices: Sequence[int]) -> None:
        self._indices = list(indices)
        self._lock = threading.Lock()
        self._q: Optional[np.ndarray] = None
        self._received_ns: Optional[int] = None
        self._subscriber = subscriber_factory(topic, message_type)
        self._subscriber.Init(self._handler, 10)

    def _handler(self, message: Any) -> None:
        try:
            q = np.array([message.motor_state[i].q for i in self._indices], dtype=np.float64)
        except (AttributeError, IndexError, TypeError):
            return  # an exception would end the SDK reader thread
        with self._lock:
            self._q, self._received_ns = q, time.monotonic_ns()

    def latest(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._q is None or self._received_ns is None:
                return None, None
            return self._q.copy(), (time.monotonic_ns() - self._received_ns) / 1e6

    def close(self) -> None:
        self._subscriber.Close()


class Dex3CommandPublisher:
    """Publish Dex3-1 joint targets directly (same mode/gains as Dex3_1_Controller)."""

    def __init__(self, publisher_factory: Any, message_type: Any, message_factory: Any) -> None:
        self._publishers, self._messages = [], []
        for topic in (TOPIC_DEX3_LEFT_CMD, TOPIC_DEX3_RIGHT_CMD):
            publisher = publisher_factory(topic, message_type)
            publisher.Init()
            message = message_factory()
            for motor_id in range(DEX3_NUM_MOTORS):
                # RIS mode byte: id (4 bit) | status 0x01 << 4 | timeout 0 << 7
                message.motor_cmd[motor_id].mode = (motor_id & 0x0F) | (0x01 << 4)
                message.motor_cmd[motor_id].dq = 0.0
                message.motor_cmd[motor_id].tau = 0.0
                message.motor_cmd[motor_id].kp = DEX3_KP
                message.motor_cmd[motor_id].kd = DEX3_KD
            self._publishers.append(publisher)
            self._messages.append(message)

    def send(self, left_q: Sequence[float], right_q: Sequence[float]) -> None:
        for publisher, message, q in zip(self._publishers, self._messages, (left_q, right_q)):
            for motor_id in range(DEX3_NUM_MOTORS):
                message.motor_cmd[motor_id].q = float(q[motor_id])
            publisher.Write(message)


class NeckCommandSender:
    """Stream the latest neck yaw/pitch target to PC2 as head-pose packets."""

    def __init__(self, ip: str, port: int, rate_hz: float, recenter_packets: int = 10) -> None:
        import socket

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._destination = (ip, port)
        self._period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._target = (0.0, 0.0)
        self._sequence = 0
        self._recenter_left = recenter_packets
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="neck-command-udp", daemon=True)
        self._thread.start()

    def set_target(self, yaw_rad: float, pitch_rad: float) -> None:
        with self._lock:
            self._target = (float(yaw_rad), float(pitch_rad))

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                yaw, pitch = self._target
            recenter = self._recenter_left > 0
            self._recenter_left -= int(recenter)
            try:
                self._socket.sendto(build_neck_packet(yaw, pitch, self._sequence, recenter),
                                    self._destination)
            except OSError:
                pass
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
            self._stop.wait(self._period)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close()


class KeyState:
    def __init__(self) -> None:
        self.running = False
        self.stop = False
        self.resumed = False

    def on_press(self, key: str) -> None:
        if key == "r" and not self.running:
            self.running = True
            self.resumed = True
        elif key == "p":
            self.running = False
        elif key == "q":
            self.running = False
            self.stop = True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    runner = ACTRunner(args.checkpoint.expanduser(), args.train_script.expanduser(), args.device)
    slices = group_slices(runner.joint_groups)
    expected = {"left_arm", "right_arm", "left_ee", "right_ee"}
    if not expected <= slices.keys():
        raise ValueError(f"checkpoint joint groups {list(slices)} lack {expected - slices.keys()}")
    use_neck = "neck" in slices
    if runner.camera_keys != ["color_0", "color_1"]:
        raise ValueError(f"expected ZED color_0/color_1 cameras, got {runner.camera_keys}")

    repository = Path(args.xr_repo).expanduser().resolve()
    teleop_dir = repository / "teleop"
    if not (teleop_dir / "teleop_hand_and_arm.py").is_file():
        raise FileNotFoundError(f"xr_teleoperate teleop directory not found: {teleop_dir}")
    sys.path.insert(0, str(repository))
    os.chdir(teleop_dir)

    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    logger = logging_mp.getLogger(__name__)
    logger.info(f"policy: chunk {runner.chunk_size}, groups {runner.joint_groups}, "
                f"device {runner.device}, neck {'on' if use_neck else 'off'}")

    from sshkeyboard import listen_keyboard, stop_listening
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowState_
    from teleimager.image_client import ImageClient

    keys = KeyState()
    img_client = arm_reader = hand_readers = neck_state = neck_cmd = arm_ctrl = None
    keyboard_thread = None
    try:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)
        keyboard_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": keys.on_press, "until": None, "sequential": False},
            daemon=True,
        )
        keyboard_thread.start()

        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        if not img_client.get_cam_config()["head_camera"]["binocular"]:
            raise RuntimeError("expected the ZED binocular head camera (cam_config_zed.yaml)")
        arm_reader = LatestJointReader(ChannelSubscriber, TOPIC_LOWSTATE, LowState_,
                                       range(G1_29_ARM_MOTOR_OFFSET,
                                             G1_29_ARM_MOTOR_OFFSET + G1_29_NUM_ARM_JOINTS))
        hand_readers = [LatestJointReader(ChannelSubscriber, topic, HandState_,
                                          range(DEX3_NUM_MOTORS))
                        for topic in (TOPIC_DEX3_LEFT_STATE, TOPIC_DEX3_RIGHT_STATE)]
        if use_neck:
            neck_state = NeckStateReceiver(args.neck_state_bind, args.neck_state_port)

        arm_ik = hand_cmd = None
        if not args.dry_run:
            from teleop.robot_control.robot_arm import G1_29_ArmController
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
            from teleop.utils.motion_switcher import MotionSwitcher

            if not args.motion:
                status, _ = MotionSwitcher().Enter_Debug_Mode()
                logger.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
            if not args.no_tauff:
                arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=False)
            hand_cmd = Dex3CommandPublisher(ChannelPublisher, HandCmd_,
                                            unitree_hg_msg_dds__HandCmd_)
        else:
            logger.info("dry run: no commands will be sent")

        def read_state() -> Optional[np.ndarray]:
            arm_q, _ = arm_reader.latest()
            left_q, _ = hand_readers[0].latest()
            right_q, _ = hand_readers[1].latest()
            if arm_q is None or left_q is None or right_q is None:
                return None
            state = np.zeros(sum(int(d) for _, d in runner.joint_groups), dtype=np.float64)
            state[slices["left_arm"]] = arm_q[:7]
            state[slices["right_arm"]] = arm_q[7:]
            state[slices["left_ee"]] = left_q
            state[slices["right_ee"]] = right_q
            if use_neck:
                packet, age_ms = neck_state.latest()
                if packet is not None and packet.present_deg:
                    if age_ms is not None and age_ms > 500:
                        logger.warning(f"neck state is {age_ms:.0f} ms old")
                    state[slices["neck"]] = np.radians(packet.present_deg)
                elif not args.dry_run:
                    return None
                # dry run sends no head pose, so PC2 sends no neck state: use center
            return state

        max_step = np.zeros_like(runner.action_mean, dtype=np.float64)
        for key in ("left_arm", "right_arm"):
            max_step[slices[key]] = args.max_arm_step
        for key in ("left_ee", "right_ee"):
            max_step[slices[key]] = args.max_hand_step
        if use_neck:
            max_step[slices["neck"]] = math.radians(args.max_neck_step_deg)

        def send(command: np.ndarray) -> None:
            if args.dry_run:
                return
            arm_q = np.concatenate([command[slices["left_arm"]], command[slices["right_arm"]]])
            if arm_ik is not None:
                import pinocchio as pin
                model, data = arm_ik.reduced_robot.model, arm_ik.reduced_robot.data
                tauff = pin.rnea(model, data, arm_q, np.zeros(model.nv), np.zeros(model.nv))
            else:
                tauff = np.zeros(G1_29_NUM_ARM_JOINTS)
            arm_ctrl.ctrl_dual_arm(arm_q, tauff)
            hand_cmd.send(command[slices["left_ee"]], command[slices["right_ee"]])
            if neck_cmd is not None:
                neck_cmd.set_target(*command[slices["neck"]])

        logger.info("Press [r] to start the policy, [p] to pause, [q] to quit.")
        ensembler = TemporalEnsembler(args.ensemble_k)
        query_every = args.query_every or runner.chunk_size
        period = 1.0 / args.frequency
        command: Optional[np.ndarray] = None
        chunk: Optional[np.ndarray] = None
        t = 0
        while not keys.stop:
            loop_start = time.time()
            if not keys.running:
                if command is not None:
                    send(command)  # keep streaming the held pose while paused
                time.sleep(period)
                continue

            if keys.resumed:
                keys.resumed = False
                if use_neck and neck_cmd is None and not args.dry_run:
                    # Recenter PC2 on identity; the neck moves to center first.
                    neck_cmd = NeckCommandSender(args.neck_pose_ip, args.neck_pose_port,
                                                 args.neck_pose_rate)
                deadline = time.time() + 3.0
                state = read_state()
                while state is None and time.time() < deadline and not keys.stop:
                    time.sleep(0.05)
                    state = read_state()
                if state is None:
                    logger.error("no arm/hand/neck state; is PC2 head_pose_receiver running "
                                 "with --enable-motor? Paused.")
                    keys.running = False
                    continue
                command = state.copy()  # rate limiting starts from the measured pose
                ensembler.reset()
                t = 0
                logger.info("Policy running.")

            state = read_state()
            head_bgr = img_client.get_head_frame().bgr
            if state is None or head_bgr is None:
                logger.warning("missing state or head image; holding")
                send(command)
                time.sleep(period)
                continue

            if args.no_temporal_agg:
                if t % query_every == 0:
                    chunk = runner.predict(split_stereo_rgb(head_bgr), state)
                target = chunk[t % query_every]
            else:
                ensembler.add(t, runner.predict(split_stereo_rgb(head_bgr), state))
                target = ensembler.get(t)

            command = clip_step(target, command, max_step)
            send(command)
            if args.dry_run and t % 10 == 0:
                delta = np.abs(target - state)
                arm = max(delta[slices["left_arm"]].max(), delta[slices["right_arm"]].max())
                hand = max(delta[slices["left_ee"]].max(), delta[slices["right_ee"]].max())
                logger.info(f"t={t} |target-state| max arm {arm:.3f} hand {hand:.3f} rad")
            t += 1
            if args.max_steps is not None and t >= args.max_steps:
                logger.info("max steps reached")
                break
            time.sleep(max(0.0, period - (time.time() - loop_start)))

    except KeyboardInterrupt:
        pass
    finally:
        def attempt(label: str, action) -> None:
            try:
                action()
            except Exception as error:  # keep shutting down the rest
                print(f"Failed to {label}: {error}", file=sys.stderr)

        if neck_cmd is not None:
            def center_neck() -> None:
                # Ramp back to center; PC2 applies no rate limit of its own.
                yaw, pitch = command[slices["neck"]] if command is not None else (0.0, 0.0)
                for alpha in np.linspace(1.0, 0.0, 31):
                    neck_cmd.set_target(alpha * yaw, alpha * pitch)
                    time.sleep(1.0 / 30.0)
            attempt("center neck", center_neck)
            attempt("close neck sender", neck_cmd.close)
        if arm_ctrl is not None:
            attempt("move arms home", arm_ctrl.ctrl_dual_arm_go_home)
        if keyboard_thread is not None:
            attempt("stop keyboard listener", stop_listening)
        for reader in [arm_reader] + list(hand_readers or []):
            if reader is not None:
                attempt("close state reader", reader.close)
        if neck_state is not None:
            attempt("close neck state receiver", neck_state.close)
        if img_client is not None:
            attempt("close image client", img_client.close)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

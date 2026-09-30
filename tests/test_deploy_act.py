import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from host.deploy_act import (
    DEFAULT_TRAIN_SCRIPT,
    TemporalEnsembler,
    build_neck_packet,
    clip_step,
    group_slices,
    split_stereo_rgb,
    yaw_pitch_to_quaternion,
)
from pc2.active_camera_protocol import TwoDofPoseMapper, decode_packet, quaternion_to_yaw_pitch

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class NeckPacketTests(unittest.TestCase):
    def test_quaternion_round_trips_through_pc2_decoder(self):
        for yaw_deg in (-70, -10, 0, 25, 80):
            for pitch_deg in (-35, -5, 0, 20, 45):
                q = yaw_pitch_to_quaternion(math.radians(yaw_deg), math.radians(pitch_deg))
                yaw, pitch = quaternion_to_yaw_pitch(q)
                self.assertAlmostEqual(math.degrees(yaw), yaw_deg, places=4)
                self.assertAlmostEqual(math.degrees(pitch), pitch_deg, places=4)

    def test_recenter_then_target_gives_same_yaw_pitch_on_pc2(self):
        mapper = TwoDofPoseMapper()
        first = mapper.update(decode_packet(build_neck_packet(0.3, 0.2, 0, recenter=True)))
        self.assertAlmostEqual(first.yaw_deg, 0.0)
        self.assertAlmostEqual(first.pitch_deg, 0.0)
        target = mapper.update(decode_packet(build_neck_packet(math.radians(30),
                                                               math.radians(-15), 1)))
        self.assertAlmostEqual(target.yaw_deg, 30.0, places=3)
        self.assertAlmostEqual(target.pitch_deg, -15.0, places=3)


class HelperTests(unittest.TestCase):
    def test_clip_step_limits_each_element(self):
        out = clip_step(np.array([1.0, -1.0, 0.01]), np.zeros(3), np.array([0.1, 0.2, 0.1]))
        np.testing.assert_allclose(out, [0.1, -0.2, 0.01])

    def test_temporal_ensembler_weights_oldest_first(self):
        ens = TemporalEnsembler(k=math.log(2.0))  # weights 1, 1/2
        ens.add(0, np.array([[0.0], [10.0], [20.0]]))
        np.testing.assert_allclose(ens.get(0), [0.0])
        ens.add(1, np.array([[40.0], [50.0], [60.0]]))
        np.testing.assert_allclose(ens.get(1), [(10.0 + 0.5 * 40.0) / 1.5])

    def test_temporal_ensembler_drops_expired_chunks(self):
        ens = TemporalEnsembler(k=0.01)
        ens.add(0, np.zeros((2, 1)))
        ens.add(2, np.ones((2, 1)))
        np.testing.assert_allclose(ens.get(2), [1.0])

    def test_split_stereo_rgb(self):
        frame = np.zeros((2, 4, 3), dtype=np.uint8)
        frame[:, :2] = (1, 2, 3)  # BGR
        frame[:, 2:] = (4, 5, 6)
        left, right = split_stereo_rgb(frame)
        np.testing.assert_array_equal(left[0, 0], (3, 2, 1))
        np.testing.assert_array_equal(right[0, 0], (6, 5, 4))

    def test_group_slices(self):
        slices = group_slices([["left_arm", 7], ["right_arm", 7], ["neck", 2]])
        self.assertEqual(slices["right_arm"], slice(7, 14))
        self.assertEqual(slices["neck"], slice(14, 16))


@unittest.skipUnless(HAVE_TORCH, "torch/torchvision not installed")
class ACTRunnerTests(unittest.TestCase):
    def test_loads_train_act_checkpoint_and_predicts(self):
        from host.deploy_act import ACTRunner, load_train_module

        train_act = load_train_module(DEFAULT_TRAIN_SCRIPT)

        cfg = train_act.ACTConfig(state_dim=30, action_dim=30, num_cameras=2, chunk_size=5,
                                  hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1,
                                  dec_layers=1, pretrained_backbone=False)
        model = train_act.ACT(cfg)
        stats = {"qpos_mean": [0.0] * 30, "qpos_std": [1.0] * 30,
                 "action_mean": [2.0] * 30, "action_std": [1.0] * 30}
        groups = [["left_arm", 7], ["right_arm", 7], ["left_ee", 7], ["right_ee", 7], ["neck", 2]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.ckpt"
            torch.save({"model": model.state_dict(), "config": vars(cfg),
                        "norm_stats": stats, "joint_groups": groups,
                        "camera_keys": ["color_0", "color_1"], "image_size": [32, 48]}, path)
            runner = ACTRunner(path, DEFAULT_TRAIN_SCRIPT, "cpu")
        images = split_stereo_rgb(np.zeros((60, 160, 3), dtype=np.uint8))
        chunk = runner.predict(images, np.zeros(30))
        self.assertEqual(chunk.shape, (5, 30))
        self.assertEqual(runner.joint_groups[-1], ("neck", 2))


if __name__ == "__main__":
    unittest.main()

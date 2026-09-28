import json
import sys
import tempfile
from pathlib import Path
import types
import unittest

import numpy as np

try:
    import yaml
except ImportError:
    # The production PC2 environment installs PyYAML through requirements.txt.
    # JSON is valid YAML, so this tiny test shim is enough for config fixtures.
    yaml = types.ModuleType("yaml")
    yaml.YAMLError = ValueError
    yaml.safe_load = json.load
    yaml.safe_dump = json.dumps
    sys.modules["yaml"] = yaml

try:
    import cv2  # noqa: F401
except ImportError:
    # Config validation does not encode frames; production gets cv2 from
    # teleimager's installation.
    sys.modules["cv2"] = types.ModuleType("cv2")

from pc2.zed_stereo import (
    ZedCaptureOptions,
    ZedStereoError,
    pack_left_right,
)
from pc2.zed_teleimager_server import load_and_validate_config


class ZedStereoPackingTests(unittest.TestCase):
    def test_packs_bgra_left_then_right_as_bgr(self):
        left = np.zeros((2, 3, 4), dtype=np.uint8)
        right = np.zeros((2, 3, 4), dtype=np.uint8)
        left[:, :, 0] = 11
        right[:, :, 2] = 22

        packed = pack_left_right(left, right, (2, 3))

        self.assertEqual(packed.shape, (2, 6, 3))
        np.testing.assert_array_equal(packed[:, :3, 0], 11)
        np.testing.assert_array_equal(packed[:, 3:, 2], 22)

    def test_rejects_mismatched_eyes(self):
        with self.assertRaisesRegex(ZedStereoError, "left/right shapes differ"):
            pack_left_right(
                np.zeros((2, 3, 3), dtype=np.uint8),
                np.zeros((2, 4, 3), dtype=np.uint8),
            )

    def test_resolution_shapes_match_televuer_layout(self):
        self.assertEqual(ZedCaptureOptions("hd720").combined_shape, (720, 2560))
        self.assertEqual(ZedCaptureOptions("vga").combined_shape, (376, 1344))


class ZedTeleimagerConfigTests(unittest.TestCase):
    def _write_config(self, config):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "camera.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _valid_config(self):
        return {
            "head_camera": {
                "enable_zmq": True,
                "enable_webrtc": True,
                "binocular": True,
                "image_shape": [720, 2560],
                "fps": 60,
                "zed_resolution": "hd720",
            },
            "left_wrist_camera": {
                "enable_zmq": False,
                "enable_webrtc": False,
            },
            "right_wrist_camera": {
                "enable_zmq": False,
                "enable_webrtc": False,
            },
        }

    def test_accepts_active_stereo_only_config(self):
        config, options = load_and_validate_config(
            self._write_config(self._valid_config())
        )
        self.assertTrue(config["head_camera"]["binocular"])
        self.assertEqual(options.combined_shape, (720, 2560))

    def test_rejects_wrong_combined_shape(self):
        config = self._valid_config()
        config["head_camera"]["image_shape"] = [720, 1280]
        with self.assertRaisesRegex(ValueError, "combined left-right frame"):
            load_and_validate_config(self._write_config(config))

    def test_rejects_static_or_wrist_camera_for_paper_baseline(self):
        config = self._valid_config()
        config["left_wrist_camera"]["enable_zmq"] = True
        with self.assertRaisesRegex(ValueError, "active-stereo-only baseline"):
            load_and_validate_config(self._write_config(config))


if __name__ == "__main__":
    unittest.main()

"""Host-side Dex3-1 hand retargeting for bridge mode.

In bridge mode the Orin only publishes joint commands, so the Host turns Quest
hand-tracking landmarks into Dex3-1 joint targets here, reusing the same
HandRetargeting the robot fork uses (teleop.robot_control.hand_retargeting).
The xr_teleoperate repo must be on sys.path (the recorder adds it) and
dex_retargeting installed, so the import is deferred to construction time.
"""

from __future__ import annotations

from typing import Any

DEX3_NUM_MOTORS = 7


class HandRetarget:
    def __init__(self) -> None:
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
        self._hr = HandRetargeting(HandType.UNITREE_DEX3)

    def _one(self, hand_data: Any, retargeting, indices, remap) -> list:
        import numpy as np
        hand_data = np.asarray(hand_data, dtype=float).reshape(25, 3)
        # Match Dex3_1_Controller: an all-zero skeleton means "open hand".
        if np.all(hand_data == 0.0):
            return [0.0] * DEX3_NUM_MOTORS
        ref_value = hand_data[indices[1, :]] - hand_data[indices[0, :]]
        return list(retargeting.retarget(ref_value)[remap])

    def retarget(self, left_hand_pos: Any, right_hand_pos: Any) -> tuple[list, list]:
        """Return (left_q, right_q), each a list of 7 Dex3-1 joint targets."""
        left = self._one(left_hand_pos, self._hr.left_retargeting,
                         self._hr.left_indices, self._hr.left_dex_retargeting_to_hardware)
        right = self._one(right_hand_pos, self._hr.right_retargeting,
                          self._hr.right_indices, self._hr.right_dex_retargeting_to_hardware)
        return left, right

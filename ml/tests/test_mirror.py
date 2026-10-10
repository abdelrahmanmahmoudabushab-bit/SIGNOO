"""A mirrored clip must be what a mirror would show: the person's left side becomes
their right. Flipping x without swapping the paired body joints produced a body whose
"left shoulder" sat on the right, which no real signer ever looks like.
"""
from __future__ import annotations

import numpy as np

from core.features import FRAME_DIM, POSE_DIM, mirror

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
NOSE = 0


def _clip() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(4, FRAME_DIM)).astype(np.float32)


def _joint(clip: np.ndarray, index: int) -> np.ndarray:
    return clip[:, index * 3: index * 3 + 3]


def test_mirroring_swaps_paired_body_joints_and_flips_x():
    clip = _clip()
    flipped = mirror(clip)

    for left, right in ((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_WRIST, RIGHT_WRIST)):
        expected = _joint(clip, right) * np.array([-1, 1, 1], dtype=np.float32)
        np.testing.assert_allclose(_joint(flipped, left), expected)

    # An unpaired joint stays in its slot, x negated.
    np.testing.assert_allclose(_joint(flipped, NOSE), _joint(clip, NOSE) * [-1, 1, 1])


def test_mirroring_twice_gives_the_original_clip():
    clip = _clip()
    np.testing.assert_allclose(mirror(mirror(clip)), clip)


def test_mirroring_still_swaps_the_hand_blocks():
    clip = _clip()
    flipped = mirror(clip)
    hand = (FRAME_DIM - POSE_DIM) // 2
    # presence flag travels with its hand
    assert flipped[0, POSE_DIM] == clip[0, POSE_DIM + hand]

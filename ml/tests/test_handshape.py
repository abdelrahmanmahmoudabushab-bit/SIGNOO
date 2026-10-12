"""Hand-shape features (feature version 2) and the model_input contract."""
from __future__ import annotations

import numpy as np
import pytest

from core.features import (FRAME_DIM, HAND_DIM, HAND_SHAPE_PER_HAND, HANDSHAPE_DIM, INPUT_DIMS, POSE_DIM,
                           SEQ_LEN, add_velocity, handshape_features, mirror, model_input)

LEFT_FLAG, RIGHT_FLAG = POSE_DIM, POSE_DIM + 1 + HAND_DIM


def _hand(curl: float) -> np.ndarray:
    """(21, 3) hand: curl 0 = fingers straight, 1 = a fist. Wrist at the origin, palm toward -z."""
    pts = np.zeros((21, 3), dtype=np.float32)
    knuckle_x = {1: -0.5, 5: -0.3, 9: 0.0, 13: 0.3, 17: 0.55}
    for base, x in knuckle_x.items():
        pts[base] = [x, 1.0 if base != 1 else 0.4, 0.0]
        direction = np.array([0.0, 1.0, 0.0]) if base != 1 else np.array([-0.7, 0.7, 0.0])
        position, heading = pts[base].astype(np.float64), direction
        for k in range(1, 4):
            heading = _rotate_about_x(heading, curl * 1.2)
            position = position + 0.45 * heading
            pts[base + k] = position
    return pts


def _rotate_about_x(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _clip(left=None, right=None, frames=SEQ_LEN) -> np.ndarray:
    clip = np.zeros((frames, FRAME_DIM), dtype=np.float32)
    for flag, hand in ((LEFT_FLAG, left), (RIGHT_FLAG, right)):
        if hand is not None:
            clip[:, flag] = 1.0
            clip[:, flag + 1: flag + 1 + HAND_DIM] = hand.ravel()
    return clip


def test_shapes_and_version_1_is_unchanged():
    clip = _clip(left=_hand(0.0), right=_hand(1.0))
    assert handshape_features(clip).shape == (SEQ_LEN, HANDSHAPE_DIM) == (SEQ_LEN, 2 * HAND_SHAPE_PER_HAND)
    assert model_input(clip, 1).shape == (SEQ_LEN, INPUT_DIMS[1]) == (SEQ_LEN, FRAME_DIM * 2)
    np.testing.assert_array_equal(model_input(clip, 1), add_velocity(clip))
    assert model_input(clip, 2).shape == (SEQ_LEN, INPUT_DIMS[2])


def test_unknown_feature_version_is_refused():
    with pytest.raises(ValueError):
        model_input(_clip(), 3)


def test_absent_hand_has_zero_shape_and_present_hand_does_not():
    shape = handshape_features(_clip(left=None, right=_hand(0.3)))
    assert not shape[:, :HAND_SHAPE_PER_HAND].any()
    assert shape[:, HAND_SHAPE_PER_HAND:].any()


def test_a_fist_and_an_open_hand_look_different():
    open_hand = handshape_features(_clip(right=_hand(0.0)))[0, HAND_SHAPE_PER_HAND:]
    fist = handshape_features(_clip(right=_hand(1.0)))[0, HAND_SHAPE_PER_HAND:]
    # Flexion cosines (first 15 values): straight fingers ~1, curled well below.
    assert open_hand[:15].mean() > 0.95
    assert fist[:15].mean() < open_hand[:15].mean() - 0.2
    # Fingertips reach less far from the wrist when curled (values 19..23).
    assert fist[19:24].mean() < open_hand[19:24].mean()


def test_scale_and_position_do_not_change_hand_shape():
    hand = _hand(0.6)
    base = handshape_features(_clip(right=hand))
    moved = handshape_features(_clip(right=hand * 2.5 + np.array([3.0, -1.0, 0.5], dtype=np.float32)))
    np.testing.assert_allclose(moved, base, atol=1e-4)


def test_mirroring_swaps_hands_and_keeps_shape():
    left, right = _hand(0.2), _hand(0.9)
    clip = _clip(left=left, right=right)
    base = handshape_features(clip)
    flipped = handshape_features(mirror(clip))
    # Everything except the palm normal (last 3 values of each hand) is mirror-invariant.
    keep = HAND_SHAPE_PER_HAND - 3
    np.testing.assert_allclose(flipped[:, :keep], base[:, HAND_SHAPE_PER_HAND:HAND_SHAPE_PER_HAND + keep], atol=1e-4)
    np.testing.assert_allclose(flipped[:, HAND_SHAPE_PER_HAND:HAND_SHAPE_PER_HAND + keep], base[:, :keep], atol=1e-4)


def _face(brow_gap=0.1, mouth_gap=0.05, scale=1.0, shift=(0.0, 0.0)) -> np.ndarray:
    """(468, 3) face mesh with only the points face_features reads placed; cheek to cheek = 2."""
    f = np.zeros((468, 3), dtype=np.float32)
    pts = {234: (-1.0, 0.0), 454: (1.0, 0.0), 33: (-0.5, -0.3), 263: (0.5, -0.3), 1: (0.0, 0.2),
           159: (-0.5, -0.35), 145: (-0.5, -0.25), 386: (0.5, -0.35), 374: (0.5, -0.25),
           105: (-0.5, -0.35 - brow_gap), 334: (0.5, -0.35 - brow_gap), 107: (-0.2, -0.6), 336: (0.2, -0.6),
           13: (0.0, 0.5 - mouth_gap / 2), 14: (0.0, 0.5 + mouth_gap / 2), 61: (-0.3, 0.5), 291: (0.3, 0.5)}
    for i, (x, y) in pts.items():
        f[i] = [x * scale + shift[0], y * scale + shift[1], 0.0]
    return f


def test_face_features_respond_to_expression_and_ignore_distance():
    from core.features import FACE_DIM, face_features
    base = face_features(_face())
    assert base.shape == (FACE_DIM,) and np.isfinite(base).all()
    assert face_features(_face(brow_gap=0.3))[0] > base[0]            # raised brow
    assert face_features(_face(mouth_gap=0.3))[4] > base[4]           # open mouth
    np.testing.assert_allclose(face_features(_face(scale=0.4, shift=(0.2, -0.1))), base, atol=1e-4)


def test_no_face_gives_zeros_and_version_3_refuses_to_run_without_one():
    from core.features import FACE_DIM, face_features
    assert not face_features(None).any()
    assert not face_features(np.zeros((10, 3))).any()                 # too few points to be a mesh
    clip = _clip(right=_hand(0.5))
    face = np.zeros((SEQ_LEN, FACE_DIM), dtype=np.float32)
    assert model_input(clip, 3, face=face).shape == (SEQ_LEN, INPUT_DIMS[3])
    with pytest.raises(ValueError):
        model_input(clip, 3)
    with pytest.raises(ValueError):
        model_input(clip, 3, face=face[:10])


def _with_gap(first, last, hand_fn=lambda t: _hand(0.1 + 0.01 * t), side=RIGHT_FLAG):
    """A clip whose right hand moves over time and is absent on frames [first, last]."""
    clip = np.zeros((SEQ_LEN, FRAME_DIM), dtype=np.float32)
    for t in range(SEQ_LEN):
        if not first <= t <= last:
            clip[t, side] = 1.0
            clip[t, side + 1: side + 1 + HAND_DIM] = (hand_fn(t) + np.array([0.01 * t, 0, 0])).ravel()
    return clip


def test_short_hand_dropouts_are_bridged_and_long_ones_are_left_alone():
    from core.features import MAX_FILLED_GAP, fill_hand_gaps
    short = _with_gap(20, 20 + MAX_FILLED_GAP - 1)
    filled = fill_hand_gaps(short)
    assert (filled[20: 20 + MAX_FILLED_GAP, RIGHT_FLAG] == 1.0).all()                # now present
    before = filled[19, RIGHT_FLAG + 1: RIGHT_FLAG + 4]
    after = filled[20 + MAX_FILLED_GAP, RIGHT_FLAG + 1: RIGHT_FLAG + 4]
    middle = filled[20 + MAX_FILLED_GAP // 2, RIGHT_FLAG + 1: RIGHT_FLAG + 4]
    assert np.all(np.minimum(before, after) - 1e-5 <= middle) and np.all(middle <= np.maximum(before, after) + 1e-5)

    long_gap = _with_gap(20, 20 + MAX_FILLED_GAP)                                     # one frame too long
    assert (fill_hand_gaps(long_gap)[20: 21 + MAX_FILLED_GAP, RIGHT_FLAG] == 0.0).all()
    edge = _with_gap(0, 1)                                                            # hand appears late: not a blink
    assert (fill_hand_gaps(edge)[0:2, RIGHT_FLAG] == 0.0).all()
    tail = _with_gap(SEQ_LEN - 2, SEQ_LEN - 1)                                        # hand leaves: not a blink
    assert (fill_hand_gaps(tail)[-2:, RIGHT_FLAG] == 0.0).all()
    untouched = _clip(left=_hand(0.2), right=_hand(0.2))
    np.testing.assert_array_equal(fill_hand_gaps(untouched), untouched)


def test_planar_hand_shape_ignores_depth_and_the_default_does_not():
    from core.features import handshape_features
    flat = _hand(0.7)
    deep = flat.copy()
    deep[:, 2] += np.linspace(-0.3, 0.3, 21)                                          # depth noise only
    a, b = _clip(right=flat), _clip(right=deep)
    np.testing.assert_allclose(handshape_features(a, planar=True), handshape_features(b, planar=True), atol=1e-5)
    assert np.abs(handshape_features(a) - handshape_features(b)).max() > 1e-3


def test_feature_versions_4_and_5_carry_the_duration_and_refuse_to_run_without_it():
    from core.features import duration_column
    clip = _clip(left=_hand(0.3), right=_hand(0.8))
    for version in (4, 5):
        x = model_input(clip, version, duration=1.5)
        assert x.shape == (SEQ_LEN, INPUT_DIMS[version])
        with pytest.raises(ValueError):
            model_input(clip, version)
    slow, fast = model_input(clip, 4, duration=3.0), model_input(clip, 4, duration=0.6)
    assert not np.allclose(slow, fast)                                                 # duration reaches the model
    assert abs(duration_column(1.5, 4)).max() == 0.0
    assert duration_column(3.0, 4)[0, 0] > 0 > duration_column(0.75, 4)[0, 0]
    assert np.isfinite(duration_column(0.0, 4)).all()                                  # a zero-length clip cannot blow up


def test_degenerate_hand_gives_finite_zeros_not_nan():
    shape = handshape_features(_clip(right=np.zeros((21, 3), dtype=np.float32)))
    assert np.isfinite(shape).all()

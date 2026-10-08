"""Train/inference feature-skew check (SYSTEM_DESIGN.md §7).

`ml/train.py` and `kiosk/recognition.py` must both compute features by calling
`core.features` — never a hand-duplicated copy — or a kiosk model can silently
diverge from what it was trained on. This test proves it by calling the exact same
functions `Recognizer._features` uses, and separately doing what `ml/extract_landmarks.py`
does at training time (`normalise_frame` -> `resample`), on synthetic landmarks
built to actually exercise both hands and the pose (a trivial all-zero fixture
would pass for the wrong reason — signoo-engineering: "fixtures must be honest").
"""
from __future__ import annotations

import numpy as np

from core.contracts import Landmarks, Utterance
from core.features import FRAME_DIM, N_HAND, POSE_UPPER, add_velocity, normalise_frame, resample
from kiosk.recognition import Recognizer


def _synthetic_landmarks(n_frames: int = 20, seed: int = 0) -> list[Landmarks]:
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[11] = [-0.2, 0.5, 0.0]     # left shoulder
        pose[12] = [0.2, 0.5, 0.0]      # right shoulder
        pose[: len(POSE_UPPER)] += rng.normal(0, 0.01, (len(POSE_UPPER), 3))
        left = rng.normal(0, 0.05, (N_HAND, 3)).astype(np.float32) + [-0.3, 0.3, 0.0]
        right = rng.normal(0, 0.05, (N_HAND, 3)).astype(np.float32) + [0.3, 0.3, 0.0]
        frames.append(Landmarks(ts_ms=t * 33, pose=pose, left=left, right=right))
    return frames


def test_kiosk_recognizer_matches_training_time_features_bit_for_bit():
    landmarks = _synthetic_landmarks()

    # "Training-time" path, exactly as ml/extract_landmarks.py does it.
    training_frames = [normalise_frame(lm.pose, lm.left, lm.right) for lm in landmarks]
    training_clip = add_velocity(resample(training_frames))

    # "Kiosk-time" path — the private method kiosk/recognition.py actually calls
    # from Recognizer.classify() on every utterance.
    utterance = Utterance(landmarks=landmarks, t_start=landmarks[0].ts_ms, t_end=landmarks[-1].ts_ms)
    kiosk_clip = Recognizer._features(utterance)

    assert kiosk_clip.shape == training_clip.shape == (64, FRAME_DIM * 2)
    np.testing.assert_array_equal(kiosk_clip, training_clip)


def test_missing_hand_stays_none_not_zero():
    """SYSTEM_DESIGN.md §1.1: `None` for an absent hand is distinct from an
    all-zero array — collapsing that distinction silently corrupts features. This
    guards the boundary contract itself, not just the numeric result.
    """
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11], pose[12] = [-0.2, 0.5, 0.0], [0.2, 0.5, 0.0]
    landmarks = Landmarks(ts_ms=0, pose=pose, left=None, right=np.zeros((21, 3), dtype=np.float32))

    frame = normalise_frame(landmarks.pose, landmarks.left, landmarks.right)
    from core.features import HAND_DIM, POSE_DIM

    assert frame[POSE_DIM] == 0.0          # left-hand presence flag: absent
    assert frame[POSE_DIM + 1 + HAND_DIM] == 1.0   # right-hand presence flag: present

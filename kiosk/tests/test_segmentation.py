"""B3 Segmentation state machine (SYSTEM_DESIGN.md B3): IDLE -> ACTIVE on hands
entering, an Utterance emitted at a pause, and back to IDLE when hands leave.
"""
from __future__ import annotations

import numpy as np

from core.contracts import Landmarks
from kiosk.segmentation import SegmentState, Segmenter

POSE = np.zeros((25, 3), dtype=np.float32)


def _hand(x: float) -> np.ndarray:
    return np.full((21, 3), x, dtype=np.float32)


def _seg() -> Segmenter:
    return Segmenter(pause_ms=200, motion_threshold=0.05, hands_absent_ms=300, min_frames=3)


def test_idle_until_hands_enter():
    seg = _seg()
    assert seg.state == SegmentState.IDLE
    no_hands = Landmarks(ts_ms=0, pose=POSE, left=None, right=None)
    assert seg.feed(no_hands) is None
    assert seg.state == SegmentState.IDLE


def test_pause_emits_utterance_then_resumes_active():
    seg = _seg()
    t = 0
    # Hands enter and move for a while.
    for i in range(5):
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None))
    assert seg.state == SegmentState.ACTIVE

    # Then hold still past pause_ms.
    utterance = None
    still_x = 0.4
    while utterance is None:
        t += 30
        utterance = seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(still_x), right=None))
        if t > 2000:
            raise AssertionError("boundary never fired")

    assert utterance.t_start < utterance.t_end
    assert len(utterance.landmarks) >= 3
    # Ready to capture the next word immediately — not stuck in BOUNDARY.
    assert seg.state == SegmentState.ACTIVE


def test_a_word_ends_when_the_hands_leave_not_only_when_they_hold_still():
    """People finish a sign by dropping their hands. Waiting for a held pause would
    lose almost every real word, so leaving the frame closes the utterance."""
    seg = _seg()
    t, emitted = 0, []
    for i in range(8):                                     # sign: hands keep moving, never still
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None)))
    assert not any(emitted)
    for _ in range(20):                                    # then the hands drop out of frame
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=None, right=None)))

    utterances = [u for u in emitted if u is not None]
    assert len(utterances) == 1
    assert len(utterances[0].landmarks) == 8
    assert seg.state == SegmentState.IDLE


def test_a_brief_flicker_of_the_hand_is_not_a_word():
    seg = _seg()
    t = 0
    for i in range(2):                                     # fewer frames than min_frames
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None))
    emitted = []
    for _ in range(20):
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=None, right=None)))
    assert not any(emitted)
    assert seg.state == SegmentState.IDLE


def test_a_hand_appearing_is_moving_and_a_missing_hand_is_not():
    from kiosk.segmentation import _hands_motion

    assert _hands_motion([], [_hand(0.5)]) == float("inf")
    assert _hands_motion([_hand(0.5)], []) == 0.0
    assert _hands_motion([], []) == 0.0


def test_motion_ignores_left_right_label_flips():
    """MediaPipe reports the same hand as left on one frame and right on the next; that
    must not read as a hand vanishing and a new one appearing."""
    seg = _seg()
    t = 0
    for i in range(6):
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None))
    # The same, still hand now labelled as the right one on alternate frames.
    emitted = []
    for k in range(40):
        t += 30
        same = _hand(0.5)
        hand = {"left": same, "right": None} if k % 2 == 0 else {"left": None, "right": same}
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, **hand)))
    assert any(emitted), "a still hand with flipping labels should still reach the pause boundary"


def test_one_noisy_frame_does_not_restart_the_pause_timer():
    seg = _seg()
    t, emitted = 0, []
    for i in range(6):                                     # move, then hold still
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None))
    positions = [0.5] * 4 + [0.9] + [0.5] * 12             # a single jittery outlier mid-pause
    for x in positions:
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(x), right=None)))
    assert any(emitted)


def test_a_hand_resting_in_view_is_not_emitted_over_and_over():
    seg = _seg()
    t, emitted = 0, []
    for i in range(6):
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None)))
    for _ in range(80):                                    # then rests, perfectly still, for 2.4 s
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(0.5), right=None)))
    assert sum(u is not None for u in emitted) == 1

    for i in range(6):                                     # moving again starts a new word...
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(0.5 + i * 0.1), right=None)))
    for _ in range(20):
        t += 30
        emitted.append(seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(1.0), right=None)))
    assert sum(u is not None for u in emitted) == 2        # ...which is emitted at its own pause


def test_hands_leaving_returns_to_idle():
    seg = _seg()
    t = 0
    for i in range(3):
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=_hand(i * 0.1), right=None))
    assert seg.state == SegmentState.ACTIVE

    for _ in range(20):
        t += 30
        seg.feed(Landmarks(ts_ms=t, pose=POSE, left=None, right=None))
    assert seg.state == SegmentState.IDLE

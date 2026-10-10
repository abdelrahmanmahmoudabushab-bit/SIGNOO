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


def test_a_hand_that_just_came_back_is_moving_not_still():
    """With one hand absent in the previous frame, motion used to read as 0 and a
    flickering hand looked like a held pause."""
    from kiosk.segmentation import _hand_motion

    assert _hand_motion(None, _hand(0.5)) == float("inf")
    assert _hand_motion(_hand(0.5), None) == 0.0
    assert _hand_motion(None, None) == 0.0


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

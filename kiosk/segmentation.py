"""B3 Segmentation: utterance boundary detection (SYSTEM_DESIGN.md B3).

    IDLE ──hands enter frame──> ACTIVE ──hands hold still for N ms──> emit word
      ^                           |  ──hands leave the frame for M ms──> emit word
      +─────────reset────────────+

A word ends either when the hands hold still (a deliberate pause) or when they leave
the frame (how people actually finish a sign). After a word is emitted, the next one
starts only once the hands move again, so a hand resting in view is not read as an
endless series of words.

Honest about the limit (as the design doc is): this reads deliberate, word-by-word
signing and fluent connected signing poorly. Thresholds are config
(`kiosk/config.py`), never hardcoded, because signing tempo differs between languages
and people.

Two properties come from measuring this on real recordings (`kiosk/replay.py`):
hand tracking is noisy, and MediaPipe's left/right labels flip between frames. So
motion is matched by position rather than by label, and smoothed over a few frames so
a single noisy frame cannot restart the pause timer.
"""
from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.contracts import Landmarks, Utterance

SMOOTHING_FRAMES = 3


class SegmentState(str, enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BOUNDARY = "boundary"      # one-tick transient state, emitted the instant an
                                # utterance closes; the state machine drops back to
                                # ACTIVE or IDLE on the very next frame.


def _present_hands(landmarks: Landmarks) -> list[np.ndarray]:
    return [h for h in (landmarks.left, landmarks.right) if h is not None]


def _hands_motion(prev: list[np.ndarray], cur: list[np.ndarray]) -> float:
    """How far the hands moved between two frames, ignoring left/right labels.

    Each current hand is compared with the nearest hand of the previous frame, so a
    label flip (the same hand reported as left, then right) reads as no motion. A
    frame with no hands contributes nothing, but a hand appearing from nothing is by
    definition not still.
    """
    if not cur:
        return 0.0
    if not prev:
        return float("inf")
    return max(
        min(float(np.mean(np.linalg.norm(c - p, axis=-1))) for p in prev)
        for c in cur
    )


def _median(values) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


@dataclass
class Segmenter:
    """Feed it one `Landmarks` per processed frame; get an `Utterance` back at each
    boundary, or `None` while still accumulating.
    """

    pause_ms: int
    motion_threshold: float
    hands_absent_ms: int
    min_frames: int
    # Hands that never pause would otherwise buffer frames for as long as they move.
    # At this many frames the word is closed and emitted as it stands.
    max_frames: int = 45
    # A word must also last at least this long, first frame to last. `min_frames` alone
    # is a length in frames, so its meaning in time depends on how fast the tracker runs:
    # 6 frames is 0.2 s at 30 tracked frames a second and 0.6 s at the Jetson's 10, where
    # it threw away ordinary quick signs without a word (measured with kiosk/replay.py
    # --tracker-fps, 2026-10-13). Time is what separates a sign from a tracking flicker.
    min_ms: int = 0
    # The same point for the upper limit: a word still open after this long is closed as
    # it stands. `max_frames` stays as the bound on memory; 0 here means no time limit.
    max_ms: int = 0

    state: SegmentState = SegmentState.IDLE
    _buffer: list[Landmarks] = field(default_factory=list)
    _prev_hands: list = field(default_factory=list)
    _recent_motion: deque = field(default_factory=lambda: deque(maxlen=SMOOTHING_FRAMES))
    _armed: bool = True            # False right after a word, until the hands move again
    _still_since_ms: int | None = None
    _hands_absent_since_ms: int | None = None
    # Frames where the hands vanished in the middle of the word being buffered. A gap is
    # only counted once the hands come back: hands that leave and stay away are how a
    # sign ends, not a hole in it.
    _gap_frames: int = 0
    _open_gap: int = 0

    @property
    def buffering(self) -> bool:
        """A word is being collected right now: what the screen shows as "signing"."""
        return self.state != SegmentState.IDLE and self._armed and bool(self._buffer)

    def reset(self) -> None:
        self.state = SegmentState.IDLE
        self._buffer.clear()
        self._prev_hands = []
        self._recent_motion.clear()
        self._armed = True
        self._still_since_ms = self._hands_absent_since_ms = None
        self._gap_frames = self._open_gap = 0

    def feed(self, landmarks: Landmarks) -> Utterance | None:
        hands = _present_hands(landmarks)

        if self.state == SegmentState.IDLE:
            if hands:
                self.state = SegmentState.ACTIVE
                self._buffer = [landmarks]
                self._prev_hands = hands
                self._recent_motion.clear()
                self._armed = True
                self._still_since_ms = None
                self._hands_absent_since_ms = None
                self._gap_frames = self._open_gap = 0
            return None

        # ACTIVE (or just-emitted BOUNDARY, which behaves like ACTIVE going forward)
        if not hands:
            if self._armed and self._buffer:
                self._open_gap += 1
            # A pause is timed over frames that show a still hand. Carried across a gap,
            # the old start time would close the word the instant the hands came back.
            self._still_since_ms = None
            self._hands_absent_since_ms = (
                landmarks.ts_ms if self._hands_absent_since_ms is None else self._hands_absent_since_ms
            )
            if landmarks.ts_ms - self._hands_absent_since_ms >= self.hands_absent_ms:
                # Dropping the hands is how people end a sign, so it closes the word.
                # Only frames with hands are buffered, so nothing idle is included, and
                # a flicker too short to be a sign is still discarded.
                utterance = self._close() if self._armed and self._long_enough() else None
                self.reset()
                return utterance
            return None
        self._hands_absent_since_ms = None
        self._gap_frames += self._open_gap      # the hands came back: that was a hole in the word
        self._open_gap = 0

        self._recent_motion.append(_hands_motion(self._prev_hands, hands))
        self._prev_hands = hands
        motion = _median(self._recent_motion)

        if not self._armed:
            # A word was just emitted. Wait for the hands to move before buffering the
            # next one; a hand held in view must not be emitted again and again.
            if motion >= self.motion_threshold:
                self._armed = True
                self._buffer = [landmarks]
                self._still_since_ms = None
            self._gap_frames = 0                # nothing before the new word's first frame counts
            return None

        self._buffer.append(landmarks)

        if motion < self.motion_threshold:
            self._still_since_ms = (
                landmarks.ts_ms if self._still_since_ms is None else self._still_since_ms
            )
        else:
            self._still_since_ms = None

        paused = (
            self._still_since_ms is not None
            and landmarks.ts_ms - self._still_since_ms >= self.pause_ms
            and self._long_enough()
        )
        too_long = len(self._buffer) >= self.max_frames or (
            self.max_ms > 0 and landmarks.ts_ms - self._buffer[0].ts_ms >= self.max_ms)
        if paused or too_long:
            self.state = SegmentState.BOUNDARY
            utterance = self._close()
            # Ready for the next word once the hands move again; ACTIVE semantics resume.
            self.state = SegmentState.ACTIVE
            self._buffer = []
            self._still_since_ms = None
            self._armed = False
            self._gap_frames = self._open_gap = 0
            return utterance

        return None

    def _long_enough(self) -> bool:
        """Enough frames for the model to read, spread over enough time to be a sign."""
        return (len(self._buffer) >= self.min_frames
                and self._buffer[-1].ts_ms - self._buffer[0].ts_ms >= self.min_ms)

    def _close(self) -> Utterance:
        return Utterance(
            landmarks=list(self._buffer),
            t_start=self._buffer[0].ts_ms,
            t_end=self._buffer[-1].ts_ms,
            frames_without_hands=self._gap_frames,
        )

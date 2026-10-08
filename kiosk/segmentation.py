"""B3 Segmentation: pause-based utterance boundary detection (SYSTEM_DESIGN.md B3).

    IDLE ──hands enter frame──> ACTIVE ──motion below threshold for N ms──> BOUNDARY
      ^                            |                                            |
      +────hands leave frame───────+────────────────────────────────────────────+

Honest about the limit (as the design doc is): this reads deliberate, paused signing
well and fluent connected signing poorly. Thresholds are config (`kiosk/config.py`),
never hardcoded, because signing tempo differs between languages and people.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from core.contracts import Landmarks, Utterance


class SegmentState(str, enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BOUNDARY = "boundary"      # one-tick transient state, emitted the instant an
                                # utterance closes; the state machine drops back to
                                # ACTIVE or IDLE on the very next frame.


def _hand_motion(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    """Mean per-landmark displacement between two hand arrays, or 0 if either is absent."""
    if prev is None or cur is None:
        return 0.0
    return float(np.mean(np.linalg.norm(cur - prev, axis=-1)))


@dataclass
class Segmenter:
    """Feed it one `Landmarks` per processed frame; get an `Utterance` back at each
    pause boundary, or `None` while still accumulating.
    """

    pause_ms: int
    motion_threshold: float
    hands_absent_ms: int
    min_frames: int

    state: SegmentState = SegmentState.IDLE
    _buffer: list[Landmarks] = field(default_factory=list)
    _prev_left: np.ndarray | None = None
    _prev_right: np.ndarray | None = None
    _still_since_ms: int | None = None
    _hands_absent_since_ms: int | None = None

    def reset(self) -> None:
        self.state = SegmentState.IDLE
        self._buffer.clear()
        self._prev_left = self._prev_right = None
        self._still_since_ms = self._hands_absent_since_ms = None

    def feed(self, landmarks: Landmarks) -> Utterance | None:
        hands_present = landmarks.left is not None or landmarks.right is not None

        if self.state == SegmentState.IDLE:
            if hands_present:
                self.state = SegmentState.ACTIVE
                self._buffer = [landmarks]
                self._prev_left, self._prev_right = landmarks.left, landmarks.right
                self._still_since_ms = None
                self._hands_absent_since_ms = None
            return None

        # ACTIVE (or just-emitted BOUNDARY, which behaves like ACTIVE going forward)
        if not hands_present:
            self._hands_absent_since_ms = (
                landmarks.ts_ms if self._hands_absent_since_ms is None else self._hands_absent_since_ms
            )
            if landmarks.ts_ms - self._hands_absent_since_ms >= self.hands_absent_ms:
                self.reset()
            return None
        self._hands_absent_since_ms = None

        motion = max(
            _hand_motion(self._prev_left, landmarks.left),
            _hand_motion(self._prev_right, landmarks.right),
        )
        self._prev_left, self._prev_right = landmarks.left, landmarks.right
        self._buffer.append(landmarks)

        if motion < self.motion_threshold:
            self._still_since_ms = (
                landmarks.ts_ms if self._still_since_ms is None else self._still_since_ms
            )
        else:
            self._still_since_ms = None

        if (
            self._still_since_ms is not None
            and landmarks.ts_ms - self._still_since_ms >= self.pause_ms
            and len(self._buffer) >= self.min_frames
        ):
            self.state = SegmentState.BOUNDARY
            utterance = Utterance(
                landmarks=list(self._buffer),
                t_start=self._buffer[0].ts_ms,
                t_end=self._buffer[-1].ts_ms,
            )
            # Ready to capture the next word immediately; ACTIVE semantics resume.
            self.state = SegmentState.ACTIVE
            self._buffer = []
            self._still_since_ms = None
            return utterance

        return None

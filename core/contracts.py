"""Data contracts passed between kiosk blocks (see docs/SYSTEM_DESIGN.md §1.1).

Every block takes and returns one of these — no camera handles, no model objects, no
database sessions crossing a boundary. That is what makes NFR-N1 possible: a recorded
session (a list of Landmarks) replays through B2-B6 off-device and reproduces the
output exactly, which is the primary tool for debugging a kiosk misbehaving somewhere
else entirely.

Plain dataclasses, not model classes: nothing here imports torch, cv2, or mediapipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """B1 -> B2: one captured camera frame."""
    ts_ms: int
    image: np.ndarray


@dataclass
class Landmarks:
    """B2 -> B3: one frame's tracked body/hand landmarks.

    `None` for a part means "not detected this frame", distinct from an all-zero
    array — collapsing that distinction silently corrupts downstream features
    (see core/features.py normalise_frame).
    """
    ts_ms: int
    pose: np.ndarray | None
    left: np.ndarray | None
    right: np.ndarray | None


@dataclass
class Utterance:
    """B3 -> B4: one segmented clip, ready for classification."""
    landmarks: list[Landmarks]
    t_start: int
    t_end: int
    # Frames inside the clip where tracking lost both hands and then found them again.
    # They are not in `landmarks` (only frames with a hand are), so this is the only
    # record that the sign has holes in it.
    frames_without_hands: int = 0


@dataclass
class Hypothesis:
    """B4 -> B5: a recognition result, or an explicit rejection.

    `gloss=None` is a first-class outcome (Requirements §6.3): a confident wrong
    answer is worse than an admitted failure, so B4 returns "no hypothesis" rather
    than guessing when nothing clears the rejection threshold.
    """
    gloss: str | None
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Rendered:
    """B5 -> B6/B7: the assembled utterance, in gloss order and as bilingual text."""
    gloss_seq: list[str]
    ar: str
    en: str
    uncertain: list[int] = field(default_factory=list)   # indices into gloss_seq


@dataclass
class Answer:
    """B7 -> B6/B8: a grounded answer to a Rendered utterance.

    `refused=True` means the question fell outside the knowledge base (FR-A1) — shown
    to the visitor as a visible refusal and an escalation route (FR-A5), never
    papered over with an invented answer.
    """
    ar: str
    en: str
    source: str                      # "cloud" | "local_fallback"
    refused: bool = False
    unrenderable_words: list[str] = field(default_factory=list)   # for B8 (FR-A3)

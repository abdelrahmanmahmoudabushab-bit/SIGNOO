"""B4 Recognition: rejection is a first-class output (Requirements §6.3) — anything
below `reject_threshold` must come back as `Hypothesis(gloss=None, ...)`, never a
guess. Builds a tiny throwaway checkpoint with a hand-set head bias so the softmax
output is deterministic, rather than depending on `models/signoo.pt` (a gitignored
build artifact) or on training actually converging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from model import SignTransformer  # noqa: E402

from core.contracts import Landmarks, Utterance
from core.features import FRAME_DIM
from kiosk.recognition import ModelUnavailable, Recognizer


def _make_checkpoint(tmp_path: Path, biases: list[float], names: list[str] | None = None) -> Path:
    label_names = names or [f"sign_{i}" for i in range(len(biases))]
    # ml/train.py saves labels as ml/data.py's label_index() dict ({sign_id: index}),
    # never a plain list — match that shape exactly, or this test would pass for a
    # format the real training pipeline never produces.
    labels = {name: i for i, name in enumerate(label_names)}
    model = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=len(labels))
    with torch.no_grad():
        # Zero every weight so the input cannot move the logits, then set the
        # output bias directly — the softmax becomes a known, fixed distribution
        # regardless of what utterance is classified.
        for p in model.parameters():
            p.zero_()
        model.head[-1].bias.copy_(torch.tensor(biases))

    path = tmp_path / "toy.pt"
    torch.save({"model": model.state_dict(), "labels": labels, "input_dim": FRAME_DIM * 2}, path)
    return path


def _dummy_utterance(n_frames: int = 10) -> Utterance:
    landmarks = [
        Landmarks(ts_ms=i * 30, pose=np.zeros((33, 3), dtype=np.float32),
                   left=np.zeros((21, 3), dtype=np.float32), right=None)
        for i in range(n_frames)
    ]
    return Utterance(landmarks=landmarks, t_start=0, t_end=landmarks[-1].ts_ms)


def test_the_other_class_is_a_rejection_even_when_it_is_very_confident(tmp_path):
    """A model trained to say "not one of my signs" must be read as a refusal to
    answer, never shown to the visitor as a word, and never offered as a candidate."""
    checkpoint = _make_checkpoint(tmp_path, biases=[0.0, 0.0, 10.0], names=["a", "b", "__other__"])
    recognizer = Recognizer(checkpoint, reject_threshold=0.55, confident_threshold=0.8, top_k=3)

    hypothesis = recognizer.classify(_dummy_utterance())

    assert hypothesis.gloss is None
    assert hypothesis.confidence > 0.9                      # it IS sure: sure it is not a known sign
    assert "__other__" not in [label for label, _ in hypothesis.alternatives]


def test_low_confidence_is_rejected_not_guessed(tmp_path):
    # Near-uniform logits -> best-class probability well under any sane threshold.
    checkpoint = _make_checkpoint(tmp_path, biases=[0.0, 0.0, 0.0, 0.0])
    recognizer = Recognizer(checkpoint, reject_threshold=0.55, confident_threshold=0.8, top_k=3)

    hypothesis = recognizer.classify(_dummy_utterance())

    assert hypothesis.gloss is None
    assert hypothesis.confidence < 0.55
    assert len(hypothesis.alternatives) == 3   # still surfaced, just not committed


def test_high_confidence_returns_the_dominant_label(tmp_path):
    biases = [8.0, 0.0, 0.0, 0.0]     # softmax on this is ~ >0.999 for class 0
    checkpoint = _make_checkpoint(tmp_path, biases=biases)
    recognizer = Recognizer(checkpoint, reject_threshold=0.55, confident_threshold=0.8, top_k=3)

    hypothesis = recognizer.classify(_dummy_utterance())

    assert hypothesis.gloss == "sign_0"
    assert hypothesis.confidence > 0.8


def test_missing_checkpoint_raises_model_unavailable(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    try:
        Recognizer(missing, reject_threshold=0.5, confident_threshold=0.8, top_k=3)
        raise AssertionError("expected ModelUnavailable")
    except ModelUnavailable:
        pass

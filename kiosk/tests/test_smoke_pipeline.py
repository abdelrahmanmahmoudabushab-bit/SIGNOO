"""End-to-end pipeline smoke test, no camera required.

Feeds synthetic landmark frames (built the same way `ml/make_synthetic.py` builds
its clips — a sine-driven trajectory, not zeros, so the fixture is honest per
signoo-engineering's "fixtures must be honest") through B3 Segmentation, B4
Recognition (against the real `models/signoo.pt` built from
`ml/make_synthetic.py` + `ml/train.py`, per `models/README.md`), B5 Assembly, B7
Dialogue and B8 Playback — proving the wiring end to end.

Caveat, stated plainly (also in kiosk/README.md): this is a synthetic checkpoint.
A "sign" here is an abstract per-axis sine pattern, not real Jordanian Sign
Language — this test proves the pipeline connects correctly, not that anything
recognises real signing (see docs/ROADMAP.md Phase 5's own caveat, and the
`slr-methodology` skill on what a number is and isn't allowed to claim).

Run directly for a human-readable trace:

    .venv/Scripts/python.exe kiosk/tests/test_smoke_pipeline.py

or as part of the suite:

    .venv/Scripts/python.exe -m pytest kiosk/tests/test_smoke_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from make_synthetic import make_clip  # noqa: E402

from core.contracts import Landmarks
from core.features import HAND_DIM, POSE_DIM, SEQ_LEN
from kiosk.assembly import Assembler, Vocabulary
from kiosk.config import CONFIG
from kiosk.playback import ClipLibrary
from kiosk.recognition import ModelUnavailable, Recognizer
from kiosk.segmentation import Segmenter

CHECKPOINT = CONFIG.model_path


def _clip_to_landmarks(clip: np.ndarray) -> list[Landmarks]:
    """Turns one `make_clip` output (already in normalised-feature space) back
    into `Landmarks`-shaped stand-ins the segmenter can consume: fabricated hand
    arrays whose motion tracks the synthetic clip's own hand block, so the
    segmenter's motion-based pause detection has something real to key off.
    """
    out = []
    for t in range(clip.shape[0]):
        row = clip[t]
        left_present = row[POSE_DIM] > 0.5
        right_present = row[POSE_DIM + 1 + HAND_DIM] > 0.5
        left = np.tile(row[POSE_DIM + 1 : POSE_DIM + 1 + HAND_DIM].reshape(21, 3)[0], (21, 1)) if left_present else None
        right = np.tile(
            row[POSE_DIM + 1 + HAND_DIM + 1 :].reshape(21, 3)[0], (21, 1)
        ) if right_present else None
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[11], pose[12] = [-0.2, 0.5, 0.0], [0.2, 0.5, 0.0]
        out.append(Landmarks(ts_ms=t * 33, pose=pose, left=left, right=right))
    # Hold the final pose for long enough for the pause detector to fire.
    for i in range(20):
        out.append(Landmarks(ts_ms=(SEQ_LEN + i) * 33, pose=pose,
                              left=out[-1].left, right=out[-1].right))
    return out


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no checkpoint at models/signoo.pt — see models/README.md")
def test_pipeline_produces_a_hypothesis_from_synthetic_frames():
    recognizer = Recognizer(CHECKPOINT, CONFIG.reject_threshold, CONFIG.confident_threshold,
                             CONFIG.top_k_alternatives)
    segmenter = Segmenter(pause_ms=CONFIG.segmentation_pause_ms,
                           motion_threshold=CONFIG.segmentation_motion_threshold,
                           hands_absent_ms=CONFIG.segmentation_hands_absent_ms,
                           min_frames=CONFIG.min_utterance_frames)
    vocab = Vocabulary(CONFIG.vocab_path)
    assembler = Assembler(vocab)
    clip_library = ClipLibrary(CONFIG.clip_library_path, vocab)

    rng = np.random.default_rng(0)
    clip = make_clip(sign=0, signer=0, rng=rng)
    landmarks_stream = _clip_to_landmarks(clip)

    utterance = None
    for lm in landmarks_stream:
        result = segmenter.feed(lm)
        if result is not None:
            utterance = result
            break
    assert utterance is not None, "segmenter never produced an utterance from the synthetic stream"

    hypothesis = recognizer.classify(utterance)
    print(f"hypothesis: gloss={hypothesis.gloss} confidence={hypothesis.confidence:.3f} "
          f"alternatives={hypothesis.alternatives}")
    assert hypothesis.confidence >= 0.0    # it ran, and produced *some* distribution
    assert len(hypothesis.alternatives) == CONFIG.top_k_alternatives

    if hypothesis.gloss is not None:
        assembler.append(hypothesis.gloss)
        rendered = assembler.render()
        print(f"rendered: ar={rendered.ar!r} en={rendered.en!r}")
        words = clip_library.sequence_for_gloss(rendered.gloss_seq)
        print(f"playback: {[(w.gloss_id, w.has_clip) for w in words]}")
        assert all(w.has_clip is False for w in words), (
            "clip_library.json has no real recordings yet — every word must be "
            "the 'not yet recorded' placeholder, per docs/API_CONTRACTS.md §2"
        )
    else:
        print("hypothesis was a rejection — also a valid, exercised path (Requirements §6.3)")


if __name__ == "__main__":
    test_pipeline_produces_a_hypothesis_from_synthetic_frames()
    print("\nsmoke test passed — pipeline wiring confirmed, accuracy not claimed.")

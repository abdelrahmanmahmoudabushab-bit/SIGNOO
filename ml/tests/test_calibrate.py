"""Calibration metrics on arrays small enough to check by eye, plus one run of the
script end to end on a tiny random checkpoint to show the pieces connect."""
import json

import numpy as np
import pytest
import torch

import calibrate
from model import SignTransformer

from core.features import FEATURE_SHAPE, FRAME_DIM

CONFIDENT, REJECT = 0.80, 0.55


def test_decide_follows_the_kiosk_commit_candidates_refuse():
    confidence = [0.90, 0.80, 0.79, 0.55, 0.54]
    actions = calibrate.decide(confidence, [0] * 5, CONFIDENT, REJECT)
    # Both bars are inclusive, as in the kiosk: >= commits, < reject refuses.
    assert actions.tolist() == ["commit", "commit", "candidates", "candidates", "refuse"]


def test_the_other_class_is_refused_however_confident():
    actions = calibrate.decide([0.99, 0.99], [2, 1], CONFIDENT, REJECT, other=2)
    assert actions.tolist() == ["refuse", "commit"]


def test_known_shares_separate_right_commits_from_wrong_ones():
    confidence = [0.95, 0.90, 0.70, 0.30]
    predicted = [0, 1, 0, 0]
    truth = [0, 0, 0, 0]                     # the second clip is committed and wrong

    shares = calibrate.known_shares(confidence, predicted, truth, CONFIDENT, REJECT)

    assert shares == {"committed_correct": 0.25, "committed_wrong": 0.25, "candidates": 0.25, "refused": 0.25}


def test_a_higher_bar_moves_commits_to_candidates_not_to_refusals():
    confidence, predicted, truth = [0.95, 0.85, 0.85, 0.30], [0, 0, 1, 0], [0, 0, 0, 0]

    strict = calibrate.known_shares(confidence, predicted, truth, 0.90, REJECT)

    assert strict == {"committed_correct": 0.25, "committed_wrong": 0.0, "candidates": 0.5, "refused": 0.25}


def test_a_correct_sign_refused_as_other_counts_as_refused_not_committed():
    shares = calibrate.known_shares([0.99, 0.99], [2, 0], [0, 0], CONFIDENT, REJECT, other=2)
    assert shares["refused"] == 0.5 and shares["committed_correct"] == 0.5


def test_unknown_shares_count_every_commit_as_false():
    confidence = [0.95, 0.90, 0.70, 0.30, 0.99]
    predicted = [0, 1, 0, 0, 2]              # the last one is the other class

    shares = calibrate.unknown_shares(confidence, predicted, CONFIDENT, REJECT, other=2)

    assert shares == {"false_commit": 0.4, "candidates": 0.2, "refused": 0.4}


def test_auroc_is_one_when_separable_zero_when_inverted_half_when_identical():
    assert calibrate.auroc([0.9, 0.8], [0.3, 0.2, 0.1]) == 1.0
    assert calibrate.auroc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert calibrate.auroc([0.5, 0.5], [0.5]) == 0.5
    # One of the four known/unknown pairs is the wrong way round.
    assert calibrate.auroc([0.9, 0.4], [0.5, 0.1]) == 0.75


def test_acceptance_score_zeroes_clips_the_kiosk_would_refuse_as_other():
    assert calibrate.acceptance_score([0.9, 0.7], [2, 0], other=2).tolist() == [0.0, 0.7]


# ── the script itself ─────────────────────────────────────────────────────────

@pytest.fixture
def pilot(tmp_path):
    """A random two-sign checkpoint and a manifest with a third sign it never saw."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=2)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "labels": {"cat": 0, "dog": 1}, "input_dim": FRAME_DIM * 2},
               checkpoint)

    records = []
    for signer in ("train1", "val1", "test1", "test2"):
        for sign in ("cat", "dog", "book"):
            path = tmp_path / signer / sign / "0.npy"
            path.parent.mkdir(parents=True)
            np.save(path, rng.normal(size=FEATURE_SHAPE).astype(np.float32))
            records.append({"signer_id": signer, "sign_id": sign, "features": f"{signer}/{sign}/0.npy"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"val": ["val1"], "test": ["test1", "test2"]}), encoding="utf-8")
    return checkpoint, manifest, split


def test_calibrate_scores_only_the_test_signers_and_reports_unknown_signs(pilot, capsys):
    checkpoint, manifest, split = pilot

    calibrate.main([str(checkpoint), str(manifest), "--split", str(split), "--unknown", "book"])

    out = capsys.readouterr().out
    assert "test signers: test1, test2" in out
    assert "known: 4 clips of 2 signs" in out          # 2 test signers x 2 known signs, nobody else
    assert "unknown: 2 clips of 1 signs (book)" in out
    assert "AUROC" in out
    assert sum(line.lstrip()[:4] in {"0.60", "0.88", "0.95"} for line in out.splitlines()) == 3


def test_without_unknown_the_untaught_sign_is_named_as_ignored_not_scored(pilot, capsys):
    checkpoint, manifest, split = pilot

    calibrate.main([str(checkpoint), str(manifest), "--split", str(split), "--on", "val"])

    out = capsys.readouterr().out
    assert "val signers: val1" in out and "known: 2 clips" in out
    assert "ignored, neither known nor listed in --unknown: book" in out
    assert "AUROC" not in out


def test_listing_a_trained_sign_as_unknown_is_refused(pilot):
    checkpoint, manifest, split = pilot
    with pytest.raises(SystemExit, match="trained on"):
        calibrate.main([str(checkpoint), str(manifest), "--split", str(split), "--unknown", "dog"])

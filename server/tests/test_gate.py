"""The safety-critical test: trains two models on synthetic data — one properly,
one deliberately sabotaged (starved of epochs) — and asserts B13's promotion gate
rejects the sabotaged one while not rejecting the properly-trained one on accuracy
grounds. This is the explicit ROADMAP.md Phase 1/4 exit criterion.

Uses `ml/make_synthetic.py`'s generator and calls the real `ml/train.py` /
`ml/evaluate.py` as subprocesses (per SYSTEM_DESIGN.md B13: the gate "reuses
ml/evaluate.py ... logic", not a reimplementation) — this test exercises the actual
training and evaluation code, not a mock of it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from server.gate import evaluate_gate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable  # this venv already has torch/numpy installed


def _run(*args: str, cwd: Path = REPO_ROOT) -> None:
    import os

    # ml/*.py scripts print Arabic labels; Windows' cp1252 console default crashes
    # on them unless forced to UTF-8 (see server/README.md's Windows notes).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [PYTHON, *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {args}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    _run(
        "ml/make_synthetic.py", "--out", str(out),
        "--signs", "10", "--signers", "8", "--reps", "5",
        "--signal", "0.8", "--style", "0.15", "--seed", "0",
    )
    return out


@pytest.fixture(scope="module")
def good_metrics(synthetic_data):
    checkpoint = synthetic_data / "good.pt"
    _run(
        "ml/train.py", str(synthetic_data / "manifest.json"),
        "--split", str(synthetic_data / "split.json"),
        "--out", str(checkpoint), "--epochs", "60", "--patience", "60", "--seed", "0",
    )
    report = synthetic_data / "good_report.json"
    _run(
        "ml/evaluate.py", str(checkpoint), str(synthetic_data / "manifest.json"),
        "--split", str(synthetic_data / "split.json"), "--report", str(report),
    )
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sabotaged_metrics(synthetic_data):
    """Same data, same split, same seed — the only difference is starving it of
    training (1 epoch instead of 60). This isolates "worse model" from "different
    data", which is the honest way to prove the gate reacts to model quality.
    """
    checkpoint = synthetic_data / "sabotaged.pt"
    _run(
        "ml/train.py", str(synthetic_data / "manifest.json"),
        "--split", str(synthetic_data / "split.json"),
        "--out", str(checkpoint), "--epochs", "1", "--patience", "1", "--seed", "0",
    )
    report = synthetic_data / "sabotaged_report.json"
    _run(
        "ml/evaluate.py", str(checkpoint), str(synthetic_data / "manifest.json"),
        "--split", str(synthetic_data / "split.json"), "--report", str(report),
    )
    return json.loads(report.read_text(encoding="utf-8"))


def test_gate_rejects_sabotaged_candidate(good_metrics, sabotaged_metrics):
    assert good_metrics["top1"] > sabotaged_metrics["top1"], (
        "fixture is not honest: the sabotaged model scored as well as the good one, "
        "so this test would pass for the wrong reason"
    )

    verdict = evaluate_gate(
        candidate=sabotaged_metrics, champion=good_metrics,
        candidate_latency_ms=50.0, latency_budget_ms=300.0, shadow_run_complete=True,
    )

    assert verdict.passed is False
    # It should fail specifically on an accuracy-relative criterion, not latency or
    # shadow-run — those are irrelevant to what was actually sabotaged here.
    assert verdict.criteria["top1_improves_or_ties"] is False


def test_gate_does_not_reject_good_candidate_on_accuracy(good_metrics):
    """The properly-trained model, compared against itself as champion (a tie),
    must not be rejected on any accuracy-relative ground — it regresses nothing.
    """
    verdict = evaluate_gate(
        candidate=good_metrics, champion=good_metrics,
        candidate_latency_ms=50.0, latency_budget_ms=300.0, shadow_run_complete=True,
    )

    assert verdict.criteria["top1_improves_or_ties"] is True
    assert verdict.criteria["no_signer_regression"] is True
    assert verdict.criteria["no_sign_regression"] is True
    assert verdict.criteria["no_zero_recall_class"] is True
    assert verdict.passed is True


def test_gate_rejects_on_latency_alone():
    metrics = {"top1": 0.9, "top5": 0.99, "per_signer": {"s1": 0.9}, "per_sign": {"a": 0.9}}
    verdict = evaluate_gate(
        candidate=metrics, champion=metrics,
        candidate_latency_ms=999.0, latency_budget_ms=300.0, shadow_run_complete=True,
    )
    assert verdict.passed is False
    assert verdict.criteria["latency_within_budget"] is False


def test_gate_fails_when_per_sign_metrics_are_missing():
    """A check nobody could perform must not read as a pass."""
    with_signs = {"top1": 0.9, "top5": 0.99, "per_signer": {"s1": 0.9}, "per_sign": {"a": 0.9}}
    without = {"top1": 0.9, "top5": 0.99, "per_signer": {"s1": 0.9}}
    verdict = evaluate_gate(
        candidate=without, champion=with_signs,
        candidate_latency_ms=50.0, latency_budget_ms=300.0, shadow_run_complete=True,
    )
    assert verdict.criteria["no_zero_recall_class"] is False
    assert verdict.passed is False


def test_gate_rejects_without_completed_shadow_run():
    metrics = {"top1": 0.9, "top5": 0.99, "per_signer": {"s1": 0.9}, "per_sign": {"a": 0.9}}
    verdict = evaluate_gate(
        candidate=metrics, champion=metrics,
        candidate_latency_ms=50.0, latency_budget_ms=300.0, shadow_run_complete=False,
    )
    assert verdict.passed is False
    assert verdict.criteria["shadow_run_complete"] is False

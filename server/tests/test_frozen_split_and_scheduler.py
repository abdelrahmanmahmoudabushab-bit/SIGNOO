"""DR-5/DR-7a: the benchmark signers are frozen once and never reassigned.
FR-L5: the 3-day cycle trains a candidate automatically, and only a candidate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from server import scheduler
from server.models import ModelVersion, ReviewItem, SessionUpload
from server.training import InsufficientDataError, export_validated_manifest


def _validated(db, signer, label="a", n=2, status="confirmed"):
    upload = SessionUpload(session_id=f"s-{signer}", kiosk_id="k", language="liu",
                           consent_retention=True, started_at="t", ended_at="t")
    db.add(upload)
    db.flush()
    for i in range(n):
        db.add(ReviewItem(
            session_upload_id=upload.id, utterance_id=f"{signer}-{label}-{i}", language="liu",
            signer_id=signer, landmarks=[[0.0] * 3], predicted_gloss=label, confidence=0.9,
            status=status, validated_label=label,
        ))
    db.commit()


def _split(path):
    return json.loads((path / "split.json").read_text(encoding="utf-8"))


def test_first_export_freezes_the_benchmark_and_later_signers_go_to_train(db_session, tmp_path):
    frozen = tmp_path / "frozen.json"
    for signer in ("S1", "S2", "S3", "S4"):
        _validated(db_session, signer)

    export_validated_manifest(db_session, "liu", tmp_path / "run1", frozen)
    assert _split(tmp_path / "run1") == {"val": ["S3"], "test": ["S4"]}

    _validated(db_session, "S5")            # sorts after S4: the old rule would have made it the test signer
    export_validated_manifest(db_session, "liu", tmp_path / "run2", frozen)
    assert _split(tmp_path / "run2") == {"val": ["S3"], "test": ["S4"]}
    train_signers = {m["signer_id"] for m in json.loads((tmp_path / "run2" / "manifest.json").read_text())
                     if m["signer_id"] not in ("S3", "S4")}
    assert train_signers == {"S1", "S2", "S5"}


def test_unknown_signer_rows_never_enter_the_export(db_session, tmp_path):
    for signer in ("S1", "S2", "S3"):
        _validated(db_session, signer)
    _validated(db_session, "unknown", label="b")

    export_validated_manifest(db_session, "liu", tmp_path / "run", tmp_path / "frozen.json")

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert "unknown" not in {m["signer_id"] for m in manifest}


def test_export_refuses_when_a_frozen_benchmark_signer_has_vanished(db_session, tmp_path):
    frozen = tmp_path / "frozen.json"
    for signer in ("S1", "S2", "S3", "S4"):
        _validated(db_session, signer)
    export_validated_manifest(db_session, "liu", tmp_path / "run1", frozen)

    for item in db_session.query(ReviewItem).filter_by(signer_id="S4"):
        item.status = "flagged"
    db_session.commit()

    with pytest.raises(InsufficientDataError, match="frozen benchmark"):
        export_validated_manifest(db_session, "liu", tmp_path / "run2", frozen)


# ── scheduler ───────────────────────────────────────────────────────────────────

def _version(db, created):
    db.add(ModelVersion(version=f"v-{created.timestamp()}", language="liu", code_commit="c",
                        data_snapshot="x", status="candidate", created_at=created))
    db.commit()


def test_cycle_is_due_only_after_the_interval(db_session):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    assert scheduler.cycle_due(db_session, "liu", now) is True            # nothing trained yet
    _version(db_session, datetime(2026, 10, 9, 12, 0))
    assert scheduler.cycle_due(db_session, "liu", now, interval_hours=72) is False
    assert scheduler.cycle_due(db_session, "liu", now + timedelta(days=3), interval_hours=72) is True


def test_scheduler_will_not_run_before_the_benchmark_is_frozen(db_session, monkeypatch):
    monkeypatch.setattr(scheduler, "load_frozen_split", lambda language: None)
    monkeypatch.setattr(scheduler, "train_candidate",
                        lambda *a, **k: pytest.fail("trained against an unfrozen benchmark"))
    assert scheduler.run_cycle_once(db_session, "liu") is None


def test_scheduler_trains_a_candidate_when_due_and_never_promotes(db_session, monkeypatch):
    calls = []

    def fake_train(db, language):
        calls.append(language)
        mv = ModelVersion(version="v-auto", language=language, code_commit="c",
                          data_snapshot="x", status="candidate")
        db.add(mv)
        db.commit()
        return mv

    monkeypatch.setattr(scheduler, "load_frozen_split", lambda language: {"val": ["S3"], "test": ["S4"]})
    monkeypatch.setattr(scheduler, "train_candidate", fake_train)

    result = scheduler.run_cycle_once(db_session, "liu")

    assert calls == ["liu"]
    assert result.status == "candidate"      # promotion stays a separate, gated, human step
    assert scheduler.run_cycle_once(db_session, "liu") is None      # just trained: not due again
    assert calls == ["liu"]


def test_scheduler_swallows_insufficient_data(db_session, monkeypatch):
    def not_enough(db, language):
        raise InsufficientDataError("only 1 signer")

    monkeypatch.setattr(scheduler, "load_frozen_split", lambda language: {"val": [], "test": []})
    monkeypatch.setattr(scheduler, "train_candidate", not_enough)
    assert scheduler.run_cycle_once(db_session, "liu") is None

"""What has to hold before a server runs on the public internet: a kiosk's credential
only works for that kiosk, declined consent is never stored, a rollback actually
withdraws a model, two training runs never overlap, the benchmark cannot be anchored on
anonymous visits, unreviewed clips do not accumulate forever, and each language's
knowledge base stays its own."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import server.auth as auth_module
import server.kb as kb_module
import server.training as training
from server import scheduler
from server.config import settings
from server.models import (
    KBEntry, Kiosk, ModelVersion, ReviewItem, Rollout, SessionUpload, TrainingAttempt, TrainingLock,
)
from server.retention import purge_unreviewed


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _version(db, name, language="liu", status="promoted", checkpoint=None):
    db.add(ModelVersion(version=name, language=language, code_commit="c", data_snapshot="x",
                        status=status, checkpoint_path=checkpoint))
    db.commit()


def _rollout(db, version, kiosks, created, status="active"):
    db.add(Rollout(model_version=version, target_kiosk_ids=kiosks, status=status, created_at=created))
    db.commit()


def _register(client, kiosk_id, **headers):
    return client.post("/fleet/kiosks/register", json={"kiosk_id": kiosk_id, "language": "liu"}, headers=headers)


# ── per-kiosk credentials ───────────────────────────────────────────────────────

@pytest.fixture()
def per_kiosk(monkeypatch):
    monkeypatch.setattr(auth_module, "settings",
                        replace(settings, kiosk_token="master-key", kiosk_token_mode="per_kiosk"))


def test_a_kiosk_token_only_works_for_its_own_kiosk(client, per_kiosk):
    token_1 = auth_module.kiosk_token_for("k1")
    token_2 = auth_module.kiosk_token_for("k2")
    assert token_1 != token_2

    assert _register(client, "k1", **{"X-Kiosk-Token": token_1}).status_code == 201
    assert _register(client, "k2", **{"X-Kiosk-Token": token_2}).status_code == 201

    # k2's credential, however valid for k2, says nothing about k1.
    stolen = client.post("/fleet/kiosks/k1/heartbeat", json={"throttled": False},
                         headers={"X-Kiosk-Token": token_2})
    assert stolen.status_code == 401
    assert client.post("/fleet/kiosks/k1/heartbeat", json={}, headers={"X-Kiosk-Token": token_1}).status_code == 200


def test_the_master_token_is_not_a_kiosk_credential_in_per_kiosk_mode(client, per_kiosk):
    assert _register(client, "k1", **{"X-Kiosk-Token": "master-key"}).status_code == 401
    assert _register(client, "k1").status_code == 401


def test_upload_for_another_kiosk_id_is_refused_in_per_kiosk_mode(client, per_kiosk):
    body = {"session_id": "s1", "kiosk_id": "k1", "language": "liu", "consent_retention": True,
            "started_at": "t", "ended_at": "t", "items": []}
    assert client.post("/ingest/sessions", json=body,
                       headers={"X-Kiosk-Token": auth_module.kiosk_token_for("k2")}).status_code == 401
    assert client.post("/ingest/sessions", json=body,
                       headers={"X-Kiosk-Token": auth_module.kiosk_token_for("k1")}).status_code == 201


# ── consent ─────────────────────────────────────────────────────────────────────

def test_server_refuses_a_session_whose_visitor_declined_retention(client, db_session):
    body = {"session_id": "declined", "kiosk_id": "k1", "language": "liu", "consent_retention": False,
            "started_at": "t", "ended_at": "t",
            "items": [{"utterance_id": "u1", "landmarks": [[0.0] * 203], "predicted_gloss": "marhaba",
                       "confidence": 0.9, "signer_id": "visitor-aaaa"}]}

    resp = client.post("/ingest/sessions", json=body)

    assert resp.status_code == 422
    assert db_session.query(SessionUpload).count() == 0
    assert db_session.query(ReviewItem).count() == 0


# ── rollback really withdraws a model ───────────────────────────────────────────

def _fleet_on(client, db, version):
    assert _register(client, "k1").status_code == 201
    kiosk = db.get(Kiosk, "k1")
    kiosk.current_model_version = version
    db.commit()


def test_rollback_falls_back_to_the_previous_promoted_model(client, db_session, make_user, tmp_path):
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    _version(db_session, "liu-1", checkpoint=str(old))
    _version(db_session, "liu-2", checkpoint=str(new))
    _rollout(db_session, "liu-1", ["k1"], datetime(2026, 10, 1))
    _rollout(db_session, "liu-2", ["k1"], datetime(2026, 10, 2))
    _fleet_on(client, db_session, "liu-2")
    assert client.get("/fleet/kiosks/k1/model", params={"language": "liu"}).headers["X-Model-Version"] == "liu-2"

    resp = client.post("/registry/liu-2/rollback", headers=_headers(make_user("e@x.dev", "engineer")))
    assert resp.status_code == 200

    served = client.get("/fleet/kiosks/k1/model", params={"language": "liu"})
    assert served.headers["X-Model-Version"] == "liu-1"
    db_session.expire_all()
    assert db_session.get(Kiosk, "k1").current_model_version == "liu-1"
    assert {r.model_version: r.status for r in db_session.query(Rollout)} == {"liu-1": "active", "liu-2": "rolled_back"}


def test_rollback_with_nothing_to_fall_back_to_serves_nothing(client, db_session, make_user, tmp_path):
    only = tmp_path / "only.pt"
    only.write_bytes(b"x")
    _version(db_session, "liu-1", checkpoint=str(only))
    _rollout(db_session, "liu-1", ["k1"], datetime(2026, 10, 1))
    _fleet_on(client, db_session, "liu-1")

    client.post("/registry/liu-1/rollback", headers=_headers(make_user("e@x.dev", "engineer")))

    assert client.get("/fleet/kiosks/k1/model", params={"language": "liu"}).status_code == 204
    db_session.expire_all()
    assert db_session.get(Kiosk, "k1").current_model_version is None


def test_a_rollout_completes_only_when_every_target_reports_the_version(client, db_session, tmp_path):
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x")
    _version(db_session, "liu-1", checkpoint=str(ckpt))
    assert _register(client, "k1").status_code == 201
    assert _register(client, "k2").status_code == 201
    _rollout(db_session, "liu-1", ["k1", "k2"], datetime(2026, 10, 1))

    client.post("/fleet/kiosks/k1/heartbeat", json={"model_versions": {"liu": "liu-1"}})
    db_session.expire_all()
    assert db_session.query(Rollout).one().status == "active"          # k2 has not reported

    client.post("/fleet/kiosks/k2/heartbeat", json={"model_versions": {"liu": "liu-1"}})
    db_session.expire_all()
    assert db_session.query(Rollout).one().status == "completed"
    # Completed is finished delivery, not withdrawal: the kiosk must still be handed it.
    assert client.get("/fleet/kiosks/k1/model", params={"language": "liu"}).status_code == 200


def test_download_hash_is_recorded_once_and_served(client, db_session, tmp_path):
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"model-bytes")
    _version(db_session, "liu-1", checkpoint=str(ckpt))
    assert _register(client, "k1").status_code == 201
    _rollout(db_session, "liu-1", ["k1"], datetime(2026, 10, 1))

    resp = client.get("/fleet/kiosks/k1/model", params={"language": "liu"})

    db_session.expire_all()
    assert resp.content == b"model-bytes"
    assert db_session.get(ModelVersion, "liu-1").checkpoint_sha256 == resp.headers["X-Model-Sha256"]


# ── one training run per language ───────────────────────────────────────────────

def test_second_training_run_for_a_language_is_refused_while_the_first_holds_the_lock(db_session):
    with training.training_lock(db_session, "liu"):
        with pytest.raises(training.TrainingInProgress):
            with training.training_lock(db_session, "liu"):
                pytest.fail("two runs of one language overlapped")
        with training.training_lock(db_session, "asl"):      # another language is independent
            pass
    with training.training_lock(db_session, "liu"):           # released afterwards
        pass
    assert db_session.query(TrainingLock).count() == 0


def test_a_lock_left_by_a_dead_process_is_taken_over(db_session):
    stale = datetime.now(timezone.utc) - timedelta(minutes=settings.training_lease_minutes + 5)
    db_session.add(TrainingLock(language="liu", holder="dead-process", acquired_at=stale.replace(tzinfo=None)))
    db_session.commit()

    with training.training_lock(db_session, "liu"):
        assert db_session.get(TrainingLock, "liu").holder != "dead-process"


def test_trigger_while_training_is_409_and_scheduler_skips_without_recording_a_failure(
        client, db_session, make_user, monkeypatch):
    db_session.add(TrainingLock(language="liu", holder="other-worker",
                                acquired_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    db_session.commit()

    resp = client.post("/training/trigger", json={"language": "liu"},
                       headers=_headers(make_user("e@x.dev", "engineer")))
    assert resp.status_code == 409
    assert "already in progress" in resp.json()["detail"]

    monkeypatch.setattr(scheduler, "load_frozen_split", lambda language: {"val": ["S3"], "test": ["S4"]})
    assert scheduler.run_cycle_once(db_session, "liu") is None
    assert db_session.query(TrainingAttempt).count() == 0     # a skipped tick is not a crashed run


# ── the benchmark ───────────────────────────────────────────────────────────────

def _validated(db, signer, language="liu", label="a", n=2):
    upload = SessionUpload(session_id=f"s-{language}-{signer}", kiosk_id="k", language=language,
                           consent_retention=True, started_at="t", ended_at="t")
    db.add(upload)
    db.flush()
    for i in range(n):
        db.add(ReviewItem(session_upload_id=upload.id, utterance_id=f"{signer}-{label}-{i}", language=language,
                          signer_id=signer, landmarks=[[0.0] * 3], predicted_gloss=label, confidence=0.9,
                          status="confirmed", validated_label=label))
    db.commit()


@pytest.fixture()
def frozen_path(tmp_path, monkeypatch):
    path = tmp_path / "frozen.json"
    monkeypatch.setattr(training, "settings", replace(settings, frozen_split_path=path))
    return path


def test_anonymous_visits_cannot_anchor_the_benchmark(db_session, tmp_path):
    for visitor in ("visitor-aaaa1111", "visitor-bbbb2222", "visitor-cccc3333", "visitor-dddd4444"):
        _validated(db_session, visitor)

    with pytest.raises(training.InsufficientDataError, match="enrolled"):
        training.export_validated_manifest(db_session, "liu", tmp_path / "run", tmp_path / "frozen.json")
    assert not (tmp_path / "frozen.json").exists()          # nothing was frozen by accident


def test_visitors_still_train_but_only_enrolled_people_become_the_benchmark(db_session, tmp_path):
    for signer in ("S1", "S2", "visitor-aaaa1111", "visitor-bbbb2222"):
        _validated(db_session, signer)

    training.export_validated_manifest(db_session, "liu", tmp_path / "run", tmp_path / "frozen.json")

    split = json.loads((tmp_path / "run" / "split.json").read_text())
    assert split == {"val": ["S1"], "test": ["S2"]}
    trained_on = {m["signer_id"] for m in json.loads((tmp_path / "run" / "manifest.json").read_text())}
    assert {"visitor-aaaa1111", "visitor-bbbb2222"} <= trained_on


def test_freeze_split_refuses_anonymous_and_overlapping_signers(tmp_path):
    with pytest.raises(ValueError, match="anonymous"):
        training.freeze_split("liu", ["visitor-aaaa"], ["S2"], tmp_path / "f.json")
    with pytest.raises(ValueError, match="both"):
        training.freeze_split("liu", ["S1"], ["S1"], tmp_path / "f.json")


def test_benchmark_endpoint_freezes_once_for_named_signers_with_reviewed_data(
        client, db_session, make_user, frozen_path):
    for signer in ("S1", "S2"):
        _validated(db_session, signer)
    headers = _headers(make_user("e@x.dev", "engineer"))

    nobody = client.post("/training/benchmark", headers=headers,
                         json={"language": "liu", "val": ["S1"], "test": ["S9"]})
    assert nobody.status_code == 422 and "S9" in nobody.json()["detail"]

    ok = client.post("/training/benchmark", headers=headers, json={"language": "liu", "val": ["S1"], "test": ["S2"]})
    assert ok.status_code == 201 and ok.json()["val"] == ["S1"]
    assert json.loads(frozen_path.read_text())["liu"] == {"val": ["S1"], "test": ["S2"]}

    again = client.post("/training/benchmark", headers=headers, json={"language": "liu", "val": ["S2"], "test": ["S1"]})
    assert again.status_code == 409                          # a benchmark that moves measures nothing


def test_only_engineers_can_freeze_a_benchmark(client, make_user, frozen_path):
    resp = client.post("/training/benchmark", headers=_headers(make_user("r@x.dev", "reviewer")),
                       json={"language": "liu", "val": ["S1"], "test": ["S2"]})
    assert resp.status_code == 403


# ── retention and erasure ───────────────────────────────────────────────────────

def _item(db, upload, name, status, age_days):
    created = (datetime.now(timezone.utc) - timedelta(days=age_days)).replace(tzinfo=None)
    db.add(ReviewItem(session_upload_id=upload.id, utterance_id=name, language="liu", signer_id="visitor-x",
                      landmarks=[[0.0] * 3], predicted_gloss="a", confidence=0.9, status=status,
                      validated_label="a" if status in ("confirmed", "corrected") else None, created_at=created))


def test_purge_deletes_only_old_unreviewed_clips(db_session):
    upload = SessionUpload(session_id="s", kiosk_id="k", language="liu", consent_retention=True,
                           started_at="t", ended_at="t")
    db_session.add(upload)
    db_session.flush()
    _item(db_session, upload, "old-pending", "pending", 40)
    _item(db_session, upload, "old-flagged", "flagged", 40)
    _item(db_session, upload, "new-pending", "pending", 2)
    _item(db_session, upload, "old-confirmed", "confirmed", 400)       # the training set stays
    _item(db_session, upload, "old-corrected", "corrected", 400)
    db_session.commit()

    assert purge_unreviewed(db_session, days=30) == 2
    assert {i.utterance_id for i in db_session.query(ReviewItem)} == {"new-pending", "old-confirmed", "old-corrected"}


def test_zero_days_disables_the_purge(db_session):
    upload = SessionUpload(session_id="s", kiosk_id="k", language="liu", consent_retention=True,
                           started_at="t", ended_at="t")
    db_session.add(upload)
    db_session.flush()
    _item(db_session, upload, "ancient", "pending", 1000)
    db_session.commit()

    assert purge_unreviewed(db_session, days=0) == 0
    assert db_session.query(ReviewItem).count() == 1


def test_admin_can_erase_a_session_and_nobody_else_can(client, db_session, make_user):
    upload = SessionUpload(session_id="to-erase", kiosk_id="k", language="liu", consent_retention=True,
                           started_at="t", ended_at="t")
    db_session.add(upload)
    db_session.flush()
    _item(db_session, upload, "u1", "confirmed", 1)
    _item(db_session, upload, "u2", "pending", 1)
    db_session.commit()

    assert client.delete("/review/sessions/to-erase", headers=_headers(make_user("e@x.dev", "engineer"))).status_code == 403
    assert client.delete("/review/sessions/nope", headers=_headers(make_user("a@x.dev", "admin"))).status_code == 404

    resp = client.delete("/review/sessions/to-erase", headers=_headers(make_user("a2@x.dev", "admin")))
    assert resp.status_code == 200 and resp.json()["items_erased"] == 2
    assert db_session.query(ReviewItem).count() == 0


# ── per-language knowledge base ─────────────────────────────────────────────────

@pytest.fixture()
def two_language_files(tmp_path, monkeypatch):
    """liu and asl each get their own knowledge-base file, in a scratch tree."""
    core = tmp_path / "core"
    core.mkdir()
    (tmp_path / "kiosk" / "data").mkdir(parents=True)
    (core / "languages.json").write_text(json.dumps([
        {"code": "asl", "vocab": "core/vocab_asl.json", "knowledge_base": "kiosk/data/kb_asl.json",
         "clip_library": "core/clips_asl.json"},
    ]), encoding="utf-8")
    redirected = dataclasses.replace(
        settings, languages_path=core / "languages.json", kiosk_kb_path=tmp_path / "kiosk" / "data" / "kb_liu.json")
    monkeypatch.setattr(kb_module, "settings", redirected)
    return tmp_path / "kiosk" / "data"


def _kb(client, headers, language, answer):
    return client.post("/kb", headers=headers, json={
        "language": language, "keywords": [answer], "answer_ar": answer, "answer_en": answer})


def test_each_language_writes_only_its_own_knowledge_base_file(client, make_user, two_language_files):
    headers = _headers(make_user("e@x.dev", "engineer"))
    assert _kb(client, headers, "liu", "liu-answer").status_code == 201
    assert _kb(client, headers, "asl", "asl-answer").status_code == 201

    liu_file = json.loads((two_language_files / "kb_liu.json").read_text(encoding="utf-8"))
    asl_file = json.loads((two_language_files / "kb_asl.json").read_text(encoding="utf-8"))
    assert [e["answer_en"] for e in liu_file] == ["liu-answer"]
    assert [e["answer_en"] for e in asl_file] == ["asl-answer"]
    assert [e["language"] for e in client.get("/kb", headers=headers, params={"language": "asl"}).json()] == ["asl"]


def test_an_entry_for_a_language_with_nowhere_to_publish_is_refused(client, make_user, db_session, two_language_files):
    resp = _kb(client, _headers(make_user("e@x.dev", "engineer")), "esl", "x")
    assert resp.status_code == 422
    assert db_session.query(KBEntry).count() == 0


def test_kiosk_receives_only_its_languages_answers_and_can_cache_them(client, db_session, make_user, two_language_files):
    headers = _headers(make_user("e@x.dev", "engineer"))
    _kb(client, headers, "liu", "liu-answer")
    _kb(client, headers, "asl", "asl-answer")
    assert _register(client, "k1").status_code == 201

    first = client.get("/fleet/kiosks/k1/kb", params={"language": "asl"})
    assert first.status_code == 200
    assert [e["answer_en"] for e in first.json()] == ["asl-answer"]

    cached = client.get("/fleet/kiosks/k1/kb", params={"language": "asl"},
                        headers={"If-None-Match": first.headers["ETag"]})
    assert cached.status_code == 304

    _kb(client, headers, "asl", "another-answer")
    changed = client.get("/fleet/kiosks/k1/kb", params={"language": "asl"},
                         headers={"If-None-Match": first.headers["ETag"]})
    assert changed.status_code == 200 and len(changed.json()) == 2

    # No entries for a language: 204, so the kiosk keeps what it has instead of blanking it.
    assert client.get("/fleet/kiosks/k1/kb", params={"language": "esl"}).status_code == 204
    assert client.get("/fleet/kiosks/ghost/kb", params={"language": "asl"}).status_code == 404

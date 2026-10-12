"""Every language has its own engine: one language's reviewed data, models and
rollouts never influence another's."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from server import training
from server.models import Kiosk, ModelVersion, ReviewItem, SessionUpload


def _auth(make_user, role="engineer"):
    return {"Authorization": f"Bearer {make_user(f'{role}@signoo.dev', role)}"}


def _items(db, language, gloss, n, *, status):
    upload = SessionUpload(session_id=f"s-{language}-{status}", kiosk_id="k", language=language,
                           consent_retention=True, started_at="t", ended_at="t")
    db.add(upload)
    db.flush()
    for i in range(n):
        db.add(ReviewItem(
            session_upload_id=upload.id, utterance_id=f"{language}-{status}-{i}", language=language,
            signer_id="S1", landmarks=[[0.0]], predicted_gloss=gloss, confidence=0.9, status=status,
            validated_label=gloss if status == "confirmed" else None,
        ))
    db.commit()


def _version(db, name, language, *, status="promoted", **extra):
    db.add(ModelVersion(version=name, language=language, code_commit="c", data_snapshot="x",
                        status=status, **extra))
    db.commit()


# ── review queue ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("params", [{"language": "asl"}, {}])
def test_review_priority_ignores_other_languages_validated_data(client, make_user, db_session, params):
    # "hello" is well covered in LIU and has never been validated in ASL.
    _items(db_session, "liu", "hello", 5, status="confirmed")
    _items(db_session, "asl", "hello", 1, status="pending")

    queue = client.get("/review/queue", params=params, headers=_auth(make_user, "reviewer")).json()

    asl = [i for i in queue if i["language"] == "asl"]
    assert len(asl) == 1
    assert asl[0]["priority_reason"] == "under_represented_sign"
    assert asl[0]["priority_score"] == pytest.approx(2.0 * 0.1 + 1.0)


# ── rollouts ────────────────────────────────────────────────────────────────────

def test_rollout_of_another_language_leaves_the_kiosks_own_version_alone(client, make_user, db_session):
    client.post("/fleet/kiosks/register", json={"kiosk_id": "k1", "language": "liu"})
    client.post("/fleet/kiosks/k1/heartbeat", json={"current_model_version": "liu-1"})
    _version(db_session, "asl-1", "asl")
    _version(db_session, "liu-2", "liu")
    headers = _auth(make_user)

    resp = client.post("/fleet/rollouts", headers=headers,
                       json={"model_version": "asl-1", "target_kiosk_ids": ["k1"]})
    assert resp.status_code == 201
    db_session.expire_all()
    assert db_session.get(Kiosk, "k1").current_model_version == "liu-1"

    client.post("/fleet/rollouts", headers=headers,
                json={"model_version": "liu-2", "target_kiosk_ids": ["k1"]})
    db_session.expire_all()
    assert db_session.get(Kiosk, "k1").current_model_version == "liu-2"


@pytest.mark.parametrize("status", ["candidate", "shadow", "rejected", "rolled_back"])
def test_only_a_promoted_version_can_be_rolled_out(client, make_user, db_session, status):
    client.post("/fleet/kiosks/register", json={"kiosk_id": "k1"})
    _version(db_session, "v1", "liu", status=status)

    resp = client.post("/fleet/rollouts", headers=_auth(make_user),
                       json={"model_version": "v1", "target_kiosk_ids": ["k1"]})

    assert resp.status_code == 409
    assert client.get("/fleet/kiosks/k1/model", params={"language": "liu"}).status_code == 204


# ── training ────────────────────────────────────────────────────────────────────

def test_trigger_refuses_a_champion_of_another_language_before_training(client, make_user, db_session, monkeypatch):
    _version(db_session, "asl-champ", "asl")
    monkeypatch.setattr(training, "export_validated_manifest",
                        lambda *a, **k: pytest.fail("started a run against another language's champion"))

    resp = client.post("/training/trigger", headers=_auth(make_user),
                       json={"language": "liu", "champion_version": "asl-champ"})

    assert resp.status_code == 409
    assert "'asl' model" in resp.json()["detail"]


def test_trigger_reports_an_unknown_champion(client, make_user):
    resp = client.post("/training/trigger", headers=_auth(make_user),
                       json={"language": "liu", "champion_version": "nope"})
    assert resp.status_code == 404


# ── training runs (console FR-W5) ───────────────────────────────────────────────

def test_training_runs_are_listed_per_language_with_real_curves(client, make_user, db_session, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.with_suffix(".history.json").write_text(json.dumps(
        [{"epoch": 0, "train_loss": 1.5, "train_acc": 0.3, "val_loss": 1.4, "val_acc": 0.4}]))
    _version(db_session, "liu-run", "liu", status="candidate", checkpoint_path=str(checkpoint),
             training_started_at=datetime(2026, 10, 1, 12), champion_version="liu-champ",
             created_at=datetime(2026, 10, 1, 13))
    _version(db_session, "asl-run", "asl", status="candidate",
             training_started_at=datetime(2026, 10, 2, 12))
    _version(db_session, "imported", "liu")          # registered without a run: not a run
    headers = _auth(make_user, "operator")

    assert client.get("/training/runs").status_code == 401
    everything = client.get("/training/runs", headers=headers).json()
    liu = client.get("/training/runs", params={"language": "liu"}, headers=headers).json()

    assert [r["run_id"] for r in everything] == ["asl-run", "liu-run"]
    assert everything[0]["curves"] == [] and everything[0]["champion_version"] is None
    assert liu == [{
        "run_id": "liu-run", "language": "liu", "status": "completed",
        "started_at": "2026-10-01T12:00:00+00:00", "finished_at": "2026-10-01T13:00:00+00:00",
        "candidate_version": "liu-run", "champion_version": "liu-champ",
        "curves": [{"epoch": 0, "train_loss": 1.5, "val_accuracy": 0.4}],
    }]

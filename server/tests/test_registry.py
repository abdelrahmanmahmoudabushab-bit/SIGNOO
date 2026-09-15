"""B13: registry sequencing (candidate -> shadow -> promoted/rejected) and the
"champion stays on rejection" rule (SYSTEM_DESIGN.md §6)."""
from __future__ import annotations

from server.models import ModelVersion


def _make_candidate(db_session, version, *, top1, gate_passed):
    mv = ModelVersion(
        version=version, language="liu", code_commit="abc123",
        data_snapshot="manifest-sha256:deadbeef",
        metrics={"top1": top1, "top5": top1, "per_signer": {"S01": top1}, "per_sign": {"marhaba": top1}},
        gate={"passed": gate_passed, "criteria": {}, "details": {}},
        status="candidate", checkpoint_path=None,
    )
    db_session.add(mv)
    db_session.commit()
    return mv


def test_promote_requires_shadow_complete_first(client, make_user, db_session):
    _make_candidate(db_session, "v-good", top1=0.9, gate_passed=True)
    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/registry/v-good/promote", headers=headers)
    assert resp.status_code == 409  # still just "candidate", not "shadow"


def test_shadow_complete_then_promote_happy_path(client, make_user, db_session):
    _make_candidate(db_session, "v-good", top1=0.9, gate_passed=True)
    # Give it a recorded latency measurement so complete_shadow can re-read it.
    mv = db_session.get(ModelVersion, "v-good")
    mv.gate = {"passed": True, "criteria": {}, "details": {}, "measurements": {"candidate_latency_ms": 50.0}}
    db_session.commit()

    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/registry/v-good/shadow-complete", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "shadow"

    resp = client.post("/registry/v-good/promote", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "promoted"


def test_rejected_candidate_cannot_be_promoted_champion_stays(client, make_user, db_session):
    _make_candidate(db_session, "v-champion", top1=0.9, gate_passed=True)
    champion = db_session.get(ModelVersion, "v-champion")
    champion.status = "promoted"
    db_session.commit()

    # A sabotaged candidate that regresses badly against the champion.
    _make_candidate(db_session, "v-bad", top1=0.2, gate_passed=False)
    bad = db_session.get(ModelVersion, "v-bad")
    bad.gate = {
        "passed": False, "criteria": {"top1_improves_or_ties": False}, "details": {},
        "measurements": {"candidate_latency_ms": 50.0},
    }
    db_session.commit()

    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/registry/v-bad/shadow-complete", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"

    resp = client.post("/registry/v-bad/promote", headers=headers)
    assert resp.status_code == 409

    # Champion is untouched.
    champions = client.get("/registry?language=liu", headers=headers).json()
    assert any(m["version"] == "v-champion" and m["status"] == "promoted" for m in champions)


def test_training_trigger_reports_insufficient_data(client, make_user):
    token = make_user("eng@signoo.dev", "engineer")
    resp = client.post(
        "/training/trigger", json={"language": "liu", "epochs": 5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422
    assert "signer" in resp.json()["detail"]

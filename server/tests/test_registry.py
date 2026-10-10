"""B13: registry sequencing (candidate -> shadow -> promoted/rejected) and the
"champion stays on rejection" rule (SYSTEM_DESIGN.md §6). The shadow-evaluation cases
live in test_shadow.py, where they can seed reviewed live items and a checkpoint."""
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


def test_shadow_complete_refuses_without_live_data(client, make_user, db_session):
    mv = _make_candidate(db_session, "v-good", top1=0.9, gate_passed=True)
    mv.gate = {"passed": True, "criteria": {}, "details": {}, "measurements": {"candidate_latency_ms": 50.0}}
    db_session.commit()
    token = make_user("eng@signoo.dev", "engineer")

    resp = client.post("/registry/v-good/shadow-complete", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 409
    assert "reviewed live items" in resp.json()["detail"]


def test_training_trigger_reports_insufficient_data(client, make_user):
    token = make_user("eng@signoo.dev", "engineer")
    resp = client.post(
        "/training/trigger", json={"language": "liu", "epochs": 5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422
    assert "signer" in resp.json()["detail"]

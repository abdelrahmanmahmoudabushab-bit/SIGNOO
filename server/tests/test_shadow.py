"""FR-L7: the shadow evaluation measures a candidate on reviewed live items newer than
itself, and the registry's gate rests on that measurement instead of a button press."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

from server.models import ModelVersion, ReviewItem, SessionUpload
from server.shadow import judge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CREATED = datetime(2026, 10, 1, 12, 0, 0)
REVIEWED_AFTER = datetime(2026, 10, 2, 12, 0, 0)
REVIEWED_BEFORE = datetime(2026, 9, 30, 12, 0, 0)


def _checkpoint(path: Path, always: str) -> str:
    """A real SignTransformer whose head is rigged to always answer `always`, so the
    expected accuracy is exact and does not depend on training having converged."""
    sys.path.insert(0, str(REPO_ROOT))
    from ml.model import SignTransformer

    model = SignTransformer(input_dim=406, num_classes=2)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.copy_(torch.tensor([10.0, 0.0] if always == "a" else [0.0, 10.0]))
    torch.save({"model": model.state_dict(), "labels": {"a": 0, "b": 1}, "input_dim": 406}, path)
    return str(path)


def _metrics(top1=0.9):
    return {"top1": top1, "top5": top1, "per_signer": {"S01": top1}, "per_sign": {"a": top1, "b": top1}}


def _version(db, name, *, status, checkpoint=None, created=CREATED):
    mv = ModelVersion(
        version=name, language="liu", code_commit="abc", data_snapshot="manifest-sha256:x",
        metrics=_metrics(), status=status, checkpoint_path=checkpoint, created_at=created,
        gate={"passed": True, "criteria": {}, "details": {}, "measurements": {"candidate_latency_ms": 50.0}},
    )
    db.add(mv)
    db.commit()
    return mv


def _items(db, n, *, truth, champion_said, reviewed_at=REVIEWED_AFTER):
    upload = SessionUpload(session_id="s", kiosk_id="k", language="liu", consent_retention=True,
                           started_at="t", ended_at="t")
    db.add(upload)
    db.flush()
    for i in range(n):
        db.add(ReviewItem(
            session_upload_id=upload.id, utterance_id=f"u{reviewed_at.day}-{truth}-{i}", language="liu",
            signer_id=f"visitor-{i}", landmarks=[[0.0] * 203 for _ in range(64)],
            predicted_gloss=champion_said, confidence=0.9, status="confirmed",
            validated_label=truth, reviewed_at=reviewed_at,
        ))
    db.commit()


# ── the decision rule, no model or database ─────────────────────────────────────

def test_judge_refuses_to_decide_without_enough_items():
    result = judge(5, None, None, min_items=30, tolerance=0.005)
    assert (result.enough_data, result.ok) == (False, False)
    assert "5 of 30" in result.reason


def test_judge_passes_a_candidate_that_holds_up():
    assert judge(40, 0.90, 0.90, 30, 0.005).ok
    assert judge(40, 0.896, 0.90, 30, 0.005).ok          # inside tolerance


def test_judge_fails_a_candidate_that_falls_below_the_champion():
    assert not judge(40, 0.80, 0.90, 30, 0.005).ok


# ── through the registry API ────────────────────────────────────────────────────

def test_better_candidate_passes_shadow_and_can_be_promoted(client, make_user, db_session, tmp_path):
    champion = _version(db_session, "v-champion", status="promoted")
    champion.promoted_at = CREATED
    _version(db_session, "v-new", status="candidate", checkpoint=_checkpoint(tmp_path / "a.pt", "a"))
    _items(db_session, 30, truth="a", champion_said="b")        # champion wrong, candidate right
    headers = {"Authorization": f"Bearer {make_user('eng@signoo.dev', 'engineer')}"}

    resp = client.post("/registry/v-new/shadow-complete", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "shadow"
    assert body["gate"]["measurements"]["shadow_candidate_accuracy"] == 1.0
    assert body["gate"]["measurements"]["shadow_champion_accuracy"] == 0.0
    assert client.post("/registry/v-new/promote", headers=headers).json()["status"] == "promoted"


def test_worse_candidate_is_rejected_and_the_champion_stays(client, make_user, db_session, tmp_path):
    champion = _version(db_session, "v-champion", status="promoted")
    champion.promoted_at = CREATED
    _version(db_session, "v-bad", status="candidate", checkpoint=_checkpoint(tmp_path / "a.pt", "a"))
    _items(db_session, 30, truth="b", champion_said="b")        # champion right, candidate wrong
    headers = {"Authorization": f"Bearer {make_user('eng@signoo.dev', 'engineer')}"}

    resp = client.post("/registry/v-bad/shadow-complete", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert client.post("/registry/v-bad/promote", headers=headers).status_code == 409
    listing = client.get("/registry?language=liu", headers=headers).json()
    assert any(m["version"] == "v-champion" and m["status"] == "promoted" for m in listing)


def test_items_reviewed_before_the_candidate_existed_do_not_count(client, make_user, db_session, tmp_path):
    _version(db_session, "v-new", status="candidate", checkpoint=_checkpoint(tmp_path / "a.pt", "a"))
    _items(db_session, 30, truth="a", champion_said="a", reviewed_at=REVIEWED_BEFORE)
    headers = {"Authorization": f"Bearer {make_user('eng@signoo.dev', 'engineer')}"}

    resp = client.post("/registry/v-new/shadow-complete", headers=headers)

    assert resp.status_code == 409           # they may be in its training set
    assert "0 of 30" in resp.json()["detail"]

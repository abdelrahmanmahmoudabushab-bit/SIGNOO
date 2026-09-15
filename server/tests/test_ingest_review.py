"""B11: ingest -> review queue ordering (FR-L4) -> decision (FR-L3)."""
from __future__ import annotations


def _upload(client, *, confidence, correction=None, predicted="marhaba", signer="S01",
            session_id="sess-1", utterance_id="u-1"):
    payload = {
        "session_id": session_id, "kiosk_id": "kiosk-dev-01", "language": "liu",
        "consent_retention": True, "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:01:00Z",
        "items": [{
            "utterance_id": utterance_id, "landmarks": [[0.0] * 4], "signer_id": signer,
            "predicted_gloss": predicted, "confidence": confidence, "correction": correction,
        }],
    }
    resp = client.post("/ingest/sessions", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_upload_lands_in_review_queue_not_training(client, make_user):
    _upload(client, confidence=0.9, utterance_id="u-1")
    token = make_user("rev@signoo.dev", "reviewer")
    resp = client.get("/review/queue", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["status"] == "pending"
    assert items[0]["validated_label"] is None


def test_queue_orders_low_confidence_and_corrected_first(client, make_user):
    _upload(client, confidence=0.95, utterance_id="u-high-confidence", session_id="s1")
    _upload(client, confidence=0.3, utterance_id="u-low-confidence", session_id="s2")
    _upload(client, confidence=0.9, correction="shukran", utterance_id="u-corrected", session_id="s3")

    token = make_user("rev@signoo.dev", "reviewer")
    resp = client.get("/review/queue", headers={"Authorization": f"Bearer {token}"})
    ordered_ids = [item["utterance_id"] for item in resp.json()]

    # The high-confidence, uncorrected item is the least valuable to review and
    # must not be first.
    assert ordered_ids[0] != "u-high-confidence"
    assert ordered_ids[-1] == "u-high-confidence"


def test_decision_confirm_sets_validated_label(client, make_user):
    _upload(client, confidence=0.4, utterance_id="u-1")
    token = make_user("rev@signoo.dev", "reviewer")
    headers = {"Authorization": f"Bearer {token}"}

    queue = client.get("/review/queue", headers=headers).json()
    item_id = queue[0]["id"]

    resp = client.post(f"/review/{item_id}/decision", json={"decision": "confirm"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confirmed"
    assert body["validated_label"] == "marhaba"

    # Confirmed items leave the pending queue.
    remaining = client.get("/review/queue", headers=headers).json()
    assert remaining == []


def test_decision_correct_requires_a_label(client, make_user):
    _upload(client, confidence=0.4, utterance_id="u-1")
    token = make_user("rev@signoo.dev", "reviewer")
    headers = {"Authorization": f"Bearer {token}"}
    item_id = client.get("/review/queue", headers=headers).json()[0]["id"]

    resp = client.post(f"/review/{item_id}/decision", json={"decision": "correct"}, headers=headers)
    assert resp.status_code == 422

    resp = client.post(
        f"/review/{item_id}/decision",
        json={"decision": "correct", "validated_label": "shukran"}, headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["validated_label"] == "shukran"
    assert resp.json()["status"] == "corrected"


def test_operator_cannot_make_review_decisions(client, make_user):
    _upload(client, confidence=0.4, utterance_id="u-1")
    reviewer_token = make_user("rev@signoo.dev", "reviewer")
    operator_token = make_user("op@signoo.dev", "operator")
    item_id = client.get(
        "/review/queue", headers={"Authorization": f"Bearer {reviewer_token}"}
    ).json()[0]["id"]

    resp = client.post(
        f"/review/{item_id}/decision", json={"decision": "confirm"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 403

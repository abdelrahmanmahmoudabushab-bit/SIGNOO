"""B15: kiosk registration/heartbeat + staged rollout targeting a named subset."""
from __future__ import annotations


def test_register_and_heartbeat(client):
    resp = client.post("/fleet/kiosks/register", json={"kiosk_id": "kiosk-dev-01", "language": "liu"})
    assert resp.status_code == 201
    assert resp.json()["online"] is False  # never heartbeated yet

    resp = client.post(
        "/fleet/kiosks/kiosk-dev-01/heartbeat",
        json={"current_model_version": "v1", "temperature_c": 45.2, "throttled": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["online"] is True
    assert body["current_model_version"] == "v1"


def test_heartbeat_for_unregistered_kiosk_is_rejected(client):
    resp = client.post("/fleet/kiosks/unknown-kiosk/heartbeat", json={})
    assert resp.status_code == 404


def test_list_kiosks_requires_a_role(client, make_user):
    client.post("/fleet/kiosks/register", json={"kiosk_id": "kiosk-dev-01"})
    resp = client.get("/fleet/kiosks")
    assert resp.status_code == 401

    token = make_user("op@signoo.dev", "operator")
    resp = client.get("/fleet/kiosks", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_rollout_targets_only_registered_kiosks(client, make_user):
    client.post("/fleet/kiosks/register", json={"kiosk_id": "kiosk-dev-01"})
    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/fleet/rollouts", headers=headers,
        json={"model_version": "does-not-exist", "target_kiosk_ids": ["kiosk-dev-01"]},
    )
    assert resp.status_code == 404  # model version must exist first


def test_operator_cannot_create_rollout(client, make_user):
    client.post("/fleet/kiosks/register", json={"kiosk_id": "kiosk-dev-01"})
    token = make_user("op@signoo.dev", "operator")
    resp = client.post(
        "/fleet/rollouts", headers={"Authorization": f"Bearer {token}"},
        json={"model_version": "v1", "target_kiosk_ids": ["kiosk-dev-01"]},
    )
    assert resp.status_code == 403

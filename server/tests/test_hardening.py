"""Production hardening: insecure defaults stop the server from starting, kiosk-facing
endpoints need the shared kiosk token, and weak passwords are refused."""
from __future__ import annotations

from dataclasses import replace

import pytest

import server.auth as auth_module
from server.config import DEV_JWT_SECRET, settings, startup_problems

SESSION = {
    "session_id": "s1", "kiosk_id": "k1", "language": "liu", "consent_retention": True,
    "started_at": "t", "ended_at": "t",
    "items": [{"utterance_id": "u1", "landmarks": [[0.0] * 203], "predicted_gloss": "marhaba",
               "confidence": 0.9, "signer_id": "visitor-1"}],
}


def test_production_refuses_the_dev_secret_and_a_blank_kiosk_token():
    problems = startup_problems(replace(settings, environment="production",
                                        jwt_secret=DEV_JWT_SECRET, kiosk_token=""))
    assert len(problems) == 2


def test_production_refuses_a_short_secret():
    problems = startup_problems(replace(settings, environment="production",
                                        jwt_secret="too-short", kiosk_token="x"))
    assert any("JWT_SECRET" in p for p in problems)


def test_production_starts_with_real_secrets():
    ok = replace(settings, environment="production", jwt_secret="x" * 40, kiosk_token="kiosk-secret")
    assert startup_problems(ok) == []


def test_dev_environment_is_not_blocked():
    assert startup_problems(replace(settings, environment="dev", jwt_secret=DEV_JWT_SECRET, kiosk_token="")) == []


@pytest.fixture()
def with_kiosk_token(monkeypatch):
    monkeypatch.setattr(auth_module, "settings", replace(settings, kiosk_token="kiosk-secret"))


def test_ingest_rejects_requests_without_the_kiosk_token(client, with_kiosk_token):
    assert client.post("/ingest/sessions", json=SESSION).status_code == 401
    wrong = client.post("/ingest/sessions", json=SESSION, headers={"X-Kiosk-Token": "nope"})
    assert wrong.status_code == 401


def test_ingest_accepts_the_right_kiosk_token(client, with_kiosk_token):
    resp = client.post("/ingest/sessions", json=SESSION, headers={"X-Kiosk-Token": "kiosk-secret"})
    assert resp.status_code == 201


def test_fleet_register_and_heartbeat_need_the_token_too(client, with_kiosk_token):
    assert client.post("/fleet/kiosks/register", json={"kiosk_id": "k1"}).status_code == 401
    good = {"X-Kiosk-Token": "kiosk-secret"}
    assert client.post("/fleet/kiosks/register", json={"kiosk_id": "k1"}, headers=good).status_code == 201
    assert client.post("/fleet/kiosks/k1/heartbeat", json={}).status_code == 401
    assert client.post("/fleet/kiosks/k1/heartbeat", json={}, headers=good).status_code == 200


def test_without_a_configured_token_dev_requests_still_work(client):
    assert client.post("/ingest/sessions", json=SESSION).status_code == 201


def test_weak_passwords_are_refused(client, make_user):
    admin = {"Authorization": f"Bearer {make_user('admin@signoo.dev', 'admin')}"}
    short = client.post("/auth/users", headers=admin,
                        json={"email": "a@signoo.dev", "password": "short", "role": "reviewer"})
    assert short.status_code == 422
    fine = client.post("/auth/users", headers=admin,
                       json={"email": "b@signoo.dev", "password": "a-long-enough-one", "role": "reviewer"})
    assert fine.status_code == 201

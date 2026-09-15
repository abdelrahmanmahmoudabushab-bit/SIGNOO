from __future__ import annotations


def test_login_succeeds_with_correct_password(client, make_user):
    make_user("reviewer@signoo.dev", "reviewer", password="s3cret!")
    resp = client.post("/auth/login", json={"email": "reviewer@signoo.dev", "password": "s3cret!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "reviewer"
    assert body["access_token"]


def test_login_rejects_wrong_password(client, make_user):
    make_user("reviewer@signoo.dev", "reviewer", password="s3cret!")
    resp = client.post("/auth/login", json={"email": "reviewer@signoo.dev", "password": "wrong"})
    assert resp.status_code == 401


def test_unauthenticated_request_is_rejected(client):
    resp = client.get("/review/queue")
    assert resp.status_code == 401


def test_wrong_role_is_forbidden(client, make_user):
    token = make_user("op@signoo.dev", "operator")
    resp = client.get("/review/queue", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_correct_role_is_allowed(client, make_user):
    token = make_user("rev@signoo.dev", "reviewer")
    resp = client.get("/review/queue", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == []

"""B15 model distribution: a kiosk fetches, per language, the checkpoint of the newest
active rollout that targets it. One language's rollout never answers for another."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime

import pytest

import server.auth as auth_module
from server.config import settings
from server.models import ModelVersion, Rollout

PAYLOAD = b"liu-checkpoint-bytes"


def _version(db, name, language="liu", *, status="promoted", checkpoint=None):
    db.add(ModelVersion(version=name, language=language, code_commit="c", data_snapshot="x",
                        status=status, checkpoint_path=checkpoint))
    db.commit()


def _rollout(db, version, kiosks, created, status="active"):
    db.add(Rollout(model_version=version, target_kiosk_ids=kiosks, status=status, created_at=created))
    db.commit()


@pytest.fixture()
def kiosk(client):
    assert client.post("/fleet/kiosks/register", json={"kiosk_id": "k1", "language": "liu"}).status_code == 201
    return "k1"


@pytest.fixture()
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(PAYLOAD)
    return str(path)


def test_unregistered_kiosk_is_404(client):
    assert client.get("/fleet/kiosks/nope/model", params={"language": "liu"}).status_code == 404


def test_language_is_required(client, kiosk):
    assert client.get(f"/fleet/kiosks/{kiosk}/model").status_code == 422


def test_no_rollout_for_the_language_is_204(client, kiosk, db_session, checkpoint):
    _version(db_session, "asl-1", "asl", checkpoint=checkpoint)
    _rollout(db_session, "asl-1", [kiosk], datetime(2026, 10, 1))

    resp = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "liu"})

    assert resp.status_code == 204
    assert resp.content == b""


def test_serves_the_checkpoint_with_version_and_hash(client, kiosk, db_session, checkpoint):
    _version(db_session, "liu-1", checkpoint=checkpoint)
    _rollout(db_session, "liu-1", [kiosk], datetime(2026, 10, 1))

    resp = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "liu"})

    assert resp.status_code == 200
    assert resp.content == PAYLOAD
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["X-Model-Version"] == "liu-1"
    assert resp.headers["X-Model-Sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert resp.headers["ETag"] == '"liu-1"'


def test_matching_if_none_match_is_304(client, kiosk, db_session, checkpoint):
    _version(db_session, "liu-1", checkpoint=checkpoint)
    _rollout(db_session, "liu-1", [kiosk], datetime(2026, 10, 1))
    url = f"/fleet/kiosks/{kiosk}/model"

    assert client.get(url, params={"language": "liu"}, headers={"If-None-Match": '"liu-1"'}).status_code == 304
    stale = client.get(url, params={"language": "liu"}, headers={"If-None-Match": '"liu-0"'})
    assert stale.status_code == 200


def test_newest_active_rollout_of_that_language_wins(client, kiosk, db_session, checkpoint, tmp_path):
    newer = tmp_path / "newer.pt"
    newer.write_bytes(b"newer")
    _version(db_session, "liu-1", checkpoint=checkpoint)
    _version(db_session, "liu-2", checkpoint=str(newer))
    _version(db_session, "liu-3", checkpoint=checkpoint)
    _version(db_session, "asl-1", "asl", checkpoint=checkpoint)
    _rollout(db_session, "liu-2", [kiosk], datetime(2026, 10, 2))
    _rollout(db_session, "liu-1", [kiosk], datetime(2026, 10, 1))
    _rollout(db_session, "liu-3", [kiosk], datetime(2026, 10, 3), status="rolled_back")   # not active
    _rollout(db_session, "liu-3", ["someone-else"], datetime(2026, 10, 4))                # other kiosk
    _rollout(db_session, "asl-1", [kiosk], datetime(2026, 10, 5))                         # other language

    liu = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "liu"})
    asl = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "asl"})

    assert (liu.headers["X-Model-Version"], liu.content) == ("liu-2", b"newer")
    assert asl.headers["X-Model-Version"] == "asl-1"


def test_missing_checkpoint_is_409(client, kiosk, db_session, tmp_path):
    _version(db_session, "liu-none", checkpoint=None)
    _rollout(db_session, "liu-none", [kiosk], datetime(2026, 10, 1))
    resp = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "liu"})
    assert resp.status_code == 409
    assert "no checkpoint" in resp.json()["detail"]

    _version(db_session, "liu-gone", checkpoint=str(tmp_path / "deleted.pt"))
    _rollout(db_session, "liu-gone", [kiosk], datetime(2026, 10, 2))
    resp = client.get(f"/fleet/kiosks/{kiosk}/model", params={"language": "liu"})
    assert resp.status_code == 409
    assert "missing" in resp.json()["detail"]


def test_model_download_needs_the_kiosk_token(client, kiosk, db_session, checkpoint, monkeypatch):
    _version(db_session, "liu-1", checkpoint=checkpoint)
    _rollout(db_session, "liu-1", [kiosk], datetime(2026, 10, 1))
    monkeypatch.setattr(auth_module, "settings", replace(settings, kiosk_token="kiosk-secret"))
    url = f"/fleet/kiosks/{kiosk}/model"

    assert client.get(url, params={"language": "liu"}).status_code == 401
    ok = client.get(url, params={"language": "liu"}, headers={"X-Kiosk-Token": "kiosk-secret"})
    assert ok.status_code == 200


def test_heartbeat_stores_per_language_versions_and_keeps_them_when_omitted(client, kiosk):
    assert client.post(f"/fleet/kiosks/{kiosk}/heartbeat", json={}).json()["model_versions"] == {}

    sent = {"liu": "liu-1", "asl": "asl-4"}
    body = client.post(f"/fleet/kiosks/{kiosk}/heartbeat",
                       json={"model_versions": sent, "current_model_version": "liu-1"}).json()
    assert body["model_versions"] == sent
    assert body["current_model_version"] == "liu-1"

    assert client.post(f"/fleet/kiosks/{kiosk}/heartbeat", json={"uptime_s": 5}).json()["model_versions"] == sent
    replaced = client.post(f"/fleet/kiosks/{kiosk}/heartbeat", json={"model_versions": {"liu": "liu-2"}}).json()
    assert replaced["model_versions"] == {"liu": "liu-2"}

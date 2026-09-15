"""B14: KB CRUD, role gating, and the unrenderable-word warning (FR-W7).

`sync_kiosk_kb_file` is redirected to a tmp path for every test in this module so
the suite never writes into the real `kiosk/data/knowledge_base.json` — that file
is shared with the concurrently-developed kiosk/ agent's own working tree.
"""
from __future__ import annotations

import dataclasses

import pytest

from server.kb import unrenderable_words


@pytest.fixture(autouse=True)
def _redirect_kiosk_kb_file(tmp_path, monkeypatch):
    import server.config
    import server.kb

    redirected = dataclasses.replace(server.config.settings, kiosk_kb_path=tmp_path / "kb.json")
    monkeypatch.setattr(server.kb, "settings", redirected)


def test_unrenderable_words_flags_every_vocab_word_right_now():
    # core/clip_library.json currently has clip_path: null for every word — this is
    # the documented normal state (API_CONTRACTS.md §2), not an error.
    flagged = unrenderable_words("مرحبا", "hello", ["hello", "مرحبا"])
    assert "marhaba" in flagged


def test_unrenderable_words_ignores_words_outside_the_vocab():
    flagged = unrenderable_words("جملة عشوائية بدون كلمات مفردة", "some random sentence", [])
    assert flagged == []


def test_create_and_list_kb_entry(client, make_user):
    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/kb", headers=headers,
        json={"site_id": "default", "keywords": ["hours", "opening"],
              "answer_ar": "الافتتاح الساعة التاسعة", "answer_en": "Opening hours start at nine"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["unrenderable_words"] == []  # none of these words are in core/vocab.json

    listed = client.get("/kb", headers=headers).json()
    assert len(listed) == 1


def test_operator_can_read_but_not_write_kb(client, make_user):
    operator_token = make_user("op@signoo.dev", "operator")
    resp = client.get("/kb", headers={"Authorization": f"Bearer {operator_token}"})
    assert resp.status_code == 200

    resp = client.post(
        "/kb", headers={"Authorization": f"Bearer {operator_token}"},
        json={"keywords": [], "answer_ar": "x", "answer_en": "y"},
    )
    assert resp.status_code == 403


def test_kb_entry_flags_marhaba_as_unrenderable(client, make_user):
    token = make_user("eng@signoo.dev", "engineer")
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/kb", headers=headers,
        json={"keywords": ["مرحبا"], "answer_ar": "مرحبا بك", "answer_en": "hello there"},
    )
    assert resp.status_code == 201
    assert "marhaba" in resp.json()["unrenderable_words"]

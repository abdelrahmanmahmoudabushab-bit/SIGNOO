"""B7 Dialogue: retrieval floor, refusal, and the local-fallback path (FR-A1).

Cloud is disabled in these tests (`cloud_enabled=False`) so they never hit the
network — the point being tested is the retrieval/refusal/fallback logic itself,
not the Claude API call, which `dialogue.py`'s `_try_cloud` already isolates
behind a broad except so a real network failure degrades the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiosk.assembly import Vocabulary
from kiosk.config import CONFIG
from kiosk.dialogue import Dialogue, KnowledgeBase


@pytest.fixture
def kb_path(tmp_path) -> Path:
    entries = [{
        "id": "kb_001", "site_id": "default",
        "keywords": ["hours", "افتتاح", "opening"],
        "answer_ar": "نفتح من التاسعة حتى الخامسة.",
        "answer_en": "We are open from 9 to 5.",
        "updated_at": "2026-09-04T00:00:00Z",
    }]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def _dialogue(kb_path: Path) -> Dialogue:
    vocab = Vocabulary(CONFIG.vocab_path)
    return Dialogue(
        KnowledgeBase(kb_path), similarity_floor=0.2, claude_model="unused",
        claude_timeout_s=1.0, cloud_enabled=False, vocab=vocab, clip_library={},
    )


def test_in_scope_question_gets_local_fallback_answer(kb_path):
    dialogue = _dialogue(kb_path)
    answer = dialogue.answer("what are your opening hours")

    assert answer.refused is False
    assert answer.source == "local_fallback"
    assert answer.en == "We are open from 9 to 5."


def test_out_of_scope_question_is_refused_not_guessed(kb_path):
    dialogue = _dialogue(kb_path)
    answer = dialogue.answer("what is the meaning of life")

    assert answer.refused is True
    assert "staff" in answer.en.lower() or "موظف" in answer.ar


def test_refusal_never_calls_the_cloud_path(kb_path, monkeypatch):
    dialogue = _dialogue(kb_path)
    dialogue.cloud_enabled = True   # even if cloud were enabled...

    called = {"hit": False}

    def _fake_cloud(*a, **k):
        called["hit"] = True
        return None

    monkeypatch.setattr(dialogue, "_try_cloud", _fake_cloud)
    dialogue.answer("completely unrelated nonsense question")
    assert called["hit"] is False   # ...refusal short-circuits before any call

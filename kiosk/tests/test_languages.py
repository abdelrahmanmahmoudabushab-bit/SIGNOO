"""Pressing a language card must put the visitor on *that language's* model and
vocabulary, and a language with no trained model must be refused, never quietly
served by another language's model."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
from model import SignTransformer  # noqa: E402

from core.contracts import Landmarks, Utterance
from core.features import FRAME_DIM
from kiosk.languages import LanguageRegistry


def _checkpoint(path: Path, labels: list[str], favoured: int) -> None:
    """A real SignTransformer rigged to answer labels[favoured] for any input."""
    model = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=len(labels))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        bias = torch.zeros(len(labels))
        bias[favoured] = 10.0
        model.head[-1].bias.copy_(bias)
    torch.save({"model": model.state_dict(),
                "labels": {name: i for i, name in enumerate(labels)},
                "input_dim": FRAME_DIM * 2}, path)


def _world(tmp_path: Path, asl_confident: float | None = None) -> LanguageRegistry:
    (tmp_path / "models").mkdir()
    _checkpoint(tmp_path / "models" / "liu.pt", ["marhaba", "shukran"], favoured=0)
    _checkpoint(tmp_path / "models" / "asl.pt", ["dog", "cat"], favoured=0)
    (tmp_path / "liu_vocab.json").write_text(json.dumps(
        [{"id": "marhaba", "ar": "مرحبا", "en": "hello"}, {"id": "shukran", "ar": "شكرا", "en": "thank you"}],
        ensure_ascii=False), encoding="utf-8")
    (tmp_path / "asl_vocab.json").write_text(json.dumps(
        [{"id": "dog", "ar": "كلب", "en": "dog"}, {"id": "cat", "ar": "قطة", "en": "cat"}],
        ensure_ascii=False), encoding="utf-8")
    config = tmp_path / "languages.json"
    config.write_text(json.dumps([
        {"code": "liu", "model": "models/liu.pt", "vocab": "liu_vocab.json", "status": "live", "real_data": False},
        {"code": "asl", "model": "models/asl.pt", "vocab": "asl_vocab.json", "status": "pilot", "real_data": True,
         **({"confident_threshold": asl_confident} if asl_confident else {})},
        {"code": "esl", "model": None, "vocab": None, "status": "coming_soon", "real_data": False},
        {"code": "ssl", "model": "models/missing.pt", "vocab": "liu_vocab.json", "status": "coming_soon", "real_data": False},
    ]), encoding="utf-8")
    return LanguageRegistry(config, tmp_path, 0.55, 0.8, 3)


def _utterance() -> Utterance:
    frames = [Landmarks(ts_ms=i * 30, pose=np.zeros((33, 3), dtype=np.float32),
                        left=np.zeros((21, 3), dtype=np.float32), right=None) for i in range(10)]
    return Utterance(landmarks=frames, t_start=0, t_end=frames[-1].ts_ms)


def test_a_language_can_demand_more_confidence_before_a_word_is_committed(tmp_path):
    """A weak model (a small pilot vocabulary) should not auto-commit at the default
    confidence; below its own bar the visitor is shown candidates to tap instead."""
    registry = _world(tmp_path, asl_confident=0.95)
    assert registry.get("asl").recognizer.confident_threshold == 0.95
    assert registry.get("liu").recognizer.confident_threshold == 0.8        # untouched default


def test_below_a_languages_bar_the_word_becomes_candidates_not_a_commit(tmp_path, monkeypatch):
    import kiosk.main as kiosk_main
    from core.contracts import Hypothesis

    monkeypatch.setattr(kiosk_main, "registry", _world(tmp_path, asl_confident=0.95))
    conn = kiosk_main.KioskConnection(_FakeWS())
    asyncio.run(conn.handle({"type": "language", "lang": "asl"}))

    asyncio.run(conn._handle_hypothesis(Hypothesis("dog", 0.9, [("dog", 0.9), ("cat", 0.06)]), _utterance()))

    assert conn.ws.sent[-1]["status"] == "candidates"
    assert conn.assembler.gloss_seq == []                                   # nothing was committed


def test_only_languages_with_a_loadable_model_are_ready(tmp_path):
    registry = _world(tmp_path)
    ready = {code: pack.ready for code, pack in registry.packs.items()}
    assert ready == {"liu": True, "asl": True, "esl": False, "ssl": False}
    assert registry.get("ssl").error            # a missing checkpoint is reported, not raised


def test_each_language_is_recognised_by_its_own_model(tmp_path):
    registry = _world(tmp_path)
    assert registry.get("liu").recognizer.classify(_utterance()).gloss == "marhaba"
    assert registry.get("asl").recognizer.classify(_utterance()).gloss == "dog"


def test_summary_tells_the_frontend_what_can_be_pressed(tmp_path):
    summary = {row["code"]: row for row in _world(tmp_path).summary()}
    assert summary["asl"]["ready"] and summary["asl"]["status"] == "pilot"
    assert summary["asl"]["vocab_url"] == "/asl_vocab.json"
    assert not summary["esl"]["ready"] and summary["esl"]["vocab_url"] is None


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def test_pressing_a_card_switches_the_visitor_to_that_language(tmp_path, monkeypatch):
    import kiosk.main as kiosk_main

    monkeypatch.setattr(kiosk_main, "registry", _world(tmp_path))
    conn = kiosk_main.KioskConnection(_FakeWS())
    assert conn.pack.code == "liu"

    asyncio.run(conn.handle({"type": "language", "lang": "asl"}))

    assert conn.pack.code == "asl"
    assert conn.session.language == "asl" and conn.capture.language == "asl"
    conn.assembler.append("dog")
    assert conn.assembler.render().ar == "كلب"            # the American vocabulary, not the Jordanian one


def test_a_language_without_a_model_is_refused_and_nothing_changes(tmp_path, monkeypatch):
    import kiosk.main as kiosk_main

    monkeypatch.setattr(kiosk_main, "registry", _world(tmp_path))
    conn = kiosk_main.KioskConnection(_FakeWS())
    asyncio.run(conn.handle({"type": "language", "lang": "asl"}))

    asyncio.run(conn.handle({"type": "language", "lang": "esl"}))

    assert conn.pack.code == "asl"                          # not silently reset or reassigned
    assert conn.ws.sent[-1]["code"] == "language_unavailable"

"""Pressing a language card must put the visitor on *that language's* whole engine
(model, vocabulary, knowledge base, clip library), and a language with no trained
model must be refused, never quietly served by another language's."""
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


def _kb_entry(keywords: list[str], ar: str, en: str) -> dict:
    return {"id": "kb_1", "site_id": "default", "keywords": keywords,
            "answer_ar": ar, "answer_en": en, "updated_at": "2026-10-11T00:00:00Z"}


def _clip(path: str | None) -> dict:
    return {"clip_path": path, "recorded_by": None, "recorded_at": None}


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _world(tmp_path: Path, asl_confident: float | None = None,
           asl_segmentation: dict | None = None, asl_kb: list | None = None,
           asl_clips: bool = True) -> LanguageRegistry:
    """`asl_kb=None` leaves the American pack with no knowledge base configured at all;
    a list (even an empty one) gives it its own."""
    # The Jordanian knowledge base also matches the Arabic for "dog", so an American
    # visitor's question would be answered from it if anything were shared. The two
    # clip libraries disagree about dog/cat for the same reason.
    _write(tmp_path / "liu_kb.json", [_kb_entry(["مرحبا", "كلب"], "مرحبا شكرا", "hello thank you")])
    _write(tmp_path / "liu_clips.json", {"marhaba": _clip("clips/liu/marhaba.bvh"), "shukran": _clip(None),
                                         "dog": _clip(None), "cat": _clip("clips/liu/cat.bvh")})
    _write(tmp_path / "asl_clips.json", {"dog": _clip("clips/asl/dog.bvh"), "cat": _clip(None)})
    if asl_kb is not None:
        _write(tmp_path / "asl_kb.json", asl_kb)
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
        {"code": "liu", "model": "models/liu.pt", "vocab": "liu_vocab.json", "status": "live", "real_data": False,
         "knowledge_base": "liu_kb.json", "clip_library": "liu_clips.json"},
        {"code": "asl", "model": "models/asl.pt", "vocab": "asl_vocab.json", "status": "pilot", "real_data": True,
         **({"knowledge_base": "asl_kb.json"} if asl_kb is not None else {}),
         **({"clip_library": "asl_clips.json"} if asl_clips else {}),
         **({"confident_threshold": asl_confident} if asl_confident else {}),
         **({"segmentation": asl_segmentation} if asl_segmentation else {})},
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


def test_a_language_brings_its_own_segmentation_tuning(tmp_path, monkeypatch):
    """Signing tempo differs between languages, so word-boundary timing is part of a
    language's engine, selected together with its model."""
    import kiosk.main as kiosk_main

    registry = _world(tmp_path, asl_segmentation={"pause_ms": 321, "min_frames": 9})
    monkeypatch.setattr(kiosk_main, "registry", registry)
    conn = kiosk_main.KioskConnection(_FakeWS())
    default_pause = conn.segmenter.pause_ms
    assert default_pause != 321

    asyncio.run(conn.handle({"type": "language", "lang": "asl"}))
    assert (conn.segmenter.pause_ms, conn.segmenter.min_frames) == (321, 9)
    # Untouched settings fall back to the kiosk defaults rather than disappearing.
    assert conn.segmenter.motion_threshold == kiosk_main.CONFIG.segmentation_motion_threshold

    asyncio.run(conn._end_session())                       # the next visitor starts on the default engine
    assert conn.segmenter.pause_ms == default_pause


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
    assert summary["liu"]["has_knowledge_base"] and summary["liu"]["has_clip_library"]
    assert not summary["asl"]["has_knowledge_base"] and summary["asl"]["has_clip_library"]


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


def _connection(monkeypatch, registry, default: str = "liu"):
    """A connection on `registry`, with the cloud phrasing call switched off."""
    import kiosk.main as kiosk_main

    monkeypatch.setattr(kiosk_main, "registry", registry)
    monkeypatch.setattr(kiosk_main, "DEFAULT_LANGUAGE", default)
    monkeypatch.setattr(kiosk_main.Dialogue, "_try_cloud", lambda self, question, entry: None)
    return kiosk_main.KioskConnection(_FakeWS())


def _ask(conn, lang: str, gloss: str) -> dict:
    """Pick a language, sign one word, ask; returns the messages by type."""
    asyncio.run(conn.handle({"type": "language", "lang": lang}))
    conn.assembler.append(gloss)
    conn.ws.sent.clear()
    asyncio.run(conn.handle({"type": "ask"}))
    return {m["type"]: m for m in conn.ws.sent}


def test_a_question_is_answered_only_from_the_selected_languages_own_knowledge_base(tmp_path, monkeypatch):
    """The Jordanian knowledge base matches this question, but an American visitor is
    refused whether the American pack has an empty knowledge base or none at all."""
    for n, asl_kb in enumerate(([], None)):
        (tmp_path / str(n)).mkdir()
        registry = _world(tmp_path / str(n), asl_kb=asl_kb)
        _, score = registry.get("liu").kb.best_match("كلب")
        assert score >= 0.2                                     # the Jordanian KB would answer it
        assert (registry.get("asl").kb is None) == (asl_kb is None)

        sent = _ask(_connection(monkeypatch, registry), "asl", "dog")
        assert sent["answer"]["refused"]
        assert "playback" not in sent

    sent = _ask(_connection(monkeypatch, registry), "liu", "marhaba")
    assert not sent["answer"]["refused"] and sent["answer"]["en"] == "hello thank you"


def test_playback_and_unrenderable_words_use_the_selected_languages_own_clips(tmp_path, monkeypatch):
    """The two clip libraries say opposite things about dog and cat; an American answer
    must be described by the American one."""
    registry = _world(tmp_path, asl_kb=[_kb_entry(["كلب"], "كلب قطة", "dog cat")])

    sent = _ask(_connection(monkeypatch, registry), "asl", "dog")
    assert sent["answer"]["unrenderable_words"] == ["cat"]
    assert [(w["gloss_id"], w["has_clip"]) for w in sent["playback"]["words"]] == [("dog", True), ("cat", False)]

    sent = _ask(_connection(monkeypatch, registry), "liu", "marhaba")
    assert sent["answer"]["unrenderable_words"] == ["shukran"]
    assert [(w["gloss_id"], w["has_clip"]) for w in sent["playback"]["words"]] == [("marhaba", True), ("shukran", False)]


def test_a_language_without_a_clip_library_plays_nothing_rather_than_borrowing(tmp_path, monkeypatch):
    registry = _world(tmp_path, asl_kb=[_kb_entry(["كلب"], "كلب قطة", "dog cat")], asl_clips=False)
    assert registry.get("asl").clips is None

    sent = _ask(_connection(monkeypatch, registry), "asl", "dog")
    assert sent["answer"]["unrenderable_words"] == ["dog", "cat"]      # none of it can be signed back
    assert sent["playback"]["words"] == []


def test_ending_a_session_returns_the_connection_to_the_default_languages_engine(tmp_path, monkeypatch):
    registry = _world(tmp_path, asl_kb=[])
    liu, asl = registry.get("liu"), registry.get("asl")
    conn = _connection(monkeypatch, registry)
    asyncio.run(conn.handle({"type": "language", "lang": "asl"}))
    assert (conn.dialogue.kb, conn.clips, conn.assembler.vocab) == (asl.kb, asl.clips, asl.vocab)

    asyncio.run(conn._end_session())

    assert conn.pack is liu
    assert conn.dialogue.kb is liu.kb and conn.dialogue.vocab is liu.vocab
    assert conn.clips is liu.clips and conn.dialogue.clip_library is liu.clips._entries
    assert conn.assembler.vocab is liu.vocab


def test_a_default_language_that_is_not_ready_gets_an_empty_engine_not_a_borrowed_one(tmp_path, monkeypatch):
    """`ssl` is configured but its checkpoint is missing. A connection starting there
    must not crash and must not pick up the Jordanian words, answers or clips."""
    conn = _connection(monkeypatch, _world(tmp_path), default="ssl")
    assert conn.pack.code == "ssl" and not conn.pack.ready
    assert conn.dialogue.kb is None and conn.clips is None

    conn.assembler.append("marhaba")
    assert conn.assembler.render().ar == "⟨marhaba⟩"                     # no vocabulary: shown as unknown
    asyncio.run(conn.handle({"type": "ask"}))
    assert conn.ws.sent[-1]["type"] == "answer" and conn.ws.sent[-1]["refused"]

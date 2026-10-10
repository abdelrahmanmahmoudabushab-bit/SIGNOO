"""Per-language packs: each sign language is its own isolated engine.

`core/languages.json` lists every language the kiosk menu shows. A pack owns
everything language-specific: model, vocabulary, confidence bars, segmentation
tuning, knowledge base and clip library. A language is *ready* only when its model
and vocabulary load; anything else is refused when a visitor selects it. Nothing is
shared or borrowed across languages: a language with no knowledge base of its own
refuses questions, it does not answer from another language's.

Keeping the status in data (not in `index.html`) is what lets a language go live by
dropping in a trained checkpoint, with no code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from kiosk.assembly import Vocabulary
from kiosk.dialogue import KnowledgeBase
from kiosk.playback import ClipLibrary
from kiosk.recognition import ModelUnavailable, Recognizer
from kiosk.segmentation import Segmenter

logger = logging.getLogger("signoo.kiosk.languages")

_PATH_KEYS = ("model", "vocab", "knowledge_base", "clip_library")


def _servable(path: Path | None, repo_root: Path) -> Path | None:
    """Repo-relative form of a path the frontend can fetch, or None if it is outside the repo."""
    try:
        return path.relative_to(repo_root) if path else None
    except ValueError:
        return None


@dataclass
class LanguagePack:
    code: str
    status: str                       # live | pilot | coming_soon
    real_data: bool                   # was the model trained on real human signing?
    vocab_path: Path | None
    vocab: Vocabulary | None = None
    recognizer: Recognizer | None = None
    kb: KnowledgeBase | None = None       # B7: None => this language's dialogue refuses
    clips: ClipLibrary | None = None      # B8: None => nothing to play back
    error: str | None = None
    # Word-boundary tuning for this language; any key left out uses the kiosk default.
    segmentation: dict = field(default_factory=dict)

    def make_segmenter(self, defaults: dict) -> Segmenter:
        """This language's own B3: signing tempo and pauses differ between languages, so
        boundary timing is part of its engine, chosen together with its model."""
        return Segmenter(**{**defaults, **self.segmentation})

    @property
    def ready(self) -> bool:
        return self.recognizer is not None and self.vocab is not None

    def summary(self) -> dict:
        return {
            "code": self.code,
            "status": self.status,
            "ready": self.ready,
            "real_data": self.real_data,
            "labels": len(self.recognizer.labels) if self.recognizer else 0,
            "vocab_url": f"/{self.vocab_path.as_posix()}" if self.vocab_path and self.ready else None,
            "has_knowledge_base": self.kb is not None,
            "has_clip_library": self.clips is not None,
        }


class LanguageRegistry:
    def __init__(self, config_path: Path, repo_root: Path, reject: float, confident: float, top_k: int,
                 overrides: dict[str, dict[str, Path]] | None = None):
        """`overrides` replaces one language's file paths, keyed like its languages.json
        entry: {"liu": {"model": ..., "vocab": ..., "knowledge_base": ..., "clip_library": ...}}."""
        self.packs: dict[str, LanguagePack] = {}
        for entry in json.loads(config_path.read_text(encoding="utf-8")):
            code = entry["code"]
            paths = {key: repo_root / entry[key] for key in _PATH_KEYS if entry.get(key)}
            configured = "model" in paths and "vocab" in paths
            if configured:
                paths.update((overrides or {}).get(code, {}))
            pack = LanguagePack(
                code=code, status=entry["status"], real_data=bool(entry.get("real_data")),
                vocab_path=_servable(paths.get("vocab"), repo_root),
                segmentation=dict(entry.get("segmentation", {})),
            )
            if configured:
                # A language may set its own bars: a weak pilot model should ask the
                # visitor to confirm more often than a well-trained one.
                self._load(pack, paths,
                           entry.get("reject_threshold", reject),
                           entry.get("confident_threshold", confident), top_k)
            self.packs[code] = pack

    @staticmethod
    def _load(pack: LanguagePack, paths: dict[str, Path], reject, confident, top_k) -> None:
        try:
            pack.vocab = Vocabulary(paths["vocab"])
            pack.recognizer = Recognizer(paths["model"], reject, confident, top_k)
            logger.info("language_ready", extra={"language": pack.code, "labels": len(pack.recognizer.labels)})
        except (ModelUnavailable, OSError, ValueError) as exc:
            pack.recognizer = pack.vocab = None
            pack.error = str(exc)
            logger.info("language_unavailable", extra={"language": pack.code, "reason": str(exc)})
            return
        # Recognition still works without these; the language then refuses questions
        # (no kb) or has nothing to play back (no clips) instead of borrowing.
        for attr, key, load in (("kb", "knowledge_base", KnowledgeBase),
                                ("clips", "clip_library", lambda path: ClipLibrary(path, pack.vocab))):
            if key not in paths:
                continue
            try:
                setattr(pack, attr, load(paths[key]))
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("language_part_unavailable",
                               extra={"language": pack.code, "part": key, "reason": str(exc)})

    def get(self, code: str | None) -> LanguagePack | None:
        return self.packs.get(code) if code else None

    def summary(self) -> list[dict]:
        return [pack.summary() for pack in self.packs.values()]

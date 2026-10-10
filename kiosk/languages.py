"""Per-language packs: each sign language gets its own model and vocabulary.

`core/languages.json` lists every language the kiosk menu shows. A language is
*ready* only when its model file exists and loads; anything else is refused when a
visitor selects it, never quietly served by another language's model. A Jordanian
visitor's signs run through the Jordanian model, an American visitor's through the
American one.

Keeping the status in data (not in `index.html`) is what lets a language go live by
dropping in a trained checkpoint, with no code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from kiosk.assembly import Vocabulary
from kiosk.recognition import ModelUnavailable, Recognizer
from kiosk.segmentation import Segmenter

logger = logging.getLogger("signoo.kiosk.languages")


@dataclass
class LanguagePack:
    code: str
    status: str                       # live | pilot | coming_soon
    real_data: bool                   # was the model trained on real human signing?
    vocab_path: Path | None
    vocab: Vocabulary | None = None
    recognizer: Recognizer | None = None
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
        }


class LanguageRegistry:
    def __init__(self, config_path: Path, repo_root: Path, reject: float, confident: float, top_k: int,
                 model_overrides: dict[str, Path] | None = None):
        overrides = model_overrides or {}
        self.packs: dict[str, LanguagePack] = {}
        for entry in json.loads(config_path.read_text(encoding="utf-8")):
            pack = LanguagePack(
                code=entry["code"], status=entry["status"], real_data=bool(entry.get("real_data")),
                vocab_path=Path(entry["vocab"]) if entry.get("vocab") else None,
                segmentation=dict(entry.get("segmentation", {})),
            )
            if entry.get("model") and entry.get("vocab"):
                model_path = overrides.get(pack.code) or repo_root / entry["model"]
                # A language may set its own bars: a weak pilot model should ask the
                # visitor to confirm more often than a well-trained one.
                self._load(pack, model_path, repo_root / entry["vocab"],
                           entry.get("reject_threshold", reject),
                           entry.get("confident_threshold", confident), top_k)
            self.packs[pack.code] = pack

    @staticmethod
    def _load(pack: LanguagePack, model_path: Path, vocab_path: Path, reject, confident, top_k) -> None:
        try:
            pack.vocab = Vocabulary(vocab_path)
            pack.recognizer = Recognizer(model_path, reject, confident, top_k)
            logger.info("language_ready", extra={"language": pack.code, "labels": len(pack.recognizer.labels)})
        except (ModelUnavailable, OSError, ValueError) as exc:
            pack.recognizer = pack.vocab = None
            pack.error = str(exc)
            logger.info("language_unavailable", extra={"language": pack.code, "reason": str(exc)})

    def get(self, code: str | None) -> LanguagePack | None:
        return self.packs.get(code) if code else None

    def summary(self) -> list[dict]:
        return [pack.summary() for pack in self.packs.values()]

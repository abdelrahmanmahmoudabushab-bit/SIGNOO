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
from kiosk.recognition import Recognizer
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
    # (reject, confident, top_k) this language's Recognizer is built with, so a model
    # rolled out later (kiosk/model_sync.py) keeps them. None => no model configured.
    bars: tuple[float, float, int] | None = None
    # Version the server rolled out to this kiosk; None => the factory checkpoint.
    model_version: str | None = None
    # The checkpoint that shipped with the kiosk. A rolled-out model that the server later
    # withdraws (a rollback) is replaced by this one: it is the only model the kiosk can
    # fall back to without the network.
    factory_model: Path | None = None

    def make_segmenter(self, defaults: dict) -> Segmenter:
        """This language's own B3: signing tempo and pauses differ between languages, so
        boundary timing is part of its engine, chosen together with its model."""
        return Segmenter(**{**defaults, **self.segmentation})

    # Set by LanguageRegistry from SIGNOO_ALLOW_SYNTHETIC. A model trained only on
    # generated data recognises the generator, not people, so by default it is never
    # offered to a visitor (CLAUDE.md: a language not trained on real data says "coming
    # soon"). Development and demos opt in explicitly.
    allow_synthetic: bool = False

    @property
    def ready(self) -> bool:
        """The model and vocabulary loaded. Says nothing about whether it may be offered."""
        return self.recognizer is not None and self.vocab is not None

    @property
    def selectable(self) -> bool:
        """What a visitor may be put on: loaded, and trained on real people (or the
        deployment has explicitly allowed synthetic models)."""
        return self.ready and (self.real_data or self.allow_synthetic)

    def summary(self) -> dict:
        return {
            "code": self.code,
            "status": self.status,
            # The frontend enables a language card from this field, so it is the
            # selectable answer, not merely "a checkpoint loaded".
            "ready": self.selectable,
            "model_loaded": self.ready,
            "real_data": self.real_data,
            "labels": len(self.recognizer.labels) if self.recognizer else 0,
            "model_version": self.model_version,
            "vocab_url": f"/{self.vocab_path.as_posix()}" if self.vocab_path and self.ready else None,
            "has_knowledge_base": self.kb is not None,
            "has_clip_library": self.clips is not None,
        }


class LanguageRegistry:
    def __init__(self, config_path: Path, repo_root: Path, reject: float, confident: float, top_k: int,
                 overrides: dict[str, dict[str, Path]] | None = None,
                 installed: dict[str, dict] | None = None,
                 allow_synthetic: bool = True,
                 kb_dir: Path | None = None):
        """`overrides` replaces one language's file paths, keyed like its languages.json
        entry: {"liu": {"model": ..., "vocab": ..., "knowledge_base": ..., "clip_library": ...}}.
        `installed` is what the server rolled out earlier (model_sync.read_installed):
        {"asl": {"version": ..., "path": ...}}. It is preferred over the configured model."""
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
                allow_synthetic=allow_synthetic,
                factory_model=paths.get("model"),
            )
            if configured:
                # A language may set its own bars: a weak pilot model should ask the
                # visitor to confirm more often than a well-trained one.
                pack.bars = (entry.get("reject_threshold", reject),
                             entry.get("confident_threshold", confident), top_k)
                self._load(pack, paths, (installed or {}).get(code), kb_dir)
            self.packs[code] = pack

    @staticmethod
    def _load(pack: LanguagePack, paths: dict[str, Path], installed: dict | None = None,
              kb_dir: Path | None = None) -> None:
        try:
            pack.vocab = Vocabulary(paths["vocab"])
            if installed is not None:
                try:
                    pack.recognizer = Recognizer(installed["path"], *pack.bars)
                    pack.model_version = installed["version"]
                except Exception as exc:   # missing, corrupt, or the wrong shape
                    # Back to the factory checkpoint; with no version to report, the
                    # next sync downloads the assigned model again.
                    logger.warning("installed_model_unavailable", extra={
                        "language": pack.code, "version": installed["version"], "reason": str(exc)})
            if pack.recognizer is None:
                pack.recognizer = Recognizer(paths["model"], *pack.bars)
            logger.info("language_ready", extra={"language": pack.code, "labels": len(pack.recognizer.labels),
                                                 "model_version": pack.model_version})
        except Exception as exc:
            # Any failure at all: a checkpoint that unpickles but is the wrong shape
            # raises RuntimeError/KeyError/AttributeError, not ModelUnavailable. One bad
            # language must be unavailable, not stop the kiosk from starting.
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
            # A knowledge base the server delivered (kiosk/kb_sync.py) outlives a restart and
            # is preferred; if it will not load, the shipped seed is used rather than nothing.
            candidates = [paths[key]]
            if attr == "kb" and kb_dir is not None and (kb_dir / f"{pack.code}.json").is_file():
                candidates.insert(0, kb_dir / f"{pack.code}.json")
            for candidate in candidates:
                try:
                    setattr(pack, attr, load(candidate))
                    break
                except (OSError, ValueError, TypeError) as exc:
                    logger.warning("language_part_unavailable",
                                   extra={"language": pack.code, "part": key, "reason": str(exc)})

    def get(self, code: str | None) -> LanguagePack | None:
        return self.packs.get(code) if code else None

    def summary(self) -> list[dict]:
        return [pack.summary() for pack in self.packs.values()]

    def model_versions(self) -> dict[str, str]:
        """Rolled-out version each language is serving, for the heartbeat. Factory models
        have none, and a language that is not ready is serving nothing."""
        return {code: pack.model_version for code, pack in self.packs.items()
                if pack.ready and pack.model_version}

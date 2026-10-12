"""B14 helpers: the "unrenderable word" warning (FR-W7) and file-based sharing with
kiosk/'s B7 (SYSTEM_DESIGN.md B7/B8).

Reads `core/vocab.json` + `core/clip_library.json` fresh on every call, per the
task's instruction to read it, not copy it — deliberately not cached across calls,
since the clip library grows during docs/ROADMAP.md Phase 6 while the server keeps
running, and a stale in-process cache would silently hide newly-recorded clips.
Callers that process many entries in one request (e.g. `GET /kb`) should load once
via `load_vocab_index()`/`load_clip_library()` and pass the result to
`unrenderable_words()`, rather than letting it reload per entry.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from server.config import settings

_WORD_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def _load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def language_paths(language: str) -> dict[str, Path]:
    """The files one language's engine owns, from core/languages.json (the file the kiosk
    reads). "liu" keeps its configurable paths in `settings`, which is how it always worked
    and what tests redirect. A language with no entry, or no such part, simply has no path
    for it: nothing is borrowed from another language."""
    paths: dict[str, Path] = {}
    entries = _load_json(settings.languages_path) or []
    entry = next((e for e in entries if e.get("code") == language), {})
    for key, setting_name in (("vocab", "vocab_path"), ("knowledge_base", "kiosk_kb_path"),
                              ("clip_library", "clip_library_path")):
        if language == "liu":
            paths[key] = getattr(settings, setting_name)
        elif entry.get(key):
            paths[key] = settings.languages_path.parent.parent / entry[key]
    return paths


def kb_path_for(language: str) -> Path | None:
    return language_paths(language).get("knowledge_base")


def load_vocab_index(language: str = "liu") -> dict[str, str]:
    """Maps every whole-phrase `ar`/`en` value, lowercased, plus each id itself, to
    its vocab id, for one language's vocabulary."""
    path = language_paths(language).get("vocab")
    vocab = _load_json(path) if path else []
    index: dict[str, str] = {}
    for entry in vocab:
        index[entry["ar"].strip().lower()] = entry["id"]
        index[entry["en"].strip().lower()] = entry["id"]
        index[entry["id"]] = entry["id"]
    return index


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def unrenderable_words(
    answer_ar: str, answer_en: str, keywords: list[str],
    vocab_index: dict[str, str] | None = None,
    clip_library: dict | None = None,
    language: str = "liu",
) -> list[str]:
    """Vocab ids referenced by this entry whose clip library entry is missing or
    has `clip_path: null`. Right now that is every entry, by design (§API_CONTRACTS
    §2) — the whole clip library is unrecorded until docs/ROADMAP.md Phase 6.

    Pass `vocab_index`/`clip_library` when checking several entries in one request
    to avoid re-reading both files from disk per entry; omitted, they're loaded
    fresh for this one call.
    """
    if vocab_index is None:
        vocab_index = load_vocab_index(language)
    if clip_library is None:
        clip_library = load_clip_library(language)

    referenced: set[str] = set()
    haystacks = [answer_ar, answer_en] + list(keywords)
    for text in haystacks:
        phrase = text.strip().lower()
        if phrase in vocab_index:
            referenced.add(vocab_index[phrase])
        for token in _tokens(text):
            if token in vocab_index:
                referenced.add(vocab_index[token])

    unrenderable = []
    for word_id in sorted(referenced):
        entry = clip_library.get(word_id)
        if entry is None or entry.get("clip_path") is None:
            unrenderable.append(word_id)
    return unrenderable


def load_clip_library(language: str = "liu") -> dict:
    """Read-only view of one language's clip library for the console's KB editor preview
    (FR-W7) — the same file `unrenderable_words` above already reads, exposed as an
    endpoint so the console doesn't need its own copy of the path/parsing logic.
    """
    path = language_paths(language).get("clip_library")
    return _load_json(path) if path else {}


def sync_kiosk_kb_file(db_entries: list[dict], language: str = "liu") -> None:
    """Writes the same content kiosk/'s B7 reads for one language: `kiosk/data/knowledge_base.json`
    for "liu", the language's own file (core/languages.json) for the others. A language with
    no knowledge-base file listed has nowhere to be written, and the kiosk then refuses its
    questions: that is raised, not silently dropped, so an editor is told.

    File-based sharing for this dev-simulation phase only — a real deployment syncs
    this over the network (mTLS, store-and-forward) instead of a shared file path.
    See server/README.md.

    Written to a temp file in the same directory and moved into place with
    `os.replace` (atomic on both POSIX and Windows), so kiosk B7 reading this file
    concurrently with a write never observes a partially written document.
    """
    path = kb_path_for(language)
    if path is None:
        raise ValueError(f"language '{language}' has no knowledge_base file in core/languages.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(db_entries, ensure_ascii=False, indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise

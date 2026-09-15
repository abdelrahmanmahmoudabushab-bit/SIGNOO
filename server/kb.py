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


def load_vocab_index() -> dict[str, str]:
    """Maps every whole-phrase `ar`/`en` value, lowercased, plus each id itself, to
    its vocab id."""
    vocab = _load_json(settings.vocab_path)
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
) -> list[str]:
    """Vocab ids referenced by this entry whose clip library entry is missing or
    has `clip_path: null`. Right now that is every entry, by design (§API_CONTRACTS
    §2) — the whole clip library is unrecorded until docs/ROADMAP.md Phase 6.

    Pass `vocab_index`/`clip_library` when checking several entries in one request
    to avoid re-reading both files from disk per entry; omitted, they're loaded
    fresh for this one call.
    """
    if vocab_index is None:
        vocab_index = load_vocab_index()
    if clip_library is None:
        clip_library = _load_json(settings.clip_library_path)

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


def load_clip_library() -> dict:
    """Read-only view of `core/clip_library.json` for the console's KB editor preview
    (FR-W7) — the same file `unrenderable_words` above already reads, exposed as an
    endpoint so the console doesn't need its own copy of the path/parsing logic.
    """
    return _load_json(settings.clip_library_path)


def sync_kiosk_kb_file(db_entries: list[dict]) -> None:
    """Writes the same content kiosk/'s B7 reads, at `kiosk/data/knowledge_base.json`.

    File-based sharing for this dev-simulation phase only — a real deployment syncs
    this over the network (mTLS, store-and-forward) instead of a shared file path.
    See server/README.md.

    Written to a temp file in the same directory and moved into place with
    `os.replace` (atomic on both POSIX and Windows), so kiosk B7 reading this file
    concurrently with a write never observes a partially written document.
    """
    path = settings.kiosk_kb_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(db_entries, ensure_ascii=False, indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise

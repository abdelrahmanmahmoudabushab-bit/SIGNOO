"""B8 Sign playback: answer text -> a sequence of clip lookups (SYSTEM_DESIGN.md B8).

`core/clip_library.json` has every entry with `clip_path: null` right now — no
recordings exist yet, by design (`docs/API_CONTRACTS.md §2`). That is the normal
state this module handles, not an error: every word gets a "clip not yet recorded"
placeholder, sequenced and highlighted exactly as a real clip would be, which is
the console-side warning FR-W7 describes from the other end.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class PlaybackWord:
    gloss_id: str | None       # None when the word has no vocabulary entry at all
    ar: str
    en: str
    clip_path: str | None      # None => render the "not yet recorded" placeholder
    has_clip: bool


class ClipLibrary:
    def __init__(self, clip_library_path: Path, vocab):
        self.path = clip_library_path
        self.vocab = vocab
        self.reload()

    def reload(self) -> None:
        self._entries: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))

    def sequence_for_gloss(self, gloss_seq: list[str]) -> list[PlaybackWord]:
        """The straightforward path: playback for a recognised utterance, where
        every item is already a vocabulary gloss id.
        """
        return [self._lookup(gloss_id) for gloss_id in gloss_seq]

    def sequence_for_text(self, ar_text: str, en_text: str) -> list[PlaybackWord]:
        """B7's answer is free text, not a gloss sequence — map each Arabic word
        back to a vocabulary id (best-effort matching, since B5's dictionary was
        built for the recognised side, not generated text) and fingerspell
        (`gloss_id=None`) anything with no vocabulary entry, per FR-A3.
        """
        words = [w for w in _WORD_RE.findall(ar_text) if w]
        out = []
        for word in words:
            gloss_id = self.vocab._by_id and next(
                (gid for gid, e in self.vocab._by_id.items() if e["ar"] == word or word in e["ar"]),
                None,
            )
            if gloss_id:
                out.append(self._lookup(gloss_id))
            else:
                out.append(PlaybackWord(gloss_id=None, ar=word, en=word, clip_path=None, has_clip=False))
        return out

    def _lookup(self, gloss_id: str) -> PlaybackWord:
        entry = self.vocab._by_id.get(gloss_id, {"ar": gloss_id, "en": gloss_id})
        clip = self._entries.get(gloss_id)
        clip_path = clip.get("clip_path") if clip else None
        return PlaybackWord(
            gloss_id=gloss_id,
            ar=entry["ar"],
            en=entry["en"],
            clip_path=clip_path,
            has_clip=clip_path is not None,
        )

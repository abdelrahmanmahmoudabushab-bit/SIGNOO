"""B5 Assembly: gloss sequence -> Arabic/English text (SYSTEM_DESIGN.md B5).

Template and dictionary based, deliberately not a learned translation model — see
the rationale in SYSTEM_DESIGN.md B5. `core/vocab.json` is the single word list
shared with `server/`'s KB editor and the console (`docs/API_CONTRACTS.md §1`).
"""
from __future__ import annotations

import json
from pathlib import Path

from core.contracts import Rendered


class Vocabulary:
    def __init__(self, vocab_path: Path | None):
        """`None` is an empty vocabulary: what a connection holds when its language has
        none loaded, rather than another language's words."""
        entries = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path else []
        self._by_id = {e["id"]: e for e in entries}

    def __contains__(self, gloss_id: str) -> bool:
        return gloss_id in self._by_id

    def ar(self, gloss_id: str) -> str:
        entry = self._by_id.get(gloss_id)
        return entry["ar"] if entry else f"⟨{gloss_id}⟩"

    def en(self, gloss_id: str) -> str:
        entry = self._by_id.get(gloss_id)
        return entry["en"] if entry else f"⟨{gloss_id}⟩"


class Assembler:
    """Accumulates a session's committed gloss sequence and renders it bilingually.

    Uncertain entries (FR-K8 candidates the visitor has not yet resolved) are never
    added here — only committed, resolved words enter `gloss_seq`. This keeps the
    invariant "everything in Rendered.gloss_seq was actually decided" true, so B6
    never has to guess which words are provisional.
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.gloss_seq: list[str] = []

    def append(self, gloss_id: str) -> None:
        self.gloss_seq.append(gloss_id)

    def remove_last(self) -> None:
        if self.gloss_seq:
            self.gloss_seq.pop()

    def clear(self) -> None:
        self.gloss_seq.clear()

    def render(self) -> Rendered:
        ar = " ".join(self.vocab.ar(g) for g in self.gloss_seq)
        en = " ".join(self.vocab.en(g) for g in self.gloss_seq)
        return Rendered(gloss_seq=list(self.gloss_seq), ar=ar, en=en, uncertain=[])

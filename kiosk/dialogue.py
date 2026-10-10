"""B7 Dialogue: utterance -> grounded answer (SYSTEM_DESIGN.md B7, FR-A1-A5).

Retrieval over a curated per-site knowledge base (`docs/API_CONTRACTS.md §3`), then
one of two phrasing paths:

- **cloud** — a real Claude API call phrases the retrieved content as a short,
  concrete bilingual sentence. This is the "cloud-primary" half of FR-K14.
- **local_fallback** — the retrieved fields rendered through a fixed template, no
  network call, used automatically when the cloud call fails or is disabled/slow.
  Carries a visible `source: "local_fallback"` flag (never a retry button, per
  NFR-P3b) — `Answer.source` in `core/contracts.py` exists for exactly this.

No match above `kb_similarity_floor` is a **refusal**, never an improvised answer
(FR-A1) — the worst output here is a plausible invented one.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from core.contracts import Answer

logger = logging.getLogger("signoo.kiosk.dialogue")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenise(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


@dataclass
class KBEntry:
    id: str
    site_id: str
    keywords: list[str]
    answer_ar: str
    answer_en: str
    updated_at: str


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[KBEntry] = []
        self.reload()

    def reload(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = [KBEntry(**e) for e in raw]

    def best_match(self, question: str) -> tuple[KBEntry | None, float]:
        """Keyword-overlap retrieval (FR-A1): no embeddings, no ranking model —
        a bounded vocabulary and short utterances make substring/keyword matching
        predictable and debuggable, same rationale as B5's template rendering.
        """
        q_tokens = _tokenise(question)
        if not q_tokens:
            return None, 0.0

        best_entry, best_score = None, 0.0
        for entry in self.entries:
            kw_tokens = set()
            for kw in entry.keywords:
                kw_tokens |= _tokenise(kw)
            if not kw_tokens:
                continue
            overlap = len(q_tokens & kw_tokens)
            score = overlap / len(kw_tokens)
            if score > best_score:
                best_entry, best_score = entry, score
        return best_entry, best_score


class Dialogue:
    def __init__(self, kb: KnowledgeBase | None, similarity_floor: float, claude_model: str,
                 claude_timeout_s: float, cloud_enabled: bool, vocab, clip_library: dict):
        self.kb = kb
        self.similarity_floor = similarity_floor
        self.claude_model = claude_model
        self.claude_timeout_s = claude_timeout_s
        self.cloud_enabled = cloud_enabled
        self.vocab = vocab
        self.clip_library = clip_library
        self._client = None

    def _unrenderable_words(self, text: str) -> list[str]:
        """FR-A3: flag words the answer uses that the sign clip library cannot
        render yet (`clip_path is None`) — normal today (models/README.md), since
        no clips are recorded, but still worth surfacing so B8 can fingerspell.
        """
        tokens = _tokenise(text)
        flagged = []
        for gloss_id, entry in self.vocab._by_id.items():
            if _tokenise(entry["ar"]) & tokens or _tokenise(entry["en"]) & tokens:
                clip = self.clip_library.get(gloss_id)
                if clip is None or clip.get("clip_path") is None:
                    flagged.append(gloss_id)
        return flagged

    def answer(self, question: str) -> Answer:
        # A language with no knowledge base of its own refuses; it never answers
        # from another language's.
        entry, score = self.kb.best_match(question) if self.kb is not None else (None, 0.0)
        if entry is None or score < self.similarity_floor:
            return Answer(
                ar="عذرًا، هذا السؤال خارج نطاق ما يمكنني الإجابة عنه هنا. "
                   "يمكنك سؤال أحد الموظفين.",
                en="Sorry, that question is outside what I can answer here. "
                   "Please ask a staff member.",
                source="local_fallback",
                refused=True,
            )

        if self.cloud_enabled:
            cloud_answer = self._try_cloud(question, entry)
            if cloud_answer is not None:
                return cloud_answer

        # Local fallback: the retrieval result, verbatim, through a fixed template.
        return Answer(
            ar=entry.answer_ar,
            en=entry.answer_en,
            source="local_fallback",
            unrenderable_words=self._unrenderable_words(entry.answer_ar + " " + entry.answer_en),
        )

    def _try_cloud(self, question: str, entry: KBEntry) -> Answer | None:
        """Real Claude API call (per the delegated plan: "B7 cloud path calls the
        Claude API for real"). Any failure — network, timeout, API error — falls
        through to the local-fallback path in `answer()`; this must be seamless to
        the visitor (FR-K14), so callers never see an exception from here.
        """
        try:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic()

            prompt = (
                "You are the answering voice of a sign-language visitor kiosk. "
                "Using ONLY the facts below, write one short, concrete sentence in "
                "Modern Standard Arabic and one in English answering the visitor's "
                "question. Avoid idiom and subordinate clauses (the sentence must be "
                "sign-language-renderable). Respond as JSON: "
                '{"ar": "...", "en": "..."}\n\n'
                f"Visitor's question (as recognised, may be a short gloss sequence): {question}\n"
                f"Facts — Arabic: {entry.answer_ar}\n"
                f"Facts — English: {entry.answer_en}"
            )
            response = self._client.with_options(
                timeout=self.claude_timeout_s, max_retries=0
            ).messages.create(
                model=self.claude_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            payload = json.loads(_extract_json(text))
            ar, en = payload["ar"], payload["en"]
            return Answer(
                ar=ar,
                en=en,
                source="cloud",
                unrenderable_words=self._unrenderable_words(ar + " " + en),
            )
        except Exception as exc:
            logger.info("dialogue_cloud_fallback", extra={"reason": str(exc)})
            return None


def _extract_json(text: str) -> str:
    """Claude sometimes wraps JSON in prose or a code fence; take the first {...}."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in Claude response")
    return text[start : end + 1]

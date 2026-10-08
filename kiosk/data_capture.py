"""B10 Data capture: on consent, buffer session landmarks/predictions/corrections
locally, in exactly the shape `docs/API_CONTRACTS.md §4` specifies, ready for
`server/`'s B11 to ingest later (SYSTEM_DESIGN.md B10).

Landmarks by default, never video (NFR-S2) — the items list here is exactly what
NFR-S6 needs to locate and delete on request: one file per session, named by
`session_id`.

Nothing here POSTs to `server/`; the plan explicitly scopes that out of Phase 2
("you don't need server/ to exist or be running to build this"). See kiosk/README.md
for the note on wiring a real upload later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CapturedItem:
    utterance_id: str
    landmarks: list[list[float]]      # one 203-dim float row per frame, already
                                       # resampled (matches ml/'s manifest features)
    predicted_gloss: str | None
    confidence: float
    correction: str | None = None


@dataclass
class SessionCapture:
    session_id: str
    kiosk_id: str
    language: str
    consent_retention: bool
    started_at: str
    items: list[CapturedItem] = field(default_factory=list)
    ended_at: str | None = None

    def add(self, item: CapturedItem) -> None:
        if self.consent_retention:
            self.items.append(item)

    def to_payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "kiosk_id": self.kiosk_id,
            "language": self.language,
            "consent_retention": self.consent_retention,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "items": [
                {
                    "utterance_id": i.utterance_id,
                    "landmarks": i.landmarks,
                    "predicted_gloss": i.predicted_gloss,
                    "confidence": i.confidence,
                    "correction": i.correction,
                }
                for i in self.items
            ],
        }


def write_pending_upload(capture: SessionCapture, pending_dir: Path) -> Path | None:
    """Writes the session queue file, or does nothing if retention was declined —
    "declining it must not degrade the service" (FR-K3) also means declining it
    must not leave a file behind either.
    """
    if not capture.consent_retention:
        return None
    pending_dir.mkdir(parents=True, exist_ok=True)
    target = pending_dir / f"{capture.session_id}.json"
    target.write_text(json.dumps(capture.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def landmarks_to_rows(frames: list[np.ndarray]) -> list[list[float]]:
    return [row.tolist() for row in frames]

"""B10 Data capture: on consent, buffer session landmarks/predictions/corrections
locally, in exactly the shape `docs/API_CONTRACTS.md §4` specifies, ready for
`server/`'s B11 to ingest later (SYSTEM_DESIGN.md B10).

Landmarks by default, never video (NFR-S2) — the items list here is exactly what
NFR-S6 needs to locate and delete on request: one file per session, named by
`session_id`.

`flush_pending_uploads` delivers the queue to `server/` when an operator has set
`SIGNOO_SERVER_URL`; otherwise nothing leaves the device.
"""
from __future__ import annotations

import json
import os
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

    @property
    def signer_id(self) -> str:
        """Anonymous per-visit signer id. Visitors are never identified, so one visit
        stands in for one signer; that keeps the server's signer-independent split
        (DR-4) possible without recording who anyone is. A returning visitor is a
        new "signer", which can only make a held-out split slightly less strict,
        never leak one visit into both train and test.
        """
        return f"visitor-{self.session_id.replace('-', '')[:8]}"

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
                    "signer_id": self.signer_id,
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


def flush_pending_uploads(pending_dir: Path, server_url: str, timeout_s: float = 5.0,
                          token: str = "", rejected_dir: Path | None = None) -> int:
    """Store-and-forward B10 -> B11: POSTs each queued session file to the server's
    `/ingest/sessions` and deletes it only after the server answers 2xx. Offline, a
    server error (5xx), or a refusal that is not about the file (401/403 wrong token,
    408/429 try later) leaves it queued for the next call, so the kiosk never loses a
    consented session or blocks on the network.

    Any other 4xx means the server will never take this file as it is. Retrying it on
    every flush forever helps nobody, so it is moved to `rejected_dir` (default: a
    `rejected_uploads` folder beside the queue), still one file per session id so
    NFR-S6 deletion can find it.

    Returns how many files were delivered. A blank `server_url` disables upload and
    leaves the queue untouched; nothing leaves the device unless an operator opts in.
    """
    if not server_url or not pending_dir.exists():
        return 0
    if rejected_dir is None:
        rejected_dir = pending_dir.parent / "rejected_uploads"

    import logging
    import urllib.error
    import urllib.request

    logger = logging.getLogger("signoo.kiosk.data_capture")
    delivered = 0
    for path in sorted(pending_dir.glob("*.json")):
        try:
            request = urllib.request.Request(
                server_url.rstrip("/") + "/ingest/sessions",
                data=path.read_bytes(),
                headers={"Content-Type": "application/json",
                         **({"X-Kiosk-Token": token} if token else {})},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if 200 <= response.status < 300:
                    path.unlink()
                    delivered += 1
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 or exc.code in (401, 403, 408, 429):
                logger.warning("upload_deferred", extra={"file": path.name, "status": exc.code})
                continue
            logger.warning("upload_rejected", extra={"file": path.name, "status": exc.code})
            try:
                rejected_dir.mkdir(parents=True, exist_ok=True)
                os.replace(path, rejected_dir / path.name)
            except OSError as move_exc:   # stays queued; better retried than lost
                logger.warning("upload_reject_move_failed", extra={"file": path.name, "reason": str(move_exc)})
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.info("upload_deferred", extra={"file": path.name, "reason": str(exc)})
            break    # server unreachable: later files would fail the same way
    return delivered


def landmarks_to_rows(frames: list[np.ndarray]) -> list[list[float]]:
    return [row.tolist() for row in frames]

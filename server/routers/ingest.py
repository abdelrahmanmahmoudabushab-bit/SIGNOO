"""B11 ingest — kiosk B10 uploads land here, always into the review queue, never
into training directly (FR-L3).

Authenticated by mTLS at the transport layer in a real deployment (SYSTEM_DESIGN.md
§5 "fleet transport"); this dev server does not terminate TLS itself, so there is no
kiosk-identity check here beyond the `kiosk_id` field in the payload matching a
registered kiosk (soft check, logged if unknown — a real deployment would reject).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from server.auth import require_kiosk_token
from server.db import get_db
from server.models import Kiosk, ReviewItem, SessionUpload
from server.schemas import SessionUploadIn, SessionUploadOut

logger = logging.getLogger("server.ingest")
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/sessions", response_model=SessionUploadOut, status_code=201,
             dependencies=[Depends(require_kiosk_token)])
def upload_session(payload: SessionUploadIn, db: Session = Depends(get_db)) -> SessionUploadOut:
    if db.get(Kiosk, payload.kiosk_id) is None:
        logger.warning("session upload from unregistered kiosk_id=%s", payload.kiosk_id)

    upload = SessionUpload(
        session_id=payload.session_id,
        kiosk_id=payload.kiosk_id,
        language=payload.language,
        consent_retention=payload.consent_retention,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
    )
    db.add(upload)
    db.flush()

    for item in payload.items:
        db.add(ReviewItem(
            session_upload_id=upload.id,
            utterance_id=item.utterance_id,
            language=payload.language,
            signer_id=item.signer_id,
            landmarks=item.landmarks,
            predicted_gloss=item.predicted_gloss,
            confidence=item.confidence,
            correction=item.correction,
        ))

    db.commit()
    return SessionUploadOut(session_upload_id=upload.id, items_queued=len(payload.items))

"""B11 ingest — kiosk B10 uploads land here, always into the review queue, never
into training directly (FR-L3).

Authenticated by mTLS at the transport layer in a real deployment (SYSTEM_DESIGN.md
§5 "fleet transport"); this dev server does not terminate TLS itself, so there is no
kiosk-identity check here beyond the `kiosk_id` field in the payload matching a
registered kiosk (soft check, logged if unknown — a real deployment would reject).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.auth import authorize_kiosk, require_kiosk_token
from server.db import get_db
from server.models import Kiosk, ReviewItem, SessionUpload
from server.schemas import SessionUploadIn, SessionUploadOut

logger = logging.getLogger("server.ingest")
router = APIRouter(prefix="/ingest", tags=["ingest"])


def _stored(db: Session, session_id: str) -> SessionUploadOut | None:
    upload = db.scalars(select(SessionUpload).where(SessionUpload.session_id == session_id)).first()
    if upload is None:
        return None
    return SessionUploadOut(session_upload_id=upload.id, items_queued=len(upload.items))


@router.post("/sessions", response_model=SessionUploadOut, status_code=201)
def upload_session(payload: SessionUploadIn, response: Response,
                   kiosk_token: str = Depends(require_kiosk_token),
                   db: Session = Depends(get_db)) -> SessionUploadOut:
    """A repeat of a session already stored answers 200 with the original result and
    stores nothing, so a kiosk can safely re-send after a lost acknowledgement."""
    authorize_kiosk(kiosk_token, payload.kiosk_id)
    if not payload.consent_retention:
        # The kiosk never sends these (kiosk/data_capture.py writes no file without
        # consent). Checking here too means consent does not depend on every client
        # being well behaved: nothing a visitor declined to share is ever stored.
        raise HTTPException(status_code=422, detail="consent_retention is false; nothing may be stored")
    existing = _stored(db, payload.session_id)
    if existing is not None:
        response.status_code = 200
        return existing

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

    try:
        db.commit()
    except IntegrityError:          # the same session arrived twice at once; the other copy won
        db.rollback()
        existing = _stored(db, payload.session_id)
        if existing is None:
            raise
        response.status_code = 200
        return existing
    return SessionUploadOut(session_upload_id=upload.id, items_queued=len(payload.items))

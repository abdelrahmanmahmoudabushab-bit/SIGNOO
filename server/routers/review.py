"""B11 review queue + review-decision endpoint (FR-W3).

Nothing reaches "validated" (confirmed/corrected) without a call to the decision
endpoint here — that is what makes FR-L3 ("nothing enters training unreviewed")
true, since `server/training.py` only ever reads rows in those two states.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.db import get_db
from server.models import ReviewItem
from server.schemas import ReviewDecisionIn, ReviewItemOut

router = APIRouter(prefix="/review", tags=["review"])

REVIEW_ROLES = ("reviewer", "engineer", "admin")


def _priority_score(item: ReviewItem, validated_counts: Counter) -> float:
    """FR-L4: low-confidence, corrected, and under-represented signs first.

    Each term is normalised to roughly [0, 1] and summed rather than multiplied, so
    one very strong signal (e.g. a correction) cannot be washed out by another.
    """
    low_confidence = 1.0 - item.confidence
    corrected = 1.0 if item.correction else 0.0
    # Under-represented: fewer already-validated examples of this predicted sign
    # means reviewing this one is worth more. Capped so one totally new sign
    # doesn't dominate the ordering outright.
    under_represented = 1.0 / (1 + validated_counts.get(item.predicted_gloss, 0))
    return 2.0 * low_confidence + 1.5 * corrected + 1.0 * under_represented


def _priority_reason(item: ReviewItem, validated_counts: Counter) -> str:
    """The single dominant reason this item was prioritised (FR-W3 shows this to
    the reviewer as context) — same three signals as `_priority_score`, reported
    as whichever one is strongest rather than the opaque combined float.
    """
    if item.correction:
        return "corrected"
    if item.confidence < 0.5:
        return "low_confidence"
    if validated_counts.get(item.predicted_gloss, 0) < 3:
        return "under_represented_sign"
    return "routine"


def _to_out(item: ReviewItem, validated_counts: Counter) -> ReviewItemOut:
    return ReviewItemOut(
        id=item.id, session_id=item.session.session_id, kiosk_id=item.session.kiosk_id,
        utterance_id=item.utterance_id, language=item.language, signer_id=item.signer_id,
        predicted_gloss=item.predicted_gloss, confidence=item.confidence,
        correction=item.correction, status=item.status, validated_label=item.validated_label,
        captured_at=item.created_at.isoformat(),
        priority_score=_priority_score(item, validated_counts),
        priority_reason=_priority_reason(item, validated_counts),
    )


@router.get("/queue", response_model=list[ReviewItemOut])
def get_queue(
    language: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*REVIEW_ROLES)),
) -> list[ReviewItemOut]:
    stmt = select(ReviewItem).where(ReviewItem.status == "pending")
    if language:
        stmt = stmt.where(ReviewItem.language == language)
    pending = list(db.scalars(stmt))

    validated_stmt = select(ReviewItem.predicted_gloss, ReviewItem.validated_label).where(
        ReviewItem.status.in_(["confirmed", "corrected"])
    )
    validated_counts: Counter = Counter()
    for predicted, validated_label in db.execute(validated_stmt):
        validated_counts[validated_label or predicted] += 1

    scored = sorted(pending, key=lambda item: _priority_score(item, validated_counts), reverse=True)
    return [_to_out(item, validated_counts) for item in scored[:limit]]


@router.post("/{item_id}/decision", response_model=ReviewItemOut)
def decide(
    item_id: str,
    payload: ReviewDecisionIn,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_role(*REVIEW_ROLES)),
) -> ReviewItemOut:
    item = db.get(ReviewItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="review item not found")

    if payload.decision == "confirm":
        item.status = "confirmed"
        item.validated_label = item.predicted_gloss
    elif payload.decision == "correct":
        if not payload.validated_label:
            raise HTTPException(status_code=422, detail="validated_label required to correct")
        item.status = "corrected"
        item.validated_label = payload.validated_label
    else:  # flag
        item.status = "flagged"
        item.validated_label = None

    item.reviewed_by = user.email
    item.reviewed_at = datetime.now(timezone.utc)
    item.reviewer_note = payload.note

    db.commit()
    db.refresh(item)
    # Priority no longer matters once a decision is recorded (it has left the
    # queue), so an empty counter is fine here — score/reason are queue-ordering
    # concerns, not properties of a decided item.
    return _to_out(item, Counter())

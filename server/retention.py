"""How long uploaded visitor landmarks may sit unreviewed.

A visitor agreed to share a clip so a Deaf reviewer could check it and it could help
train the system. Until someone reviews it, it is only a stored recording of a person's
movement, and nothing in the system should be able to keep it forever by simply never
getting round to it. Reviewed clips (confirmed/corrected) are the training set and stay;
everything else is deleted once it is older than `SIGNOO_RETENTION_DAYS`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from server.config import settings
from server.models import ReviewItem

logger = logging.getLogger("server.retention")

# `flagged` is included: it is waiting on a linguist, but it too is an unreviewed clip.
PURGEABLE = ("pending", "flagged")


def purge_unreviewed(db: Session, now: datetime | None = None, days: int | None = None) -> int:
    """Deletes unreviewed clips older than the retention period; returns how many.
    0 days disables it (nothing is ever deleted automatically)."""
    days = settings.retention_days if days is None else days
    if days <= 0:
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    # Stored naive and always UTC.
    result = db.execute(
        delete(ReviewItem).where(ReviewItem.status.in_(PURGEABLE),
                                 ReviewItem.created_at < cutoff.replace(tzinfo=None))
    )
    db.commit()
    if result.rowcount:
        logger.info("retention_purged items=%d older_than_days=%d", result.rowcount, days)
    return result.rowcount

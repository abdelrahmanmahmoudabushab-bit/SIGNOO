"""FR-L5: candidate training runs on a fixed cadence, automatically.

What is automatic is only the *candidate*. A candidate is never promoted here:
promotion needs the gate, a shadow evaluation on reviewed live data, and an engineer
(server/routers/registry.py). Nothing in this module can bypass B13.

It also refuses to start a cycle until the benchmark has been frozen by a deliberate
first run (`POST /training/trigger`), because an automatic cycle scored against a
benchmark nobody has looked at is how a silent quality collapse (R10) starts.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.config import settings
from server.db import SessionLocal
from server.models import ModelVersion
from server.training import InsufficientDataError, load_frozen_split, train_candidate

logger = logging.getLogger("server.scheduler")


def cycle_due(db: Session, language: str, now: datetime | None = None,
              interval_hours: float | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    interval = timedelta(hours=interval_hours if interval_hours is not None
                         else settings.auto_train_interval_hours)
    latest = db.scalar(select(func.max(ModelVersion.created_at)).where(ModelVersion.language == language))
    if latest is None:
        return True
    if latest.tzinfo is None:                 # SQLite hands datetimes back naive
        latest = latest.replace(tzinfo=timezone.utc)
    return now - latest >= interval


def run_cycle_once(db: Session, language: str, now: datetime | None = None) -> ModelVersion | None:
    """One scheduler tick for one language. Returns the new candidate, or None when
    nothing ran (not due, no frozen benchmark, or not enough reviewed data yet)."""
    if load_frozen_split(language) is None:
        logger.info("auto_train_skipped language=%s reason=benchmark_not_frozen", language)
        return None
    if not cycle_due(db, language, now):
        return None
    try:
        candidate = train_candidate(db, language)
    except InsufficientDataError as exc:
        logger.info("auto_train_skipped language=%s reason=%s", language, exc)
        return None
    logger.info("auto_train_candidate language=%s version=%s", language, candidate.version)
    return candidate


def _tick(language: str) -> None:
    db = SessionLocal()
    try:
        run_cycle_once(db, language)
    except Exception:
        logger.exception("auto_train_failed language=%s", language)   # never kill the loop
    finally:
        db.close()


async def scheduler_loop() -> None:
    # Sleep first: a restart must not immediately kick off a multi-minute training run.
    while True:
        await asyncio.sleep(settings.auto_train_poll_minutes * 60)
        for language in settings.auto_train_languages:
            await asyncio.to_thread(_tick, language)

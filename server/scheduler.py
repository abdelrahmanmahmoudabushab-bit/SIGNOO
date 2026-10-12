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
from server.models import ModelVersion, TrainingAttempt
from server.retention import purge_unreviewed
from server.training import InsufficientDataError, TrainingInProgress, load_frozen_split, train_candidate

logger = logging.getLogger("server.scheduler")


def cycle_due(db: Session, language: str, now: datetime | None = None,
              interval_hours: float | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    interval = timedelta(hours=interval_hours if interval_hours is not None
                         else settings.auto_train_interval_hours)
    trained = db.scalar(select(func.max(ModelVersion.created_at)).where(ModelVersion.language == language))
    failed = db.get(TrainingAttempt, language)
    # A crashed run counts like a finished one: it waits for the next cycle too.
    times = [t for t in (trained, failed.attempted_at if failed else None) if t is not None]
    if not times:
        return True
    latest = max(t.replace(tzinfo=timezone.utc) for t in times)   # stored naive, always UTC
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
    except TrainingInProgress as exc:
        # Another worker (or an engineer's manual trigger) is already training this
        # language. Not a failure, and it must not be recorded as one: that would delay
        # the next real attempt by a whole cycle.
        logger.info("auto_train_skipped language=%s reason=%s", language, exc)
        return None
    except Exception as exc:
        db.rollback()
        attempt = db.get(TrainingAttempt, language) or TrainingAttempt(language=language)
        attempt.attempted_at = now or datetime.now(timezone.utc)
        attempt.error = f"{type(exc).__name__}: {exc}"[:2000]
        db.add(attempt)
        db.commit()
        raise
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


def _purge_tick() -> None:
    db = SessionLocal()
    try:
        purge_unreviewed(db)
    except Exception:
        logger.exception("retention_purge_failed")   # never kill the loop
    finally:
        db.close()


async def scheduler_loop() -> None:
    # Sleep first: a restart must not immediately kick off a multi-minute training run.
    while True:
        await asyncio.sleep(settings.auto_train_poll_minutes * 60)
        await asyncio.to_thread(_purge_tick)
        if not settings.auto_train_enabled:
            continue
        for language in settings.auto_train_languages:
            await asyncio.to_thread(_tick, language)

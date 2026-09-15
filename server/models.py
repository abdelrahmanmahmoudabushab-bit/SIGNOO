"""SQLAlchemy ORM models for all of B11-B16.

Column types are kept to the SQLAlchemy-generic set (String, Float, Boolean, JSON,
DateTime) so the schema is identical on Postgres and SQLite — see `db.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── B16: users / roles ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    # operator | reviewer | engineer | admin — API_CONTRACTS.md §5
    role: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


# ── B11: ingest / review ────────────────────────────────────────────────────────

class SessionUpload(Base):
    """One POST from kiosk B10 — the session-upload envelope, API_CONTRACTS.md §4."""

    __tablename__ = "session_uploads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, index=True)
    kiosk_id: Mapped[str] = mapped_column(String, index=True)
    language: Mapped[str] = mapped_column(String)
    consent_retention: Mapped[bool] = mapped_column(Boolean)
    started_at: Mapped[str] = mapped_column(String)
    ended_at: Mapped[str] = mapped_column(String)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    items: Mapped[list["ReviewItem"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ReviewItem(Base):
    """One utterance from an upload, sitting in the review queue (FR-L3/L4).

    Nothing here is "validated" until a reviewer calls the decision endpoint —
    `validated_label` stays null and `status` stays "pending" until then. Training
    (B12) only ever reads rows with status == "confirmed" or "corrected".
    """

    __tablename__ = "review_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_upload_id: Mapped[str] = mapped_column(ForeignKey("session_uploads.id"))
    utterance_id: Mapped[str] = mapped_column(String, index=True)
    language: Mapped[str] = mapped_column(String, index=True)
    signer_id: Mapped[str] = mapped_column(String, default="unknown", index=True)

    landmarks: Mapped[list] = mapped_column(JSON)          # 203-dim float array per frame
    predicted_gloss: Mapped[str] = mapped_column(String, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    correction: Mapped[str | None] = mapped_column(String, nullable=True)  # weak label, FR-L2

    # pending | confirmed | corrected | flagged
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    validated_label: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    session: Mapped["SessionUpload"] = relationship(back_populates="items")


# ── B13: model registry ─────────────────────────────────────────────────────────

class ModelVersion(Base):
    """Matches API_CONTRACTS.md §6 exactly."""

    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    language: Mapped[str] = mapped_column(String, default="liu", index=True)
    code_commit: Mapped[str] = mapped_column(String)
    data_snapshot: Mapped[str] = mapped_column(String)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    gate: Mapped[dict] = mapped_column(JSON, default=dict)
    # candidate | shadow | promoted | rejected | rolled_back
    status: Mapped[str] = mapped_column(String, default="candidate", index=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    shadow_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ── B14: knowledge base ──────────────────────────────────────────────────────────

class KBEntry(Base):
    """Matches API_CONTRACTS.md §3 exactly."""

    __tablename__ = "kb_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    site_id: Mapped[str] = mapped_column(String, default="default", index=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    answer_ar: Mapped[str] = mapped_column(String)
    answer_en: Mapped[str] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


# ── B15: fleet control ───────────────────────────────────────────────────────────

class Kiosk(Base):
    __tablename__ = "kiosks"

    kiosk_id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, default="default")
    language: Mapped[str] = mapped_column(String, default="liu")
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    throttled: Mapped[bool] = mapped_column(Boolean, default=False)
    uptime_s: Mapped[float | None] = mapped_column(Float, nullable=True)


class Rollout(Base):
    """A staged rollout targets a named subset of registered kiosks (B15)."""

    __tablename__ = "rollouts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    target_kiosk_ids: Mapped[list] = mapped_column(JSON, default=list)
    # pending | active | rolled_back | completed
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

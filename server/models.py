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
    # Unique: a kiosk re-sends a session whose acknowledgement it never received, and
    # a second copy would put the same clips in the queue (and in training) twice.
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True)
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
    # Set only by server/training.py: when the run began, and the same-language
    # champion its gate was judged against (null = there was none yet).
    training_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    champion_version: Mapped[str | None] = mapped_column(String, nullable=True)
    # Hashed once when training finishes so serving a download never re-reads the file
    # just to hash it. Null on rows that predate the column; the endpoint then hashes.
    checkpoint_sha256: Mapped[str | None] = mapped_column(String, nullable=True)


class TrainingLock(Base):
    """One row per language while a training run is in progress.

    The scheduler (in every server worker) and `POST /training/trigger` all start runs
    through `server/training.py::train_candidate`. Without this, two of them could train
    the same language at once and write two candidates gated against the same champion.
    A row older than the configured lease is treated as left behind by a crashed process.
    """

    __tablename__ = "training_locks"

    language: Mapped[str] = mapped_column(String, primary_key=True)
    holder: Mapped[str] = mapped_column(String)
    acquired_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class TrainingAttempt(Base):
    """The last automatic training run that crashed, per language. A crash leaves no
    ModelVersion, so without this the scheduler would retry it on every poll."""

    __tablename__ = "training_attempts"

    language: Mapped[str] = mapped_column(String, primary_key=True)
    attempted_at: Mapped[datetime] = mapped_column(DateTime)
    error: Mapped[str] = mapped_column(String)


# ── B14: knowledge base ──────────────────────────────────────────────────────────

class KBEntry(Base):
    """Matches API_CONTRACTS.md §3 exactly."""

    __tablename__ = "kb_entries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # Each language has its own knowledge base (SYSTEM_DESIGN.md §1.1b): an answer is
    # written in, and retrieved for, one language's engine only.
    language: Mapped[str] = mapped_column(String, default="liu", index=True)
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
    # Language code -> version the kiosk reports installed for that language's engine.
    model_versions: Mapped[dict] = mapped_column(JSON, default=dict)


class Rollout(Base):
    """A staged rollout targets a named subset of registered kiosks (B15)."""

    __tablename__ = "rollouts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    # pending | active | rolled_back | completed. `completed` is set by the server once
    # every targeted kiosk reports running the version (routers/fleet.py).
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    targets: Mapped[list["RolloutTarget"]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by="RolloutTarget.kiosk_id")

    @property
    def target_kiosk_ids(self) -> list[str]:
        return [t.kiosk_id for t in self.targets]

    @target_kiosk_ids.setter
    def target_kiosk_ids(self, kiosk_ids: list[str]) -> None:
        self.targets = [RolloutTarget(kiosk_id=k) for k in dict.fromkeys(kiosk_ids)]


class RolloutTarget(Base):
    """Which kiosks a rollout targets. A table, not a JSON list, so "does this rollout
    target this kiosk" is an indexed lookup instead of a scan of every rollout."""

    __tablename__ = "rollout_targets"

    rollout_id: Mapped[str] = mapped_column(ForeignKey("rollouts.id"), primary_key=True)
    kiosk_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)

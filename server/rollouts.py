"""B15 rollout rules, shared by the fleet router (kiosks asking for a model, heartbeats)
and the registry router (a rollback). Kept out of both so neither imports the other.

A rollout's life: `active` while it is being delivered, `completed` once every targeted
kiosk reports running that version, `rolled_back` when the version is pulled. Only
`active` and `completed` rollouts are served; a pulled one never is.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.models import Kiosk, ModelVersion, Rollout, RolloutTarget

SERVED = ("active", "completed")


def desired_version(db: Session, kiosk_id: str, language: str) -> ModelVersion | None:
    """The model of the newest served rollout that targets this kiosk *in this language*
    and is still promoted. A rollout of another language's model never affects the
    answer. A rolled-back version stops being served at once: the kiosk falls back to its
    previous still-promoted rollout, or to nothing (204)."""
    stmt = (
        select(ModelVersion)
        .join(Rollout, Rollout.model_version == ModelVersion.version)
        .join(RolloutTarget, RolloutTarget.rollout_id == Rollout.id)
        .where(RolloutTarget.kiosk_id == kiosk_id, Rollout.status.in_(SERVED),
               ModelVersion.language == language, ModelVersion.status == "promoted")
        .order_by(Rollout.created_at.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def refresh_completion(db: Session, kiosk_id: str) -> None:
    """Marks `active` rollouts that include this kiosk `completed` once every target
    reports the version installed. The caller commits."""
    stmt = (
        select(Rollout)
        .join(RolloutTarget, RolloutTarget.rollout_id == Rollout.id)
        .where(RolloutTarget.kiosk_id == kiosk_id, Rollout.status == "active")
    )
    for rollout in db.scalars(stmt).unique():
        version = db.get(ModelVersion, rollout.model_version)
        if version is None:
            continue
        kiosks = [db.get(Kiosk, k) for k in rollout.target_kiosk_ids]
        # A target the server has never heard from has not installed anything.
        if all(k is not None and (k.model_versions or {}).get(version.language) == version.version
               for k in kiosks):
            rollout.status = "completed"


def retire(db: Session, version: str) -> None:
    """A version was rolled back: stop serving its rollouts, and point every kiosk the
    server believed was on it at whatever it would be served now. The caller commits.

    This changes what the server will hand out and what it reports. A kiosk keeps
    running its installed model until its next sync, which is what moves it back.
    """
    mv = db.get(ModelVersion, version)
    for rollout in db.scalars(select(Rollout).where(Rollout.model_version == version)):
        if rollout.status in SERVED:
            rollout.status = "rolled_back"
    if mv is None:
        return
    db.flush()
    for kiosk in db.scalars(select(Kiosk).where(Kiosk.current_model_version == version)):
        fallback = desired_version(db, kiosk.kiosk_id, mv.language)
        kiosk.current_model_version = fallback.version if fallback else None

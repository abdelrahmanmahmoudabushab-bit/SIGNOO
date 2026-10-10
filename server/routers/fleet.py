"""B15 fleet control: kiosk registration/heartbeat + staged rollout targeting.

Registration and heartbeat are called by the kiosk itself (mTLS-authenticated in a
real deployment, same as ingest — see routers/ingest.py's note). Everything else is
console-facing and role-gated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_kiosk_token, require_role
from server.db import get_db
from server.models import Kiosk, ModelVersion, Rollout
from server.schemas import (
    KioskHeartbeatIn, KioskOut, KioskRegisterIn, RolloutIn, RolloutOut,
)

router = APIRouter(prefix="/fleet", tags=["fleet"])

READ_ROLES = ("operator", "reviewer", "engineer", "admin")
ROLLOUT_ROLES = ("engineer", "admin")
ONLINE_WINDOW = timedelta(seconds=90)  # FR-W1: "last contact" — offline past this


def _to_out(kiosk: Kiosk) -> KioskOut:
    online = (
        kiosk.last_heartbeat_at is not None
        and datetime.now(timezone.utc) - kiosk.last_heartbeat_at.replace(tzinfo=timezone.utc) < ONLINE_WINDOW
    )
    return KioskOut(
        kiosk_id=kiosk.kiosk_id, site_id=kiosk.site_id, language=kiosk.language,
        current_model_version=kiosk.current_model_version,
        last_heartbeat_at=kiosk.last_heartbeat_at.isoformat() if kiosk.last_heartbeat_at else None,
        temperature_c=kiosk.temperature_c, throttled=kiosk.throttled, uptime_s=kiosk.uptime_s,
        online=online,
    )


@router.post("/kiosks/register", response_model=KioskOut, status_code=201,
             dependencies=[Depends(require_kiosk_token)])
def register(payload: KioskRegisterIn, db: Session = Depends(get_db)) -> KioskOut:
    existing = db.get(Kiosk, payload.kiosk_id)
    if existing is not None:
        existing.site_id = payload.site_id
        existing.language = payload.language
        db.commit()
        return _to_out(existing)

    kiosk = Kiosk(kiosk_id=payload.kiosk_id, site_id=payload.site_id, language=payload.language)
    db.add(kiosk)
    db.commit()
    return _to_out(kiosk)


@router.post("/kiosks/{kiosk_id}/heartbeat", response_model=KioskOut,
             dependencies=[Depends(require_kiosk_token)])
def heartbeat(kiosk_id: str, payload: KioskHeartbeatIn, db: Session = Depends(get_db)) -> KioskOut:
    kiosk = db.get(Kiosk, kiosk_id)
    if kiosk is None:
        raise HTTPException(status_code=404, detail="kiosk not registered")

    kiosk.last_heartbeat_at = datetime.now(timezone.utc)
    # Every field below is optional on the wire; omitting one must preserve the
    # kiosk's last-known value, not reset it — a partial heartbeat that leaves out
    # `throttled` must never silently clear a real overheating condition.
    if payload.current_model_version is not None:
        kiosk.current_model_version = payload.current_model_version
    if payload.temperature_c is not None:
        kiosk.temperature_c = payload.temperature_c
    if payload.throttled is not None:
        kiosk.throttled = payload.throttled
    if payload.uptime_s is not None:
        kiosk.uptime_s = payload.uptime_s
    db.commit()
    return _to_out(kiosk)


@router.get("/kiosks", response_model=list[KioskOut])
def list_kiosks(
    db: Session = Depends(get_db), _: CurrentUser = Depends(require_role(*READ_ROLES))
) -> list[KioskOut]:
    return [_to_out(k) for k in db.scalars(select(Kiosk))]


@router.post("/rollouts", response_model=RolloutOut, status_code=201)
def create_rollout(
    payload: RolloutIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*ROLLOUT_ROLES)),
) -> RolloutOut:
    if db.get(ModelVersion, payload.model_version) is None:
        raise HTTPException(status_code=404, detail="model version not found")
    unknown = [k for k in payload.target_kiosk_ids if db.get(Kiosk, k) is None]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unregistered kiosks: {unknown}")

    rollout = Rollout(model_version=payload.model_version, target_kiosk_ids=payload.target_kiosk_ids,
                       status="active")
    db.add(rollout)
    db.flush()
    for kiosk_id in payload.target_kiosk_ids:
        db.get(Kiosk, kiosk_id).current_model_version = payload.model_version
    db.commit()
    db.refresh(rollout)
    return RolloutOut.model_validate(rollout)


@router.get("/rollouts", response_model=list[RolloutOut])
def list_rollouts(
    db: Session = Depends(get_db), _: CurrentUser = Depends(require_role(*READ_ROLES))
) -> list[RolloutOut]:
    return [RolloutOut.model_validate(r) for r in db.scalars(select(Rollout))]

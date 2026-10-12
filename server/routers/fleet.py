"""B15 fleet control: kiosk registration/heartbeat + staged rollout targeting.

Registration and heartbeat are called by the kiosk itself (mTLS-authenticated in a
real deployment, same as ingest — see routers/ingest.py's note). Everything else is
console-facing and role-gated.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, authorize_kiosk, require_kiosk_token, require_role
from server.config import REPO_ROOT
from server.db import get_db
from server.models import KBEntry, Kiosk, ModelVersion, Rollout
from server.rollouts import desired_version, refresh_completion
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
        online=online, model_versions=kiosk.model_versions or {},
    )


@router.post("/kiosks/register", response_model=KioskOut, status_code=201)
def register(payload: KioskRegisterIn, kiosk_token: str = Depends(require_kiosk_token),
             db: Session = Depends(get_db)) -> KioskOut:
    authorize_kiosk(kiosk_token, payload.kiosk_id)
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


@router.post("/kiosks/{kiosk_id}/heartbeat", response_model=KioskOut)
def heartbeat(kiosk_id: str, payload: KioskHeartbeatIn, kiosk_token: str = Depends(require_kiosk_token),
              db: Session = Depends(get_db)) -> KioskOut:
    authorize_kiosk(kiosk_token, kiosk_id)
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
    if payload.model_versions is not None:
        kiosk.model_versions = payload.model_versions
        # What a kiosk reports installed is what finishes a rollout.
        refresh_completion(db, kiosk_id)
    db.commit()
    return _to_out(kiosk)


@router.get("/kiosks/{kiosk_id}/model")
def get_model(
    kiosk_id: str,
    language: str = Query(min_length=1),
    if_none_match: str | None = Header(default=None),
    kiosk_token: str = Depends(require_kiosk_token),
    db: Session = Depends(get_db),
) -> Response:
    """The checkpoint this kiosk should be running for one language. 204 when no
    rollout assigns it one, 304 when it already has it (`If-None-Match: "<version>"`)."""
    authorize_kiosk(kiosk_token, kiosk_id)
    if db.get(Kiosk, kiosk_id) is None:
        raise HTTPException(status_code=404, detail="kiosk not registered")
    desired = desired_version(db, kiosk_id, language)
    if desired is None:
        return Response(status_code=204)

    etag = f'"{desired.version}"'
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})

    if not desired.checkpoint_path:
        raise HTTPException(status_code=409, detail=f"version '{desired.version}' has no checkpoint recorded")
    path = Path(desired.checkpoint_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise HTTPException(status_code=409,
                            detail=f"checkpoint file for version '{desired.version}' is missing on the server")
    if not desired.checkpoint_sha256:
        # A row from before the column existed: hash once, in chunks, and remember it.
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        desired.checkpoint_sha256 = digest.hexdigest()
        db.commit()
    # Streamed from disk: the file is never held in memory, however many kiosks ask at once.
    return FileResponse(path, media_type="application/octet-stream", headers={
        "X-Model-Version": desired.version,
        "X-Model-Sha256": desired.checkpoint_sha256,
        "ETag": etag,
    })


@router.get("/kiosks/{kiosk_id}/kb")
def get_knowledge_base(
    kiosk_id: str,
    language: str = Query(min_length=1),
    if_none_match: str | None = Header(default=None),
    kiosk_token: str = Depends(require_kiosk_token),
    db: Session = Depends(get_db),
) -> Response:
    """The knowledge base this kiosk should answer from, for one language: that language's
    own entries for the kiosk's site (and the `default` site). This is how the console's
    edits reach a kiosk that does not share a disk with the server. 204 when the language
    has no entries yet, which tells the kiosk to keep what it has rather than blank it;
    304 when its copy is current (`If-None-Match`)."""
    authorize_kiosk(kiosk_token, kiosk_id)
    kiosk = db.get(Kiosk, kiosk_id)
    if kiosk is None:
        raise HTTPException(status_code=404, detail="kiosk not registered")
    entries = sorted(
        db.scalars(select(KBEntry).where(KBEntry.language == language,
                                         KBEntry.site_id.in_((kiosk.site_id, "default")))),
        key=lambda e: e.id,
    )
    if not entries:
        return Response(status_code=204)
    body = [
        {"id": e.id, "site_id": e.site_id, "keywords": e.keywords, "answer_ar": e.answer_ar,
         "answer_en": e.answer_en, "updated_at": e.updated_at.isoformat()}
        for e in entries
    ]
    text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    etag = '"' + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32] + '"'
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(content=text, media_type="application/json", headers={"ETag": etag})


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
    model_version = db.get(ModelVersion, payload.model_version)
    if model_version is None:
        raise HTTPException(status_code=404, detail="model version not found")
    if model_version.status != "promoted":
        # A rollout is what kiosks download and run (GET .../model), so rolling out an
        # unpromoted version would put a model in front of visitors without the gate.
        raise HTTPException(
            status_code=409,
            detail=f"version is '{model_version.status}'; only a promoted version can be rolled out",
        )
    unknown = [k for k in payload.target_kiosk_ids if db.get(Kiosk, k) is None]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unregistered kiosks: {unknown}")

    rollout = Rollout(model_version=payload.model_version, target_kiosk_ids=payload.target_kiosk_ids,
                       status="active")
    db.add(rollout)
    db.flush()
    # `current_model_version` describes the kiosk's registered language only; a rollout
    # of another language's model must not overwrite it.
    for kiosk_id in payload.target_kiosk_ids:
        kiosk = db.get(Kiosk, kiosk_id)
        if kiosk.language == model_version.language:
            kiosk.current_model_version = payload.model_version
    # A kiosk that already reports this version needs nothing delivered.
    for kiosk_id in payload.target_kiosk_ids:
        refresh_completion(db, kiosk_id)
    db.commit()
    db.refresh(rollout)
    return RolloutOut.model_validate(rollout)


@router.get("/rollouts", response_model=list[RolloutOut])
def list_rollouts(
    db: Session = Depends(get_db), _: CurrentUser = Depends(require_role(*READ_ROLES))
) -> list[RolloutOut]:
    return [RolloutOut.model_validate(r) for r in db.scalars(select(Rollout))]

"""B12 training trigger — the manual twin of the automatic 3-day cycle in
`server/scheduler.py`; both go through `server.training.train_candidate`."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.db import get_db
from server.schemas import ModelVersionOut, TrainTriggerIn
from server.training import ChampionNotFound, InsufficientDataError, train_candidate

router = APIRouter(prefix="/training", tags=["training"])
TRIGGER_ROLES = ("engineer", "admin")


@router.post("/trigger", response_model=ModelVersionOut, status_code=201)
def trigger(
    payload: TrainTriggerIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*TRIGGER_ROLES)),
) -> ModelVersionOut:
    try:
        model_version = train_candidate(db, payload.language, payload.epochs, payload.champion_version)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChampionNotFound as exc:
        raise HTTPException(status_code=404, detail=f"champion '{exc}' not found") from exc
    return ModelVersionOut.model_validate(model_version)

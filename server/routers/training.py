"""B12 training trigger — the manual twin of the automatic 3-day cycle in
`server/scheduler.py`; both go through `server.training.train_candidate`."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.db import get_db
from server.models import ModelVersion
from server.schemas import (
    BenchmarkIn, BenchmarkOut, ModelVersionOut, TrainingCurvePoint, TrainingRunOut, TrainTriggerIn,
)
from server.training import (
    BenchmarkAlreadyFrozen, ChampionLanguageMismatch, ChampionNotFound, InsufficientDataError,
    TrainingInProgress, freeze_benchmark, train_candidate,
)

router = APIRouter(prefix="/training", tags=["training"])
TRIGGER_ROLES = ("engineer", "admin")
READ_ROLES = ("operator", "reviewer", "engineer", "admin")


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
    except ChampionLanguageMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TrainingInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ModelVersionOut.model_validate(model_version)


@router.post("/benchmark", response_model=BenchmarkOut, status_code=201)
def freeze(
    payload: BenchmarkIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*TRIGGER_ROLES)),
) -> BenchmarkOut:
    """Freezes one language's validation and test signers. Once, deliberately: every
    candidate is gated on this set for good (DR-5, DR-7a), so it is an engineer's choice
    of named people rather than whoever happened to review first."""
    try:
        frozen = freeze_benchmark(db, payload.language, payload.val, payload.test)
    except BenchmarkAlreadyFrozen as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BenchmarkOut(language=payload.language, val=frozen["val"], test=frozen["test"])


def _iso(value: datetime) -> str:
    return value.replace(tzinfo=timezone.utc).isoformat()   # stored naive, always UTC


def _curves(checkpoint_path: str | None) -> list[TrainingCurvePoint]:
    """Per-epoch history `ml/train.py` writes beside the checkpoint; empty if it is gone."""
    if not checkpoint_path:
        return []
    history = Path(checkpoint_path).with_suffix(".history.json")
    if not history.is_file():
        return []
    return [
        TrainingCurvePoint(epoch=h["epoch"], train_loss=h["train_loss"], val_accuracy=h["val_acc"])
        for h in json.loads(history.read_text(encoding="utf-8"))
    ]


@router.get("/runs", response_model=list[TrainingRunOut])
def list_runs(
    language: str | None = None,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> list[TrainingRunOut]:
    """Runs that produced a registered version, newest first. Only finished runs leave a
    row, so every entry is `completed`; a running or crashed run is not recorded
    anywhere yet and is not listed. Versions registered without a training run
    (no recorded start time) are not runs and are left out.
    """
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.training_started_at.is_not(None))
        .order_by(ModelVersion.training_started_at.desc())
    )
    if language:
        stmt = stmt.where(ModelVersion.language == language)
    return [
        TrainingRunOut(
            run_id=m.version, language=m.language,
            started_at=_iso(m.training_started_at), finished_at=_iso(m.created_at),
            status="completed", candidate_version=m.version,
            champion_version=m.champion_version, curves=_curves(m.checkpoint_path),
        )
        for m in db.scalars(stmt)
    ]

"""B12 training trigger — a job, not a live cron (see server/training.py docstring
and server/README.md for why FR-L5's 3-day cadence is future work)."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.config import settings
from server.db import get_db
from server.gate import evaluate_gate
from server.models import ModelVersion
from server.schemas import ModelVersionOut, TrainTriggerIn
from server.training import (
    InsufficientDataError, export_validated_manifest, measure_latency_ms,
    run_evaluate, run_train,
)

router = APIRouter(prefix="/training", tags=["training"])
TRIGGER_ROLES = ("engineer", "admin")


def _code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(settings.ml_dir.parent),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@router.post("/trigger", response_model=ModelVersionOut, status_code=201)
def trigger(
    payload: TrainTriggerIn,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*TRIGGER_ROLES)),
) -> ModelVersionOut:
    now = datetime.now(timezone.utc)
    version = f"{now.strftime('%Y-%m-%dT%H:%M:%S')}Z-{_code_commit()}"
    run_dir = settings.training_runs_dir / version

    try:
        manifest_path, split_path = export_validated_manifest(db, payload.language, run_dir / "data")
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    checkpoint_path = run_dir / "checkpoint.pt"
    run_train(manifest_path, split_path, checkpoint_path, epochs=payload.epochs)

    report_path = run_dir / "eval_report.json"
    metrics = run_evaluate(checkpoint_path, manifest_path, split_path, report_path)
    latency_ms = measure_latency_ms(checkpoint_path)

    champion = None
    champion_version = payload.champion_version
    if champion_version is None:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.language == payload.language, ModelVersion.status == "promoted")
            .order_by(ModelVersion.promoted_at.desc())
        )
        champion = db.scalars(stmt).first()
    else:
        champion = db.get(ModelVersion, champion_version)
        if champion is None:
            raise HTTPException(status_code=404, detail=f"champion '{champion_version}' not found")

    verdict = evaluate_gate(
        metrics, champion.metrics if champion else None,
        candidate_latency_ms=latency_ms, latency_budget_ms=settings.latency_budget_ms,
        shadow_run_complete=False,
    )

    # Data-snapshot hash: sha256 of the manifest content, per NFR-N2's
    # (code_commit, data_snapshot) reproducibility pair.
    import hashlib
    data_snapshot = "manifest-sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    model_version = ModelVersion(
        version=version, language=payload.language, code_commit=_code_commit(),
        data_snapshot=data_snapshot, metrics=metrics, gate=verdict.as_dict(),
        status="candidate", checkpoint_path=str(checkpoint_path),
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return ModelVersionOut.model_validate(model_version)

"""B13 registry + promotion-gate actions.

The gate function itself lives in `server/gate.py` (kept separate so the rejection
test in `server/tests/test_gate.py` can call it directly, with no FastAPI/DB in the
loop). This router only sequences candidate -> shadow -> promoted/rejected and
persists the verdict (DR-6, DR-7).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.auth import CurrentUser, require_role
from server.config import settings
from server.db import get_db
from server.gate import evaluate_gate
from server.models import ModelVersion
from server.schemas import ModelVersionOut
from server.shadow import evaluate_shadow

router = APIRouter(prefix="/registry", tags=["registry"])
READ_ROLES = ("operator", "reviewer", "engineer", "admin")
GATE_ROLES = ("engineer", "admin")


@router.get("", response_model=list[ModelVersionOut])
def list_versions(
    language: str | None = None,
    db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> list[ModelVersionOut]:
    stmt = select(ModelVersion)
    if language:
        stmt = stmt.where(ModelVersion.language == language)
    return [ModelVersionOut.model_validate(m) for m in db.scalars(stmt)]


@router.get("/{version}", response_model=ModelVersionOut)
def get_version(
    version: str, db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*READ_ROLES)),
) -> ModelVersionOut:
    model_version = db.get(ModelVersion, version)
    if model_version is None:
        raise HTTPException(status_code=404, detail="model version not found")
    return ModelVersionOut.model_validate(model_version)


@router.post("/{version}/shadow-complete", response_model=ModelVersionOut)
def complete_shadow(
    version: str, db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*GATE_ROLES)),
) -> ModelVersionOut:
    """FR-L7: scores the candidate on reviewed live items newer than itself
    (server/shadow.py) and recomputes the gate from that measurement. 409 until
    enough such items exist. Fleet-wide live-metric monitoring during a staged
    rollout is separate and not built; this is the pre-promotion check.
    """
    candidate = db.get(ModelVersion, version)
    if candidate is None:
        raise HTTPException(status_code=404, detail="model version not found")
    if candidate.status in ("promoted", "rolled_back"):
        # Re-judging a live or retired model would silently demote it to "shadow"/"rejected".
        raise HTTPException(status_code=409, detail=f"version is '{candidate.status}'; nothing to judge")

    champion_stmt = (
        select(ModelVersion)
        .where(ModelVersion.language == candidate.language, ModelVersion.status == "promoted")
        .order_by(ModelVersion.promoted_at.desc())
    )
    champion = db.scalars(champion_stmt).first()

    # Latency was already measured at training time and is stable for a fixed
    # checkpoint, so it is re-read from the stored gate (a structured measurement,
    # not re-parsed out of the human-readable `details` text) rather than
    # re-benchmarked.
    candidate_latency_ms = candidate.gate.get("measurements", {}).get("candidate_latency_ms")
    if candidate_latency_ms is None:
        # Broken invariant, not a recoverable case: every candidate's gate is written
        # by evaluate_gate() (server/routers/training.py), which always populates
        # this. Silently defaulting to 0ms would make the latency criterion always
        # pass — exactly wrong for a safety-critical check (NFR-P1).
        raise HTTPException(
            status_code=409,
            detail=f"version '{version}' has no recorded latency measurement; cannot evaluate the gate",
        )

    try:
        shadow = evaluate_shadow(db, candidate, champion)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not shadow.enough_data:
        raise HTTPException(status_code=409, detail=shadow.reason)

    verdict = evaluate_gate(
        candidate.metrics, champion.metrics if champion else None,
        candidate_latency_ms=candidate_latency_ms,
        latency_budget_ms=settings.latency_budget_ms,
        shadow_run_complete=shadow.ok,
    )
    verdict.details["shadow_run_complete"] = shadow.reason
    verdict.measurements.update({
        "shadow_items": float(shadow.items),
        "shadow_candidate_accuracy": shadow.candidate_accuracy,
        **({"shadow_champion_accuracy": shadow.champion_accuracy}
           if shadow.champion_accuracy is not None else {}),
    })
    candidate.gate = verdict.as_dict()
    candidate.shadow_started_at = candidate.shadow_started_at or datetime.now(timezone.utc)
    candidate.status = "shadow" if verdict.passed else "rejected"
    db.commit()
    db.refresh(candidate)
    return ModelVersionOut.model_validate(candidate)


@router.post("/{version}/promote", response_model=ModelVersionOut)
def promote(
    version: str, db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*GATE_ROLES)),
) -> ModelVersionOut:
    candidate = db.get(ModelVersion, version)
    if candidate is None:
        raise HTTPException(status_code=404, detail="model version not found")
    if candidate.status != "shadow":
        raise HTTPException(
            status_code=409,
            detail=f"version is '{candidate.status}', must complete shadow run first "
                   f"(POST /registry/{version}/shadow-complete)",
        )
    if not candidate.gate.get("passed"):
        # SYSTEM_DESIGN §6: "Promotion gate rejects candidate -> champion stays".
        raise HTTPException(status_code=409, detail="gate did not pass; champion unchanged")

    candidate.status = "promoted"
    candidate.promoted_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(candidate)
    return ModelVersionOut.model_validate(candidate)


@router.post("/{version}/rollback", response_model=ModelVersionOut)
def rollback(
    version: str, db: Session = Depends(get_db),
    _: CurrentUser = Depends(require_role(*GATE_ROLES)),
) -> ModelVersionOut:
    """One-action manual rollback (FR-L8). Does not itself revert fleet kiosks'
    `current_model_version` — pair with a new rollout targeting the previous
    champion's version (routers/fleet.py) to actually move traffic back.
    """
    model_version = db.get(ModelVersion, version)
    if model_version is None:
        raise HTTPException(status_code=404, detail="model version not found")
    model_version.status = "rolled_back"
    db.commit()
    db.refresh(model_version)
    return ModelVersionOut.model_validate(model_version)

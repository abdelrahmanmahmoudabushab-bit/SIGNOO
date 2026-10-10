"""FR-L7 shadow evaluation: how a candidate does on live data it has never seen.

"Live data" here means reviewed kiosk items that were validated *after* the candidate
was created. Everything the candidate trained on was validated before it existed, so
this window cannot overlap its training set. The champion's side of the comparison is
what the kiosk actually said at the time (`predicted_gloss`) against the same reviewer
verdicts, so both models are scored on the same items.

This replaces a "shadow run complete" button that asserted the result: the gate's
shadow criterion now rests on a measurement, and refuses to answer until there are
enough items to measure on.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.config import settings
from server.models import ModelVersion, ReviewItem

FRAME_DIM = 203      # core.features.FRAME_DIM; kept literal so a bad row is rejected, not reshaped


@dataclass
class ShadowResult:
    items: int
    candidate_accuracy: float | None
    champion_accuracy: float | None
    enough_data: bool
    ok: bool
    reason: str


def judge(items: int, candidate_accuracy: float | None, champion_accuracy: float | None,
          min_items: int, tolerance: float) -> ShadowResult:
    """The decision rule, kept pure so it can be tested without a model or a DB."""
    if items < min_items:
        return ShadowResult(
            items, candidate_accuracy, champion_accuracy, False, False,
            f"shadow window has {items} of {min_items} reviewed live items newer than this "
            f"candidate; review more before it can be judged",
        )
    if champion_accuracy is None:
        return ShadowResult(
            items, candidate_accuracy, None, True, True,
            f"no champion to compare with; candidate scored {candidate_accuracy:.1%} on {items} live items",
        )
    ok = candidate_accuracy >= champion_accuracy - tolerance
    verb = "held up against" if ok else "fell below"
    return ShadowResult(
        items, candidate_accuracy, champion_accuracy, True, ok,
        f"candidate {candidate_accuracy:.1%} {verb} champion {champion_accuracy:.1%} "
        f"on {items} live items (tolerance {tolerance:.1%})",
    )


def _window(db: Session, candidate: ModelVersion) -> list[ReviewItem]:
    stmt = select(ReviewItem).where(
        ReviewItem.language == candidate.language,
        ReviewItem.status.in_(["confirmed", "corrected"]),
        ReviewItem.reviewed_at > candidate.created_at,
    )
    return [i for i in db.scalars(stmt) if i.validated_label]


def _score_candidate(checkpoint_path: str, items: list[ReviewItem]) -> float:
    import torch

    sys.path.insert(0, str(settings.ml_dir.parent))
    from core.features import add_velocity
    from ml.model import SignTransformer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = {index: sign for sign, index in ckpt["labels"].items()}
    model = SignTransformer(input_dim=ckpt["input_dim"], num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = np.stack([add_velocity(np.asarray(i.landmarks, dtype=np.float32)) for i in items])
    with torch.no_grad():
        predicted = model(torch.from_numpy(batch).float()).argmax(dim=1).tolist()
    correct = sum(labels[p] == i.validated_label for p, i in zip(predicted, items))
    return correct / len(items)


def evaluate_shadow(db: Session, candidate: ModelVersion, champion: ModelVersion | None) -> ShadowResult:
    items = [i for i in _window(db, candidate)
             if np.asarray(i.landmarks).ndim == 2 and np.asarray(i.landmarks).shape[1] == FRAME_DIM]
    if len(items) < settings.shadow_min_items:
        return judge(len(items), None, None, settings.shadow_min_items, settings.shadow_tolerance)
    if not candidate.checkpoint_path:
        raise ValueError("candidate has no checkpoint to score")

    candidate_accuracy = _score_candidate(candidate.checkpoint_path, items)
    champion_accuracy = None
    if champion is not None:
        champion_accuracy = sum(i.predicted_gloss == i.validated_label for i in items) / len(items)
    return judge(len(items), candidate_accuracy, champion_accuracy,
                 settings.shadow_min_items, settings.shadow_tolerance)

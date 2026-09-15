"""B12: builds a training manifest from validated review-queue data and invokes
`ml/train.py` / `ml/evaluate.py` as subprocesses — it does not reimplement training.

This is the "triggerable job" version described in SYSTEM_DESIGN.md B12: no live
3-day cron exists yet because there is no reviewer throughput to feed one. The
automatic cadence (FR-L5) is future work once real review volume exists — see
server/README.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.config import settings
from server.models import ReviewItem

MIN_SIGNERS_FOR_SPLIT = 3  # need at least train/val/test to each get one signer

# ml/*.py scripts print Arabic labels; Windows' cp1252 console default crashes on
# them unless the child process is forced to UTF-8 (server/README.md's Windows note).
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


class InsufficientDataError(RuntimeError):
    """Raised when there isn't enough validated, reviewed data yet to train on.

    This is the expected steady state right after launch, before any reviewer has
    worked the queue (FR-L3/L4) — not a bug.
    """


def _validated_items(db: Session, language: str) -> list[ReviewItem]:
    stmt = select(ReviewItem).where(
        ReviewItem.language == language,
        ReviewItem.status.in_(["confirmed", "corrected"]),
    )
    return list(db.scalars(stmt))


def export_validated_manifest(db: Session, language: str, out_dir: Path) -> tuple[Path, Path]:
    """Writes landmark .npy files + manifest.json + split.json for `ml/train.py`.

    Only rows a Deaf reviewer has confirmed or corrected are exported (FR-L3:
    "nothing enters training unreviewed") — `validated_label` (not the raw
    `predicted_gloss`) is the label written to the manifest.
    """
    items = _validated_items(db, language)
    signers = sorted({item.signer_id for item in items})
    if len(signers) < MIN_SIGNERS_FOR_SPLIT:
        raise InsufficientDataError(
            f"only {len(signers)} distinct signer(s) with validated '{language}' data "
            f"(need >= {MIN_SIGNERS_FOR_SPLIT} for a signer-independent train/val/test "
            f"split, DR-4) — reviewer throughput has not produced enough yet"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in items:
        label = item.validated_label or item.predicted_gloss
        rel = Path(item.signer_id) / label / f"{item.utterance_id}.npy"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / rel, np.asarray(item.landmarks, dtype=np.float32))
        manifest.append({
            "signer_id": item.signer_id,
            "sign_id": label,
            "label_ar": label,
            "label_en": label,
            "handedness": "right",
            "source": "kiosk-review-queue",
            "features": str(rel).replace("\\", "/"),
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Simple deterministic split: freeze the last two signers alphabetically for
    # val/test. A real deployment freezes DR-5's test signers at project start
    # instead of recomputing this every run; documented as a follow-up.
    split = {"val": signers[-2:-1], "test": signers[-1:]}
    split_path = out_dir / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    return manifest_path, split_path


def run_train(manifest_path: Path, split_path: Path, out_checkpoint: Path, epochs: int = 40,
              extra_args: list[str] | None = None) -> None:
    cmd = [
        settings.python_executable, str(settings.ml_dir / "train.py"),
        str(manifest_path), "--split", str(split_path),
        "--out", str(out_checkpoint), "--epochs", str(epochs),
    ] + (extra_args or [])
    subprocess.run(cmd, check=True, cwd=str(settings.ml_dir.parent), env=_SUBPROCESS_ENV)


def run_evaluate(checkpoint_path: Path, manifest_path: Path, split_path: Path,
                  report_path: Path) -> dict:
    cmd = [
        settings.python_executable, str(settings.ml_dir / "evaluate.py"),
        str(checkpoint_path), str(manifest_path), "--split", str(split_path),
        "--report", str(report_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(settings.ml_dir.parent), env=_SUBPROCESS_ENV)
    return json.loads(report_path.read_text(encoding="utf-8"))


def measure_latency_ms(checkpoint_path: Path) -> float:
    """One forward pass, timed — a dev-machine stand-in for an on-device benchmark.

    Real latency validation happens on the Jetson (NFR-P1); this gives the gate
    something concrete to compare against `latency_budget_ms` in the meantime.
    """
    import torch

    sys.path.insert(0, str(settings.ml_dir.parent))
    from ml.model import SignTransformer  # noqa: E402

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SignTransformer(input_dim=ckpt["input_dim"], num_classes=len(ckpt["labels"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 64, ckpt["input_dim"])
    with torch.no_grad():
        model(dummy)  # warm-up
        start = time.perf_counter()
        for _ in range(20):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / 20) * 1000

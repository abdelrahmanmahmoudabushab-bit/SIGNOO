"""B12: builds a training manifest from validated review-queue data and invokes
`ml/train.py` / `ml/evaluate.py` as subprocesses — it does not reimplement training.

This is the "triggerable job" version described in SYSTEM_DESIGN.md B12: no live
3-day cron exists yet because there is no reviewer throughput to feed one. The
automatic cadence (FR-L5) is future work once real review volume exists — see
server/README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.config import settings
from server.gate import evaluate_gate
from server.models import ModelVersion, ReviewItem

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


def load_frozen_split(language: str, path: Path | None = None) -> dict | None:
    path = path or settings.frozen_split_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(language)


def freeze_split(language: str, val: list[str], test: list[str], path: Path | None = None) -> dict:
    """Records the benchmark signers for a language, once. They are never reassigned:
    a benchmark that moves as signers arrive stops measuring progress (DR-5, DR-7a).
    """
    path = path or settings.frozen_split_path
    path.parent.mkdir(parents=True, exist_ok=True)
    everything = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    everything[language] = {"val": sorted(val), "test": sorted(test)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(everything, indent=2))
    os.replace(tmp, path)
    return everything[language]


def export_validated_manifest(db: Session, language: str, out_dir: Path,
                              frozen_path: Path | None = None) -> tuple[Path, Path]:
    """Writes landmark .npy files + manifest.json + split.json for `ml/train.py`.

    Only rows a Deaf reviewer has confirmed or corrected are exported (FR-L3:
    "nothing enters training unreviewed") — `validated_label` (not the raw
    `predicted_gloss`) is the label written to the manifest.

    The first export for a language freezes its validation and test signers; every
    later export keeps them and sends any newly arrived signer to train.
    """
    # Rows with no signer identity cannot be placed in a signer-independent split;
    # pooling them as one "unknown" signer would be a fake signer, so they stay out.
    items = [i for i in _validated_items(db, language) if i.signer_id != "unknown"]
    signers = sorted({item.signer_id for item in items})
    if len(signers) < MIN_SIGNERS_FOR_SPLIT:
        raise InsufficientDataError(
            f"only {len(signers)} distinct signer(s) with validated '{language}' data "
            f"(need >= {MIN_SIGNERS_FOR_SPLIT} for a signer-independent train/val/test "
            f"split, DR-4) — reviewer throughput has not produced enough yet"
        )

    frozen = load_frozen_split(language, frozen_path)
    if frozen is None:
        frozen = freeze_split(language, signers[-2:-1], signers[-1:], frozen_path)
    benchmark = set(frozen["val"]) | set(frozen["test"])
    missing = benchmark - set(signers)
    if missing:
        raise InsufficientDataError(
            f"frozen benchmark signer(s) {sorted(missing)} no longer have validated data; "
            f"refusing to train against a changed benchmark"
        )
    if not set(signers) - benchmark:
        raise InsufficientDataError(
            "every signer with validated data is a frozen benchmark signer; "
            "training needs at least one other signer"
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

    split = {"val": frozen["val"], "test": frozen["test"]}
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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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


class ChampionNotFound(LookupError):
    pass


def code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(settings.ml_dir.parent),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def find_champion(db: Session, language: str) -> ModelVersion | None:
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.language == language, ModelVersion.status == "promoted")
        .order_by(ModelVersion.promoted_at.desc())
    )
    return db.scalars(stmt).first()


def train_candidate(db: Session, language: str, epochs: int = 40,
                    champion_version: str | None = None) -> ModelVersion:
    """Exports reviewed data, trains, evaluates on the frozen benchmark, runs the gate
    and records the result as a `candidate`. Never promotes: promotion is a separate,
    human-triggered step that needs the shadow evaluation and a passed gate.

    Raises InsufficientDataError when there is not yet enough reviewed data, and
    ChampionNotFound for an explicit champion that does not exist.
    """
    now = datetime.now(timezone.utc)
    commit = code_commit()
    version = f"{now.strftime('%Y-%m-%dT%H:%M:%S')}Z-{commit}"
    run_dir = settings.training_runs_dir / version

    manifest_path, split_path = export_validated_manifest(db, language, run_dir / "data")

    checkpoint_path = run_dir / "checkpoint.pt"
    run_train(manifest_path, split_path, checkpoint_path, epochs=epochs)
    metrics = run_evaluate(checkpoint_path, manifest_path, split_path, run_dir / "eval_report.json")
    latency_ms = measure_latency_ms(checkpoint_path)

    if champion_version is None:
        champion = find_champion(db, language)
    else:
        champion = db.get(ModelVersion, champion_version)
        if champion is None:
            raise ChampionNotFound(champion_version)

    verdict = evaluate_gate(
        metrics, champion.metrics if champion else None,
        candidate_latency_ms=latency_ms, latency_budget_ms=settings.latency_budget_ms,
        shadow_run_complete=False,
    )

    model_version = ModelVersion(
        version=version, language=language, code_commit=commit,
        data_snapshot="manifest-sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        metrics=metrics, gate=verdict.as_dict(), status="candidate",
        checkpoint_path=str(checkpoint_path),
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return model_version

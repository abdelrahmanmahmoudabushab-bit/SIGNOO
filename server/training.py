"""B12: builds a training manifest from validated review-queue data and invokes
`ml/train.py` / `ml/evaluate.py` as subprocesses — it does not reimplement training.

A run is started either by `server/scheduler.py` (the automatic 3-day cadence, FR-L5) or
by an engineer through `POST /training/trigger`. Both go through `train_candidate`, which
holds a per-language lock for the length of the run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.config import settings
from server.gate import evaluate_gate
from server.models import ModelVersion, ReviewItem, TrainingLock
from server.schemas import SAFE_ID_PATTERN, SAFE_LABEL_PATTERN

logger = logging.getLogger("server.training")

MIN_SIGNERS_FOR_SPLIT = 3  # need at least train/val/test to each get one signer
EXPORT_BATCH = 200         # landmark arrays loaded per query while writing the export

# ml/*.py scripts print Arabic labels; Windows' cp1252 console default crashes on
# them unless the child process is forced to UTF-8 (server/README.md's Windows note).
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


class InsufficientDataError(RuntimeError):
    """Raised when there isn't enough validated, reviewed data yet to train on.

    This is the expected steady state right after launch, before any reviewer has
    worked the queue (FR-L3/L4) — not a bug.
    """


class TrainingInProgress(RuntimeError):
    """Another run already holds this language's training lock."""


@contextmanager
def training_lock(db: Session, language: str):
    """Only one training run per language at a time, across every worker process.

    The automatic scheduler runs in each server worker and an engineer can also trigger a
    run by hand. Two runs of one language would both be gated against the same champion
    and could both end up as candidates; one waiting on the other would make a request
    hang for minutes. So the second one is refused immediately instead.
    """
    now = datetime.now(timezone.utc)
    lease = timedelta(minutes=settings.training_lease_minutes)
    held = db.get(TrainingLock, language)
    if held is not None:
        if now - held.acquired_at.replace(tzinfo=timezone.utc) < lease:
            raise TrainingInProgress(f"a '{language}' training run is already in progress")
        # Older than any run should last: its process died without releasing it.
        logger.warning("training_lock_taken_over language=%s held_by=%s", language, held.holder)
        db.delete(held)
        db.flush()
    holder = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    db.add(TrainingLock(language=language, holder=holder, acquired_at=now))
    try:
        db.commit()
    except IntegrityError:    # two processes inserted at once; the other one has it
        db.rollback()
        raise TrainingInProgress(f"a '{language}' training run is already in progress") from None
    try:
        yield
    finally:
        db.rollback()         # a failed run may have left the session mid-transaction
        mine = db.get(TrainingLock, language)
        if mine is not None and mine.holder == holder:
            db.delete(mine)
            db.commit()


def is_enrolled_signer(signer_id: str) -> bool:
    """A named person who signed on purpose (a data-collection session), as opposed to an
    anonymous kiosk visit. Kiosk visits are `visitor-<hex>`, one id per visit, so the same
    person returning is a new id: they are good training data but cannot be told apart
    from each other, which makes them unusable as benchmark signers (see
    `export_validated_manifest`)."""
    return signer_id != "unknown" and not signer_id.startswith("visitor-")


def _validated_rows(db: Session, language: str):
    """Identifiers of the validated items, without their landmark arrays. Those are the
    bulk of each row and are fetched in batches when written out."""
    stmt = select(
        ReviewItem.id, ReviewItem.signer_id, ReviewItem.utterance_id,
        ReviewItem.validated_label, ReviewItem.predicted_gloss,
    ).where(
        ReviewItem.language == language,
        ReviewItem.status.in_(["confirmed", "corrected"]),
    )
    return list(db.execute(stmt))


def _export_path(item: ReviewItem, out_root: Path) -> Path | None:
    """Where this item's clip is written, relative to the export directory, or None
    when its identifiers are not plain path components. Ingest validates them, but
    rows already in the database predate that check, and a write that escapes the
    run directory is refused here regardless of how the row got in."""
    label = item.validated_label or item.predicted_gloss
    if not (re.fullmatch(SAFE_ID_PATTERN, item.signer_id) and re.fullmatch(SAFE_ID_PATTERN, item.utterance_id)
            and re.fullmatch(SAFE_LABEL_PATTERN, label)):
        return None
    rel = Path(item.signer_id) / label / f"{item.utterance_id}.npy"
    if not (out_root / rel).resolve().is_relative_to(out_root):
        return None
    return rel


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
    anonymous = sorted(s for s in [*val, *test] if not is_enrolled_signer(s))
    if anonymous:
        raise ValueError(f"benchmark signers must be enrolled people, not anonymous visits: {anonymous}")
    if set(val) & set(test):
        raise ValueError(f"a signer cannot be in both validation and test: {sorted(set(val) & set(test))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    everything = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    everything[language] = {"val": sorted(val), "test": sorted(test)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(everything, indent=2))
    os.replace(tmp, path)
    return everything[language]


class BenchmarkAlreadyFrozen(ValueError):
    pass


def freeze_benchmark(db: Session, language: str, val: list[str], test: list[str],
                     path: Path | None = None) -> dict:
    """Deliberately freezes a language's benchmark signers (an engineer's decision, not an
    accident of which visitors arrived first). Refuses to replace one that exists: a
    benchmark that moves stops measuring progress (DR-5, DR-7a). Each named signer must
    already have reviewed data, so the benchmark is something that can be scored."""
    if load_frozen_split(language, path) is not None:
        raise BenchmarkAlreadyFrozen(f"the '{language}' benchmark is already frozen and cannot be changed")
    if not val or not test:
        raise ValueError("both validation and test need at least one signer")
    have = {row.signer_id for row in _validated_rows(db, language)}
    missing = sorted((set(val) | set(test)) - have)
    if missing:
        raise ValueError(f"no reviewed '{language}' data for signer(s): {missing}")
    return freeze_split(language, val, test, path)


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
    out_root = out_dir.resolve()
    items = []
    for item in _validated_rows(db, language):
        if item.signer_id == "unknown":
            continue
        if _export_path(item, out_root) is None:
            logger.warning("export_skipped_unsafe_item id=%s language=%s", item.id, language)
            continue
        items.append(item)
    signers = sorted({item.signer_id for item in items})
    if len(signers) < MIN_SIGNERS_FOR_SPLIT:
        raise InsufficientDataError(
            f"only {len(signers)} distinct signer(s) with validated '{language}' data "
            f"(need >= {MIN_SIGNERS_FOR_SPLIT} for a signer-independent train/val/test "
            f"split, DR-4) — reviewer throughput has not produced enough yet"
        )

    frozen = load_frozen_split(language, frozen_path)
    if frozen is None:
        # The benchmark is what every future model is judged against, forever. A kiosk
        # visit is its own "signer" (one id per visit), so the same person can appear as
        # several visitors and leak from training into the benchmark; and picking by
        # sorted random ids would make the benchmark an arbitrary pair of visits. Only a
        # person who signed on purpose and is named as such can anchor it.
        enrolled = [s for s in signers if is_enrolled_signer(s)]
        if len(enrolled) < 2:
            raise InsufficientDataError(
                f"the '{language}' benchmark is not frozen and needs two enrolled signers "
                f"(named people from a data-collection session, not anonymous kiosk visits); "
                f"{len(enrolled)} have validated data. Freeze it with POST /training/benchmark "
                f"once they do"
            )
        frozen = freeze_split(language, enrolled[-2:-1], enrolled[-1:], frozen_path)
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
    # Landmarks are the bulk of every row; load them a batch at a time so memory use does
    # not grow with the size of the reviewed set.
    for start in range(0, len(items), EXPORT_BATCH):
        batch = items[start:start + EXPORT_BATCH]
        landmarks = {row.id: row.landmarks for row in db.execute(
            select(ReviewItem.id, ReviewItem.landmarks).where(ReviewItem.id.in_([i.id for i in batch]))
        )}
        for item in batch:
            label = item.validated_label or item.predicted_gloss
            rel = _export_path(item, out_root)
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / rel, np.asarray(landmarks[item.id], dtype=np.float32))
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
    from ml.model import build_model  # noqa: E402

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt)

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


class ChampionLanguageMismatch(ValueError):
    """An explicit champion belongs to another language; its metrics come from a
    different benchmark and say nothing about this language's candidate."""


def code_commit() -> str:
    # A container image has no .git; the build stamps the commit it was built from.
    stamped = os.environ.get("SIGNOO_CODE_COMMIT", "").strip()
    if stamped:
        return stamped
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

    Raises InsufficientDataError when there is not yet enough reviewed data,
    ChampionNotFound for an explicit champion that does not exist,
    ChampionLanguageMismatch for one that belongs to another language, and
    TrainingInProgress when this language is already being trained.
    """
    # Resolved before any training so a bad champion fails in milliseconds, not minutes.
    if champion_version is None:
        champion = find_champion(db, language)
    else:
        champion = db.get(ModelVersion, champion_version)
        if champion is None:
            raise ChampionNotFound(champion_version)
        if champion.language != language:
            raise ChampionLanguageMismatch(
                f"champion '{champion_version}' is a '{champion.language}' model; a '{language}' "
                f"candidate can only be gated against a '{language}' champion"
            )

    with training_lock(db, language):
        now = datetime.now(timezone.utc)
        commit = code_commit()
        version = f"{now.strftime('%Y-%m-%dT%H:%M:%S')}Z-{commit}"
        run_dir = settings.training_runs_dir / version

        manifest_path, split_path = export_validated_manifest(db, language, run_dir / "data")

        checkpoint_path = run_dir / "checkpoint.pt"
        run_train(manifest_path, split_path, checkpoint_path, epochs=epochs)
        metrics = run_evaluate(checkpoint_path, manifest_path, split_path, run_dir / "eval_report.json")
        latency_ms = measure_latency_ms(checkpoint_path)

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
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            training_started_at=now, champion_version=champion.version if champion else None,
        )
        db.add(model_version)
        db.commit()
        db.refresh(model_version)
        return model_version

"""Configuration, resolved once at startup from the environment.

Nothing downstream reads an environment variable directly — see
`signoo-engineering`'s rule that config enters at the edge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Postgres in normal operation (docker-compose service `db`); falls back to a
    # local SQLite file when DATABASE_URL is unset, which is how this project's own
    # dev sandbox (no Docker available) was exercised. Both go through SQLAlchemy so
    # nothing else in the code needs to know which one is live.
    database_url: str = os.environ.get(
        "SIGNOO_DATABASE_URL", f"sqlite:///{REPO_ROOT / 'server' / 'signoo_dev.db'}"
    )
    jwt_secret: str = os.environ.get("SIGNOO_JWT_SECRET", "dev-secret-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = int(os.environ.get("SIGNOO_JWT_EXPIRE_MINUTES", "480"))

    vocab_path: Path = field(default_factory=lambda: REPO_ROOT / "core" / "vocab.json")
    clip_library_path: Path = field(
        default_factory=lambda: REPO_ROOT / "core" / "clip_library.json"
    )
    # File-based sharing with kiosk/'s B7 for this dev-simulation phase (see
    # server/README.md "Knowledge base sharing" for why this is a file, not just an API).
    kiosk_kb_path: Path = field(
        default_factory=lambda: REPO_ROOT / "kiosk" / "data" / "knowledge_base.json"
    )

    ml_dir: Path = field(default_factory=lambda: REPO_ROOT / "ml")
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")
    training_runs_dir: Path = field(default_factory=lambda: REPO_ROOT / "server" / "runs")

    python_executable: str = os.environ.get("SIGNOO_PYTHON", "python")

    # NFR-P1: end-of-sign-to-word-on-screen budget the promotion gate's latency
    # criterion (DR-6 #6) checks candidates against.
    latency_budget_ms: float = 300.0

    # The validation/test signers are chosen once per language and then frozen here, so
    # the benchmark does not drift as new signers arrive (DR-5, DR-7a).
    frozen_split_path: Path = field(
        default_factory=lambda: REPO_ROOT / "server" / "runs" / "frozen_split.json"
    )

    # FR-L5: candidate training runs automatically on this cadence. Promotion never
    # does; it always goes through the gate and an engineer (server/scheduler.py).
    auto_train_enabled: bool = os.environ.get("SIGNOO_AUTO_TRAIN", "1") == "1"
    auto_train_interval_hours: float = float(os.environ.get("SIGNOO_AUTO_TRAIN_HOURS", "72"))
    auto_train_poll_minutes: float = float(os.environ.get("SIGNOO_AUTO_TRAIN_POLL_MIN", "60"))
    auto_train_languages: tuple[str, ...] = ("liu",)

    # FR-L7 shadow evaluation: how many reviewed live items, newer than the candidate,
    # it must be scored on, and how far below the champion it may fall.
    shadow_min_items: int = int(os.environ.get("SIGNOO_SHADOW_MIN_ITEMS", "30"))
    shadow_tolerance: float = 0.005


settings = Settings()

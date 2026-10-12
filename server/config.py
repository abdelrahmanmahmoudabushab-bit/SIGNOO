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
    # "dev" relaxes the startup checks below; anything else is treated as production.
    environment: str = os.environ.get("SIGNOO_ENV", "production")
    jwt_secret: str = os.environ.get("SIGNOO_JWT_SECRET", "dev-secret-change-me")
    # Shared secret kiosks send as `X-Kiosk-Token` on ingest/register/heartbeat. Blank
    # means no check, which production startup refuses (see server/main.py).
    kiosk_token: str = os.environ.get("SIGNOO_KIOSK_TOKEN", "")
    # "shared": every kiosk sends `kiosk_token` itself (simple; one stolen kiosk exposes
    # the whole fleet). "per_kiosk": each kiosk sends HMAC-SHA256(kiosk_token, kiosk_id),
    # so a token only works for the one kiosk id it was issued for and `kiosk_token`
    # stays on the server as the issuing key (server/scripts/kiosk_token.py prints them).
    kiosk_token_mode: str = os.environ.get("SIGNOO_KIOSK_TOKEN_MODE", "shared")
    # Browser origins allowed to call the API (the console's address). Empty = same-origin only.
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.environ.get("SIGNOO_CORS_ORIGINS", "").split(",") if o.strip()
    )
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

    # The same file the kiosk reads: which files each language's engine owns.
    languages_path: Path = field(default_factory=lambda: REPO_ROOT / "core" / "languages.json")

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
    # Each language trains its own candidate from its own reviewed data and is gated
    # against its own champion and its own frozen benchmark; they never mix.
    auto_train_languages: tuple[str, ...] = tuple(
        code.strip() for code in os.environ.get("SIGNOO_AUTO_TRAIN_LANGUAGES", "liu").split(",") if code.strip()
    )

    # A training run holds a per-language lock for at most this long; a lock older than
    # that was left by a process that died mid-run and may be taken over.
    training_lease_minutes: float = float(os.environ.get("SIGNOO_TRAINING_LEASE_MIN", "180"))

    # Uploaded clips nobody has reviewed are deleted after this many days, so a backlog
    # cannot keep a visitor's landmarks indefinitely. Reviewed clips (confirmed/corrected)
    # are kept: they are the training set. 0 disables the purge.
    retention_days: int = int(os.environ.get("SIGNOO_RETENTION_DAYS", "30"))

    # FR-L7 shadow evaluation: how many reviewed live items, newer than the candidate,
    # it must be scored on, and how far below the champion it may fall.
    shadow_min_items: int = int(os.environ.get("SIGNOO_SHADOW_MIN_ITEMS", "30"))
    shadow_tolerance: float = 0.005


DEV_JWT_SECRET = "dev-secret-change-me"


def startup_problems(s: Settings) -> list[str]:
    """Insecure defaults that must never reach a real deployment. Empty in `dev`."""
    if s.environment == "dev":
        return []
    problems = []
    if s.jwt_secret == DEV_JWT_SECRET or len(s.jwt_secret) < 32:
        problems.append("SIGNOO_JWT_SECRET must be set to a random value of at least 32 characters")
    if not s.kiosk_token:
        problems.append("SIGNOO_KIOSK_TOKEN must be set so only your kiosks can upload and register")
    if s.kiosk_token_mode not in ("shared", "per_kiosk"):
        problems.append(f"SIGNOO_KIOSK_TOKEN_MODE must be 'shared' or 'per_kiosk', not {s.kiosk_token_mode!r}")
    return problems


settings = Settings()

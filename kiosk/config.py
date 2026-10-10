"""Kiosk configuration, resolved once at startup (signoo-engineering: config enters
at the edge). Every value is overridable by an environment variable so the same
image runs in dev (webcam-less CI box) and on a real kiosk without a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # B4 Recognition
    model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SIGNOO_MODEL_PATH", str(REPO_ROOT / "models" / "signoo.pt"))
        )
    )
    # Below this softmax probability, B4 emits Hypothesis(gloss=None) — a rejection,
    # never a guess (Requirements §6.3).
    reject_threshold: float = field(
        default_factory=lambda: _env_float("SIGNOO_REJECT_THRESHOLD", 0.55)
    )
    # Between reject_threshold and confident_threshold: shown as touch-target
    # candidates (FR-K8), never auto-committed.
    confident_threshold: float = field(
        default_factory=lambda: _env_float("SIGNOO_CONFIDENT_THRESHOLD", 0.80)
    )
    top_k_alternatives: int = field(default_factory=lambda: _env_int("SIGNOO_TOP_K", 3))

    # Which sign languages exist, and each one's model + vocabulary (kiosk/languages.py).
    languages_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SIGNOO_LANGUAGES_PATH", str(REPO_ROOT / "core" / "languages.json"))
        )
    )

    # B3 Segmentation — thresholds as config, per SYSTEM_DESIGN.md B3, never guessed
    # in code. Tuned on synthetic data for now; real values need real recordings.
    segmentation_pause_ms: int = field(
        default_factory=lambda: _env_int("SIGNOO_PAUSE_MS", 500)
    )
    segmentation_motion_threshold: float = field(
        default_factory=lambda: _env_float("SIGNOO_MOTION_THRESHOLD", 0.02)
    )
    segmentation_hands_absent_ms: int = field(
        default_factory=lambda: _env_int("SIGNOO_HANDS_ABSENT_MS", 800)
    )
    min_utterance_frames: int = field(
        default_factory=lambda: _env_int("SIGNOO_MIN_UTTERANCE_FRAMES", 6)
    )

    # B7 Dialogue
    knowledge_base_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SIGNOO_KB_PATH", str(REPO_ROOT / "kiosk" / "data" / "knowledge_base.json")
            )
        )
    )
    kb_similarity_floor: float = field(
        default_factory=lambda: _env_float("SIGNOO_KB_SIMILARITY_FLOOR", 0.2)
    )
    claude_model: str = field(
        default_factory=lambda: os.environ.get("SIGNOO_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    )
    claude_timeout_s: float = field(
        default_factory=lambda: _env_float("SIGNOO_CLAUDE_TIMEOUT_S", 4.0)
    )
    dialogue_cloud_enabled: bool = field(
        default_factory=lambda: _env_bool("SIGNOO_DIALOGUE_CLOUD", True)
    )

    # B6 Output
    tts_enabled: bool = field(default_factory=lambda: _env_bool("SIGNOO_TTS_ENABLED", True))

    # B8 Sign playback
    clip_library_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SIGNOO_CLIP_LIBRARY_PATH", str(REPO_ROOT / "core" / "clip_library.json"))
        )
    )
    vocab_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SIGNOO_VOCAB_PATH", str(REPO_ROOT / "core" / "vocab.json"))
        )
    )

    # B9 Session/telemetry
    idle_timeout_s: float = field(
        default_factory=lambda: _env_float("SIGNOO_IDLE_TIMEOUT_S", 45.0)
    )

    # B10 Data capture
    pending_uploads_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SIGNOO_PENDING_UPLOADS_DIR", str(REPO_ROOT / "kiosk" / "data" / "pending_uploads")
            )
        )
    )
    kiosk_id: str = field(default_factory=lambda: os.environ.get("SIGNOO_KIOSK_ID", "kiosk-dev-01"))
    # Base URL of server/ (e.g. http://localhost:8001). Blank = sessions stay queued
    # on disk and nothing leaves the device.
    server_url: str = field(default_factory=lambda: os.environ.get("SIGNOO_SERVER_URL", ""))
    # Shared secret the server requires on upload/register/heartbeat (X-Kiosk-Token).
    kiosk_token: str = field(default_factory=lambda: os.environ.get("SIGNOO_KIOSK_TOKEN", ""))
    heartbeat_interval_s: float = field(
        default_factory=lambda: _env_float("SIGNOO_HEARTBEAT_INTERVAL_S", 30.0)
    )

    # Perception — every second captured frame, per SYSTEM_DESIGN.md B2 budget.
    perception_stride: int = field(default_factory=lambda: _env_int("SIGNOO_PERCEPTION_STRIDE", 2))


CONFIG = Config()

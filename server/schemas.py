"""Pydantic request/response shapes, matching docs/API_CONTRACTS.md exactly.

`kiosk/` and `console/` are built in parallel against the same contracts file — do
not rename or reshape fields here without updating that doc first.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

FRAME_DIM = 203          # core.features.FRAME_DIM: one landmark row per frame

# Identifiers become directory and file names in the training export
# (server/training.py), so they are restricted to one safe path component here, at
# the boundary. The leading alphanumeric rules out "." and "..".
SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
SAFE_LABEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
SafeId = Annotated[str, StringConstraints(pattern=SAFE_ID_PATTERN)]
SafeLabel = Annotated[str, StringConstraints(pattern=SAFE_LABEL_PATTERN)]
LanguageCode = Annotated[str, StringConstraints(pattern=r"^[a-z]{2,8}$")]

MAX_ITEMS_PER_UPLOAD = 200
MAX_FRAMES_PER_ITEM = 256    # the kiosk caps an utterance at 45 frames by default


# ── §4 session upload (kiosk B10 -> server B11) ─────────────────────────────────

class SessionItemIn(BaseModel):
    utterance_id: SafeId
    landmarks: list[list[float]] = Field(min_length=1, max_length=MAX_FRAMES_PER_ITEM)
    predicted_gloss: SafeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    correction: SafeLabel | None = None
    signer_id: SafeId = "unknown"

    @field_validator("landmarks")
    @classmethod
    def _rows_are_frames(cls, rows: list[list[float]]) -> list[list[float]]:
        if any(len(row) != FRAME_DIM for row in rows):
            raise ValueError(f"every landmark row must have exactly {FRAME_DIM} values")
        return rows


class SessionUploadIn(BaseModel):
    session_id: SafeId
    kiosk_id: SafeId
    language: LanguageCode
    consent_retention: bool
    started_at: str = Field(max_length=64)
    ended_at: str = Field(max_length=64)
    items: list[SessionItemIn] = Field(max_length=MAX_ITEMS_PER_UPLOAD)


class SessionUploadOut(BaseModel):
    session_upload_id: str
    items_queued: int


# ── B11 review ───────────────────────────────────────────────────────────────────

class ReviewItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    kiosk_id: str
    utterance_id: str
    language: str
    signer_id: str
    predicted_gloss: str
    confidence: float
    correction: str | None
    status: str
    validated_label: str | None
    captured_at: str
    priority_score: float = 0.0
    priority_reason: str = "routine"


class ReviewDecisionIn(BaseModel):
    # confirm: predicted_gloss is correct. correct: validated_label overrides it.
    # flag: send to a linguist, no label recorded yet.
    decision: str = Field(pattern="^(confirm|correct|flag)$")
    validated_label: SafeLabel | None = None
    note: str | None = Field(default=None, max_length=1000)


# ── §3 knowledge base (B14) ──────────────────────────────────────────────────────

class KBEntryIn(BaseModel):
    # Whose engine this answer belongs to; defaults to the original language so a client
    # written before per-language knowledge bases keeps working.
    language: LanguageCode = "liu"
    site_id: str = "default"
    keywords: list[str]
    answer_ar: str
    answer_en: str


class KBEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    language: str = "liu"
    site_id: str
    keywords: list[str]
    answer_ar: str
    answer_en: str
    updated_at: str
    unrenderable_words: list[str] = []


# ── B15 fleet ─────────────────────────────────────────────────────────────────────

class KioskRegisterIn(BaseModel):
    kiosk_id: str
    site_id: str = "default"
    language: str = "liu"


class KioskHeartbeatIn(BaseModel):
    # Every field is optional and, when omitted, leaves the kiosk's last-known value
    # alone (server/routers/fleet.py::heartbeat) rather than resetting it — a partial
    # heartbeat must not be able to mask a real overheating/throttling condition.
    current_model_version: str | None = None
    temperature_c: float | None = None
    throttled: bool | None = None
    uptime_s: float | None = None
    # Language code -> installed model version, one entry per language engine.
    model_versions: dict[str, str] | None = None


class KioskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kiosk_id: str
    site_id: str
    language: str
    current_model_version: str | None
    last_heartbeat_at: str | None
    temperature_c: float | None
    throttled: bool
    uptime_s: float | None
    online: bool
    model_versions: dict[str, str] = {}


class RolloutIn(BaseModel):
    model_version: str
    target_kiosk_ids: list[str]


class RolloutOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version: str
    target_kiosk_ids: list[str]
    status: str


# ── §6 model registry (B13) ──────────────────────────────────────────────────────

class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: str
    language: str
    code_commit: str
    data_snapshot: str
    metrics: dict
    gate: dict
    status: str
    checkpoint_path: str | None


class TrainingCurvePoint(BaseModel):
    epoch: int
    train_loss: float
    val_accuracy: float


class TrainingRunOut(BaseModel):
    """console/src/api/types.ts `TrainingRun`, plus `language`."""

    run_id: str
    language: str
    started_at: str
    finished_at: str | None
    status: str
    candidate_version: str | None
    champion_version: str | None
    curves: list[TrainingCurvePoint]


class TrainTriggerIn(BaseModel):
    language: str = "liu"
    epochs: int = 40
    champion_version: str | None = None


class BenchmarkIn(BaseModel):
    language: LanguageCode
    val: list[SafeId] = Field(min_length=1)
    test: list[SafeId] = Field(min_length=1)


class BenchmarkOut(BaseModel):
    language: str
    val: list[str]
    test: list[str]


# ── §5 auth ───────────────────────────────────────────────────────────────────────

class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class CreateUserIn(BaseModel):
    email: str
    password: str = Field(min_length=10)
    role: str


class UserOut(BaseModel):
    id: str
    email: str
    role: str

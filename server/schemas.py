"""Pydantic request/response shapes, matching docs/API_CONTRACTS.md exactly.

`kiosk/` and `console/` are built in parallel against the same contracts file — do
not rename or reshape fields here without updating that doc first.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── §4 session upload (kiosk B10 -> server B11) ─────────────────────────────────

class SessionItemIn(BaseModel):
    utterance_id: str
    landmarks: list[list[float]]
    predicted_gloss: str
    confidence: float
    correction: str | None = None
    signer_id: str = "unknown"


class SessionUploadIn(BaseModel):
    session_id: str
    kiosk_id: str
    language: str
    consent_retention: bool
    started_at: str
    ended_at: str
    items: list[SessionItemIn]


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
    validated_label: str | None = None
    note: str | None = None


# ── §3 knowledge base (B14) ──────────────────────────────────────────────────────

class KBEntryIn(BaseModel):
    site_id: str = "default"
    keywords: list[str]
    answer_ar: str
    answer_en: str


class KBEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
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


class TrainTriggerIn(BaseModel):
    language: str = "liu"
    epochs: int = 40
    champion_version: str | None = None


# ── §5 auth ───────────────────────────────────────────────────────────────────────

class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class CreateUserIn(BaseModel):
    email: str
    password: str
    role: str


class UserOut(BaseModel):
    id: str
    email: str
    role: str

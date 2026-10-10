"""B9 Session/telemetry: session state machine, structured logs, correlation id
(SYSTEM_DESIGN.md B9).

Every state is visible on screen (`deaf-kiosk-ux` skill) and every transition is
logged with the correlation id shared with `server/` (NFR-N3). Temperature and
throttle reporting returns "unavailable" rather than a fake number — this is a
Windows/Linux dev machine, not a Jetson, and lying about hardware telemetry is
worse than admitting we don't have it.
"""
from __future__ import annotations

import enum
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field


class JsonFormatter(logging.Formatter):
    """Stdlib logging with a JSON formatter (per the delegated plan), and UTF-8
    safe — Windows' console defaults to cp1252, which crashes on Arabic text and
    the → in an f-string the moment someone prints it directly. Routing everything
    through `json.dumps(..., ensure_ascii=False)` and a UTF-8 stream avoids that.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in payload and key not in (
                "args", "msg", "levelno", "levelname", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "name",
            ):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("signoo")
    if root.handlers:
        return
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")   # avoid the Windows cp1252 crash
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


class SessionState(str, enum.Enum):
    ATTRACT = "attract"        # idle screen, FR-K1 (the browser's "Hello" screen)
    LANGUAGE = "language"       # FR-K2
    CONSENT = "consent"         # FR-K3
    ACTIVE = "active"           # capturing/recognising/answering, FR-K4-K12 (Translate + Ask)
    ENDED = "ended"             # FR-K13 (the browser's "Thank you" screen)


@dataclass
class Session:
    """One visitor's session. Erased on return to idle (NFR-S3) — nothing here
    outlives the session object itself; `kiosk/main.py` drops the reference.
    """

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kiosk_id: str = "kiosk-dev-01"
    state: SessionState = SessionState.ATTRACT
    language: str | None = None
    consent_processing: bool = False
    consent_retention: bool = False
    started_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("signoo.kiosk.session"))

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def idle_for(self) -> float:
        return time.time() - self.last_activity_at

    def transition(self, new_state: SessionState, **extra) -> None:
        old = self.state
        self.state = new_state
        self.touch()
        self.logger.info(
            "session_transition",
            extra={"correlation_id": self.correlation_id, "from": old.value,
                   "to": new_state.value, **extra},
        )


def telemetry_snapshot() -> dict:
    """Latency/frame-rate fields are filled in by the caller per-connection; the
    hardware fields below are honestly "unavailable" on anything that is not a
    Jetson (SYSTEM_DESIGN.md B9), rather than a plausible-looking fake number.
    """
    return {
        "temperature_c": "unavailable",
        "throttled": "unavailable",
        "memory_mb": "unavailable",
    }

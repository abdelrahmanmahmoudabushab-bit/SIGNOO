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
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path


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


def read_temperature_c(root: str = "/sys/class/thermal") -> float | None:
    """Hottest thermal zone in degrees C (Jetson and most Linux boards), or None when
    the platform exposes none, as on a Windows dev machine."""
    temps = []
    for path in Path(root).glob("thermal_zone*/temp"):
        try:
            temps.append(int(path.read_text().strip()) / 1000)
        except (OSError, ValueError, TypeError):
            # TypeError: on the Jetson (Python 3.10) a thermal zone with no sensor behind
            # it can return nothing from read(), and the decoder fails with
            # "can't concat NoneType to bytes" instead of raising OSError. One dead zone
            # must not take /health and the heartbeat down with it.
            continue
    return max(temps) if temps else None


def read_memory_mb(path: str = "/proc/self/status") -> float | None:
    """This process's resident memory in MB, from the kernel (Jetson and any Linux), or
    None where there is no /proc, as on a Windows dev machine. NFR-M2 needs this number
    watched over days: a leak shows up on day nine, not in a demo."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)   # the kernel reports kB
    except (OSError, ValueError, IndexError):
        pass
    return None


class LiveStats:
    """What the pipeline is doing right now (B9: latency per stage, frame rate, interaction
    outcomes). One kiosk has one visitor, so one of these serves the whole process; it is
    written from the event loop and read by /health and the heartbeat thread, hence the lock.

    It holds counts and timings only: no landmarks, no words, nothing about who signed.
    """

    WINDOW_S = 5.0          # the tracker rate is the average over this many seconds

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._tracker_results: deque[float] = deque()
        self._frame_seconds: deque[float] = deque(maxlen=100)
        self._classify_ms: float | None = None
        self._outcomes: Counter = Counter()

    def note_tracker_result(self) -> None:
        now = self._clock()
        with self._lock:
            self._tracker_results.append(now)
            while self._tracker_results and now - self._tracker_results[0] > self.WINDOW_S:
                self._tracker_results.popleft()

    def note_frame(self, seconds: float) -> None:
        with self._lock:
            self._frame_seconds.append(seconds)

    def note_outcome(self, status: str, classify_seconds: float | None = None) -> None:
        with self._lock:
            self._outcomes[status] += 1
            if classify_seconds is not None:
                self._classify_ms = round(classify_seconds * 1000, 1)

    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            recent = [t for t in self._tracker_results if now - t <= self.WINDOW_S]
            frames = list(self._frame_seconds)
            return {
                # None, not 0, when nobody is in front of the camera: "no visitor" must not
                # read as "the tracker has stalled".
                "tracker_fps": round(len(recent) / self.WINDOW_S, 1) if recent else None,
                "frame_ms": round(sum(frames) / len(frames) * 1000, 1) if frames else None,
                "classify_ms": self._classify_ms,
                "outcomes": dict(self._outcomes),
            }


def telemetry_snapshot(live: LiveStats | None = None) -> dict:
    """Temperature and memory are real where the platform reports them and honestly
    "unavailable" where it does not (SYSTEM_DESIGN.md B9), never a plausible fake number.
    `live` adds the pipeline's own figures: tracker rate, stage latencies, outcomes.
    """
    temperature = read_temperature_c()
    memory = read_memory_mb()
    snapshot = {
        "temperature_c": temperature if temperature is not None else "unavailable",
        "throttled": "unavailable",
        "memory_mb": memory if memory is not None else "unavailable",
    }
    if live is not None:
        snapshot.update(live.snapshot())
    return snapshot

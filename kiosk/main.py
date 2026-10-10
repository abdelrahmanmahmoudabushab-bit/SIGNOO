"""SIGNOO kiosk FastAPI process (B1-B10) — SYSTEM_DESIGN.md §2.

Run:

    .venv/Scripts/python.exe -m uvicorn kiosk.main:app --reload --port 8000

then open http://localhost:8000/ — this process serves both the API/WebSocket and
the static frontend (`index.html`, `assets/*.js`), so there is nothing else to run.

## Wire protocol (browser <-> `/ws/session`)

JSON text frames both ways. Camera capture and rendering stay in the browser
(`getUserMedia`); every frame is base64 JPEG, sent to this backend, which is where
B2 Perception (MediaPipe Holistic), B3 Segmentation, B4 Recognition, B5 Assembly,
B7 Dialogue and B8 Playback all run — the browser only draws what it's told.

Client -> server:

    {"type": "language", "lang": "liu"}
    {"type": "consent", "processing": true, "retention": false}
    {"type": "frame", "ts_ms": 12345, "data": "<base64 jpeg, optionally a data: URL>"}
    {"type": "control", "action": "backspace" | "clear" | "end_session" | "extend"}
    {"type": "select_candidate", "gloss_id": "marhaba"}
    {"type": "speak", "lang": "ar" | "en"}
    {"type": "ask"}
    {"type": "ping"}

Server -> client:

    {"type": "state", "session_state": "attract|language|consent|active|ended",
     "correlation_id": "..."}
        # attract = the browser's Hello screen; ended = its Thank-you screen.
        # Presence auto-detect and the thank-you animation are client-side only.
    {"type": "landmarks", "pose": [[x,y,z],...] | null, "left": ... | null, "right": ... | null}
    {"type": "framing", "ok": true|false, "reason": "no_body"|"hands_out_of_frame"|null}
    {"type": "hypothesis", "status": "committed"|"candidates"|"rejected",
     "gloss": "marhaba"|null, "confidence": 0.9, "alternatives": [["marhaba", 0.9], ...]}
    {"type": "rendered", "gloss_seq": [...], "ar": "...", "en": "...", "uncertain": []}
    {"type": "tts_done", "lang": "ar", "ok": true}
    {"type": "answer", "ar": "...", "en": "...", "source": "cloud"|"local_fallback",
     "refused": false, "unrenderable_words": [...]}
    {"type": "playback", "words": [{"gloss_id":..., "ar":..., "en":..., "has_clip": false}, ...]}
    {"type": "idle_warning", "seconds_left": 10}
    {"type": "session_reset"}
    {"type": "error", "code": "...", "message": "..."}
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from core.contracts import Hypothesis
from kiosk.assembly import Assembler, Vocabulary
from kiosk.config import CONFIG, REPO_ROOT
from kiosk.data_capture import (
    CapturedItem, SessionCapture, flush_pending_uploads, write_pending_upload,
)
from kiosk.dialogue import Dialogue, KnowledgeBase
from kiosk.output import TTSEngine
from kiosk.perception import HolisticPerception, PerceptionUnavailable, decode_base64_frame
from kiosk.playback import ClipLibrary
from kiosk.languages import LanguageRegistry
from kiosk.segmentation import Segmenter
from kiosk.session import Session, SessionState, configure_logging, telemetry_snapshot
import json

configure_logging()
logger = logging.getLogger("signoo.kiosk.main")

app = FastAPI(title="SIGNOO kiosk")

# ── shared, process-wide singletons (models load once at startup, per SYSTEM_DESIGN.md §4.2) ──
_vocab = Vocabulary(CONFIG.vocab_path)
_clip_library = ClipLibrary(CONFIG.clip_library_path, _vocab)
_kb = KnowledgeBase(CONFIG.knowledge_base_path)
_tts = TTSEngine(enabled=CONFIG.tts_enabled)

# One model + vocabulary per sign language (core/languages.json). A corrupt or missing
# checkpoint leaves that one language unavailable (SYSTEM_DESIGN.md §6) rather than
# stopping the process: the frontend still needs to serve a clear out-of-service state.
DEFAULT_LANGUAGE = "liu"
registry = LanguageRegistry(
    CONFIG.languages_path, REPO_ROOT,
    CONFIG.reject_threshold, CONFIG.confident_threshold, CONFIG.top_k_alternatives,
    model_overrides={DEFAULT_LANGUAGE: CONFIG.model_path},   # keeps SIGNOO_MODEL_PATH working
)

_background_tasks: set[asyncio.Task] = set()


def _schedule_flush() -> None:
    """Delivers queued sessions to server/ off the event loop, never blocking a
    visitor; a no-op unless SIGNOO_SERVER_URL is set."""
    if not CONFIG.server_url:
        return
    task = asyncio.create_task(asyncio.to_thread(
        flush_pending_uploads, CONFIG.pending_uploads_dir, CONFIG.server_url
    ))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.on_event("startup")
async def _flush_on_startup() -> None:
    _schedule_flush()   # retry anything a previous run could not deliver


@app.get("/health")
def health() -> dict:
    default = registry.get(DEFAULT_LANGUAGE)
    return {
        "status": "ok" if default is not None and default.ready else "degraded",
        "model_loaded": default is not None and default.ready,
        "labels": default.summary()["labels"] if default else 0,
        "languages": registry.summary(),
        "telemetry": telemetry_snapshot(),
    }


@app.get("/languages")
def languages() -> list[dict]:
    """Which language cards can be pressed: ready, status, and where its vocabulary is."""
    return registry.summary()


class KioskConnection:
    """One visitor's session: owns the per-connection perception/segmentation
    state and dispatches wire-protocol messages to the right block.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session = Session(kiosk_id=CONFIG.kiosk_id)
        self.pack = registry.get(DEFAULT_LANGUAGE)
        self.assembler = Assembler(self._vocab_for(self.pack))
        self.dialogue = Dialogue(
            _kb, CONFIG.kb_similarity_floor, CONFIG.claude_model, CONFIG.claude_timeout_s,
            CONFIG.dialogue_cloud_enabled, _vocab, _clip_library._entries,
        )
        self.segmenter = Segmenter(
            pause_ms=CONFIG.segmentation_pause_ms,
            motion_threshold=CONFIG.segmentation_motion_threshold,
            hands_absent_ms=CONFIG.segmentation_hands_absent_ms,
            min_frames=CONFIG.min_utterance_frames,
        )
        self.capture = SessionCapture(
            session_id=self.session.correlation_id,
            kiosk_id=CONFIG.kiosk_id,
            language="liu",
            consent_retention=False,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.perception: HolisticPerception | None = None
        self.pending_candidates: list[tuple[str, float]] | None = None

    @staticmethod
    def _vocab_for(pack) -> Vocabulary:
        return pack.vocab if pack is not None and pack.vocab is not None else _vocab

    async def send(self, message: dict) -> None:
        await self.ws.send_text(json.dumps(message, ensure_ascii=False))

    async def send_state(self) -> None:
        await self.send({
            "type": "state",
            "session_state": self.session.state.value,
            "correlation_id": self.session.correlation_id,
        })

    async def handle(self, message: dict) -> None:
        msg_type = message.get("type")
        if msg_type not in ("ping", "frame"):
            # Neither a keep-alive ping nor a bare camera frame is activity (FR-K13):
            # the browser streams frames for as long as the tab is open, so counting
            # them would mean a visitor who walked away never times out. A frame
            # counts only when it shows hands (see _handle_frame).
            self.session.touch()

        if msg_type == "language":
            code = message.get("lang")
            pack = registry.get(code)
            if pack is None or not pack.ready:
                # Never fall back to another language's model: a visitor who picked
                # American must not be recognised by the Jordanian model, or vice versa.
                await self.send({"type": "error", "code": "language_unavailable",
                                 "message": f"{code!r} has no trained model yet"})
                return
            self.pack = pack
            self.assembler = Assembler(self._vocab_for(pack))
            self.session.language = code
            self.capture.language = code
            self.session.transition(SessionState.LANGUAGE, language=self.session.language)
            await self.send_state()

        elif msg_type == "consent":
            self.session.consent_processing = bool(message.get("processing"))
            self.session.consent_retention = bool(message.get("retention"))
            self.capture.consent_retention = self.session.consent_retention
            if self.session.consent_processing:
                self._start_active()
                self.session.transition(SessionState.ACTIVE)
                await self.send_state()
            else:
                self.session.transition(SessionState.CONSENT)
                await self.send_state()

        elif msg_type == "frame":
            await self._handle_frame(message)

        elif msg_type == "control":
            await self._handle_control(message.get("action"))

        elif msg_type == "select_candidate":
            await self._handle_select_candidate(message.get("gloss_id"))

        elif msg_type == "speak":
            await self._handle_speak(message.get("lang", "ar"))

        elif msg_type == "ask":
            await self._handle_ask()

        elif msg_type == "ping":
            await self.send({"type": "pong"})

        else:
            await self.send({"type": "error", "code": "unknown_message",
                              "message": f"unrecognised message type {msg_type!r}"})

    def _start_active(self) -> None:
        if self.perception is None:
            try:
                self.perception = HolisticPerception(stride=CONFIG.perception_stride)
            except PerceptionUnavailable as exc:
                logger.warning("perception_unavailable", extra={"error": str(exc)})
                self.perception = None   # frontend gets framing errors instead of a crash

    async def _handle_frame(self, message: dict) -> None:
        if self.session.state != SessionState.ACTIVE or not self.session.consent_processing:
            return
        if self.perception is None:
            await self.send({"type": "framing", "ok": False, "reason": "perception_unavailable"})
            return

        try:
            jpeg = decode_base64_frame(message["data"])
            self.perception.submit_frame(jpeg, int(message.get("ts_ms", time.time() * 1000)))
        except PerceptionUnavailable as exc:
            await self.send({"type": "error", "code": "perception_error", "message": str(exc)})
            return

        landmarks = self.perception.latest()
        if landmarks is None:
            return

        if landmarks.left is not None or landmarks.right is not None:
            self.session.touch()   # a visitor signing is activity; an empty room is not
        await self.send({
            "type": "landmarks",
            "pose": landmarks.pose.tolist() if landmarks.pose is not None else None,
            "left": landmarks.left.tolist() if landmarks.left is not None else None,
            "right": landmarks.right.tolist() if landmarks.right is not None else None,
        })
        await self.send({
            "type": "framing",
            "ok": landmarks.pose is not None,
            "reason": None if landmarks.pose is not None else "no_body",
        })

        utterance = self.segmenter.feed(landmarks)
        if utterance is None:
            return
        recognizer = self.pack.recognizer if self.pack is not None else None
        if recognizer is None:
            await self.send({"type": "error", "code": "model_unavailable",
                              "message": "no recognition model is loaded"})
            return

        # classify() is a synchronous PyTorch forward pass — called inline, it
        # blocks this process's single asyncio event loop for its whole
        # duration, during which no frame, ping, or message for any session
        # gets handled. On hardware without a TensorRT-converted model (this
        # deployment), that's enough to read as a periodic stutter right when
        # a word boundary is detected. classify() is pure/read-only (model in
        # .eval() + torch.no_grad(), no shared state) so running it in a
        # worker thread is safe.
        hypothesis: Hypothesis = await asyncio.to_thread(recognizer.classify, utterance)
        await self._handle_hypothesis(hypothesis, utterance)

    async def _handle_hypothesis(self, hypothesis: Hypothesis, utterance) -> None:
        from core.features import normalise_frame, resample

        raw_frames = resample(
            [normalise_frame(lm.pose, lm.left, lm.right) for lm in utterance.landmarks]
        ).tolist()

        if hypothesis.gloss is None:
            await self.send({"type": "hypothesis", "status": "rejected",
                              "gloss": None, "confidence": hypothesis.confidence,
                              "alternatives": hypothesis.alternatives})
            return

        if hypothesis.confidence >= CONFIG.confident_threshold:
            self.assembler.append(hypothesis.gloss)
            self.capture.add(CapturedItem(
                utterance_id=str(uuid.uuid4()), landmarks=raw_frames,
                predicted_gloss=hypothesis.gloss, confidence=hypothesis.confidence,
            ))
            await self.send({"type": "hypothesis", "status": "committed",
                              "gloss": hypothesis.gloss, "confidence": hypothesis.confidence,
                              "alternatives": hypothesis.alternatives})
            await self.send(self._rendered_message())
        else:
            # FR-K8: below the confident threshold but above rejection — offer
            # candidates as touch targets, never silently substitute.
            self.pending_candidates = (hypothesis.alternatives, raw_frames)
            await self.send({"type": "hypothesis", "status": "candidates",
                              "gloss": None, "confidence": hypothesis.confidence,
                              "alternatives": hypothesis.alternatives})

    async def _handle_select_candidate(self, gloss_id: str | None) -> None:
        if not gloss_id or self.pending_candidates is None:
            return
        alternatives, raw_frames = self.pending_candidates
        top_choice = alternatives[0][0] if alternatives else None
        confidence = next((c for g, c in alternatives if g == gloss_id), 0.0)

        self.assembler.append(gloss_id)
        self.capture.add(CapturedItem(
            utterance_id=str(uuid.uuid4()), landmarks=raw_frames,
            predicted_gloss=top_choice, confidence=confidence,
            # A weak label (FR-L2), never verified ground truth until Deaf review.
            correction=gloss_id if gloss_id != top_choice else None,
        ))
        self.pending_candidates = None
        await self.send(self._rendered_message())

    async def _handle_control(self, action: str | None) -> None:
        if action == "backspace":
            self.assembler.remove_last()
            await self.send(self._rendered_message())
        elif action == "clear":
            self.assembler.clear()
            await self.send(self._rendered_message())
        elif action == "extend":
            self.session.touch()
        elif action == "end_session":
            await self._end_session()
        else:
            await self.send({"type": "error", "code": "unknown_control", "message": str(action)})

    async def _handle_speak(self, lang: str) -> None:
        rendered = self.assembler.render()
        text = rendered.ar if lang == "ar" else rendered.en
        future = _tts.speak_async(text, lang)

        async def _await_and_notify():
            ok = await asyncio.wrap_future(future)
            try:
                await self.send({"type": "tts_done", "lang": lang, "ok": bool(ok)})
            except Exception:
                pass   # connection may already be closed; TTS completing is not fatal

        asyncio.create_task(_await_and_notify())

    async def _handle_ask(self) -> None:
        rendered = self.assembler.render()
        question = rendered.ar or " ".join(rendered.gloss_seq)
        # dialogue.answer() uses the synchronous anthropic.Anthropic() client —
        # called inline, a cloud round-trip (seconds, not milliseconds) blocks
        # this process's one asyncio event loop the whole time, freezing every
        # connected session's frames/pings/messages along with it. Worse than
        # classify()'s blocking cost above, same fix.
        answer = await asyncio.to_thread(self.dialogue.answer, question)
        await self.send({
            "type": "answer", "ar": answer.ar, "en": answer.en, "source": answer.source,
            "refused": answer.refused, "unrenderable_words": answer.unrenderable_words,
        })
        if not answer.refused:
            words = _clip_library.sequence_for_text(answer.ar, answer.en)
            await self.send({
                "type": "playback",
                "words": [
                    {"gloss_id": w.gloss_id, "ar": w.ar, "en": w.en, "has_clip": w.has_clip}
                    for w in words
                ],
            })

    def _rendered_message(self) -> dict:
        rendered = self.assembler.render()
        return {"type": "rendered", "gloss_seq": rendered.gloss_seq, "ar": rendered.ar,
                "en": rendered.en, "uncertain": rendered.uncertain}

    async def _end_session(self) -> None:
        self.capture.ended_at = datetime.now(timezone.utc).isoformat()
        target = write_pending_upload(self.capture, CONFIG.pending_uploads_dir)
        self.session.transition(SessionState.ENDED, uploaded_to=str(target) if target else None)
        await self.send({"type": "session_reset"})
        self._reset_for_next_visitor()
        if target:
            _schedule_flush()

    def _reset_for_next_visitor(self) -> None:
        """NFR-S3: session data is erased on return to idle. A person who walks
        away has consented to nothing — the next visitor must never see the
        previous one's conversation.
        """
        if self.perception is not None:
            self.perception.close()
            self.perception = None
        self.pack = registry.get(DEFAULT_LANGUAGE)
        self.assembler = Assembler(self._vocab_for(self.pack))
        self.segmenter.reset()
        self.pending_candidates = None
        old_kiosk_id = self.session.kiosk_id
        self.session = Session(kiosk_id=old_kiosk_id)
        self.capture = SessionCapture(
            session_id=self.session.correlation_id, kiosk_id=old_kiosk_id, language="liu",
            consent_retention=False, started_at=datetime.now(timezone.utc).isoformat(),
        )

    async def check_idle(self) -> bool:
        """Returns True if the connection should keep running."""
        if self.session.state not in (SessionState.ACTIVE, SessionState.CONSENT, SessionState.LANGUAGE):
            return True
        idle = self.session.idle_for()
        remaining = CONFIG.idle_timeout_s - idle
        if remaining <= 0:
            await self._end_session()
        elif remaining <= 10:
            await self.send({"type": "idle_warning", "seconds_left": int(remaining)})
        return True


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    await ws.accept()
    conn = KioskConnection(ws)
    logger.info("session_started", extra={"correlation_id": conn.session.correlation_id})
    await conn.send_state()
    last_idle_check = time.monotonic()
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=2.0)
            except asyncio.TimeoutError:
                await conn.check_idle()
                last_idle_check = time.monotonic()
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await conn.send({"type": "error", "code": "bad_json", "message": "not valid JSON"})
                continue
            await conn.handle(message)
            # While the camera streams, the receive above never times out, so the
            # idle check has to run on the clock, not only on silence.
            if time.monotonic() - last_idle_check >= 2.0:
                await conn.check_idle()
                last_idle_check = time.monotonic()
    except WebSocketDisconnect:
        logger.info("session_disconnected", extra={"correlation_id": conn.session.correlation_id})
    finally:
        if conn.perception is not None:
            conn.perception.close()


class _HttpOnlyStaticFiles(StaticFiles):
    """`StaticFiles` hard-asserts `scope["type"] == "http"` — correct for it to
    refuse anything else, wrong for that refusal to be an unhandled
    `AssertionError` that uvicorn logs as a crash on every occurrence. A
    websocket-scoped request only reaches this mount when its path didn't match
    `/ws/session` (a stray probe, a typo'd client, dev tooling) — reject it
    cleanly instead of raising.
    """

    async def __call__(self, scope, receive, send) -> None:  # noqa: D102
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1000})
            return
        await super().__call__(scope, receive, send)


# Static frontend last, so it never shadows the API/WebSocket routes above.
app.mount("/", _HttpOnlyStaticFiles(directory=str(REPO_ROOT), html=True), name="frontend")

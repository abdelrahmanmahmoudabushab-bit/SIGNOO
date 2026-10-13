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
    {"type": "framing", "ok": true|false,
     "reason": "too_dark"|"too_bright"|"no_body"|"perception_unavailable"|null}
        # FR-K5. Sent when it changes, not with every frame.
    {"type": "signing", "state": "watching"|"signing"|"recognising"}
        # NFR-X1: hands up and a word being collected / the word is closed and the model
        # is running. Sent when it changes.
    {"type": "hypothesis", "status": "committed"|"candidates"|"rejected",
     "gloss": "marhaba"|null, "confidence": 0.9, "alternatives": [["marhaba", 0.9], ...],
     "reason": "not_recognised"|"hands_lost"}
        # `reason` only on a rejection: the model was not sure enough, or tracking lost
        # the hands for most of the sign (then nothing was classified at all).
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
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.contracts import Hypothesis
from kiosk.assembly import Assembler, Vocabulary
from kiosk.config import CONFIG, REPO_ROOT, SEGMENTATION_DEFAULTS
from kiosk.data_capture import (
    CapturedItem, SessionCapture, flush_pending_uploads, write_pending_upload,
)
from kiosk.dialogue import Dialogue
from kiosk.fleet import FleetReporter
from kiosk.output import TTSEngine
from kiosk.perception import HolisticPerception, PerceptionUnavailable, decode_base64_frame, jpeg_size
from kiosk.languages import LanguageRegistry
from kiosk.kb_sync import KnowledgeSync
from kiosk.model_sync import ModelSync, read_installed
from kiosk.segmentation import Segmenter
from kiosk.session import LiveStats, Session, SessionState, configure_logging, telemetry_snapshot

configure_logging()
logger = logging.getLogger("signoo.kiosk.main")


async def _heartbeat_loop() -> None:
    reporter = FleetReporter(CONFIG.server_url, CONFIG.kiosk_id, CONFIG.kiosk_token,
                             model_versions=registry.model_versions)
    sync = ModelSync(CONFIG.server_url, CONFIG.kiosk_id, registry, CONFIG.synced_models_dir,
                     CONFIG.kiosk_token)
    kb_sync = KnowledgeSync(CONFIG.server_url, CONFIG.kiosk_id, registry, CONFIG.synced_kb_dir,
                            CONFIG.kiosk_token)
    while True:
        # Blocking HTTP (and a new model's load), kept off the event loop. Models and
        # knowledge bases are only asked for once the server has heard from us this tick.
        if await asyncio.to_thread(reporter.tick):
            await asyncio.to_thread(sync.tick)
            await asyncio.to_thread(kb_sync.tick)
        await asyncio.sleep(CONFIG.heartbeat_interval_s)


LIVE = LiveStats()                              # B9: what the pipeline is doing, for /health
_warm_up_seconds: dict[str, float | None] = {}  # per language, from the start-up test pass


def _warm_up_models() -> None:
    for code, pack in registry.packs.items():
        recognizer = pack.recognizer
        if recognizer is None:
            continue
        seconds = recognizer.warm_up()
        _warm_up_seconds[code] = None if seconds is None else round(seconds, 3)
        log = logger.info if seconds is not None else logger.error
        log("model_warmed_up" if seconds is not None else "model_warm_up_failed",
            extra={"language": code, "seconds": _warm_up_seconds[code]})


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _schedule_flush()   # retry anything a previous run could not deliver
    # Off the event loop and before the first visitor: each language's model takes its slow
    # first forward pass now, not on the first sign somebody makes.
    await asyncio.to_thread(_warm_up_models)
    heartbeat = asyncio.create_task(_heartbeat_loop()) if CONFIG.server_url else None
    try:
        yield
    finally:
        if heartbeat is not None:
            heartbeat.cancel()


app = FastAPI(title="SIGNOO kiosk", lifespan=_lifespan)


@app.middleware("http")
async def _never_reuse_a_stale_frontend(request, call_next):
    """The page, its scripts and the vocabulary files are revalidated on every load.

    Found on the Jetson: Electron kept a week-old copy of assets/app.js in its cache, so
    after an update the screen still showed only the first language as open and the new
    language cards stayed "coming soon" while the backend said they were ready. They are a
    few small files on localhost; revalidating costs nothing and a stale kiosk costs a lot.
    """
    response = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith(("/ws", "/health", "/languages")):
        response.headers["Cache-Control"] = "no-cache"
    return response

# ── process-wide singletons (models load once at startup, per SYSTEM_DESIGN.md §4.2) ──
_tts = TTSEngine(enabled=CONFIG.tts_enabled)

# One isolated engine per sign language (core/languages.json): model, vocabulary,
# confidence bars, segmentation tuning, knowledge base and clip library. A corrupt or
# missing checkpoint leaves that one language unavailable (SYSTEM_DESIGN.md §6) rather
# than stopping the process: the frontend still needs to serve a clear out-of-service state.
DEFAULT_LANGUAGE = "liu"
registry = LanguageRegistry(
    CONFIG.languages_path, REPO_ROOT,
    CONFIG.reject_threshold, CONFIG.confident_threshold, CONFIG.top_k_alternatives,
    # Keeps SIGNOO_MODEL_PATH / _VOCAB_PATH / _KB_PATH / _CLIP_LIBRARY_PATH working;
    # they configure the default language only.
    overrides={DEFAULT_LANGUAGE: {
        "model": CONFIG.model_path, "vocab": CONFIG.vocab_path,
        "knowledge_base": CONFIG.knowledge_base_path, "clip_library": CONFIG.clip_library_path,
    }},
    # A model the server rolled out earlier outlives a restart (kiosk/model_sync.py).
    installed=read_installed(CONFIG.synced_models_dir),
    allow_synthetic=CONFIG.allow_synthetic,
    kb_dir=CONFIG.synced_kb_dir,
)

_background_tasks: set[asyncio.Task] = set()


def _schedule_flush() -> None:
    """Delivers queued sessions to server/ off the event loop, never blocking a
    visitor; a no-op unless SIGNOO_SERVER_URL is set."""
    if not CONFIG.server_url:
        return
    task = asyncio.create_task(asyncio.to_thread(
        flush_pending_uploads, CONFIG.pending_uploads_dir, CONFIG.server_url, 5.0, CONFIG.kiosk_token
    ))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.get("/health")
def health() -> dict:
    default = registry.get(DEFAULT_LANGUAGE)
    return {
        "status": "ok" if default is not None and default.ready else "degraded",
        "model_loaded": default is not None and default.ready,
        "labels": default.summary()["labels"] if default else 0,
        "languages": registry.summary(),
        "telemetry": telemetry_snapshot(LIVE),
        # Seconds each language's model took for its start-up test pass; null means the
        # pass failed and that model may not run at all.
        "warm_up_s": dict(_warm_up_seconds),
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
        self._use_engine(registry.get(DEFAULT_LANGUAGE))
        self.capture = SessionCapture(
            session_id=self.session.correlation_id,
            kiosk_id=CONFIG.kiosk_id,
            language=DEFAULT_LANGUAGE,
            consent_retention=False,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.perception: HolisticPerception | None = None
        self._perception_task: asyncio.Task | None = None   # tracker start-up begun at language pick
        self._perception_generation = 0   # bumped when a visit ends: a tracker still starting belongs to nobody
        self._perception_task_generation = -1   # the visit `_perception_task` was started for
        self._closed = False              # the connection is gone; nothing may be attached to it any more
        self._perception_lock = threading.Lock()
        self._frame_seconds = 0.0
        self._frame_count = 0
        self._last_framing: dict | None = None      # last framing / sign-state message sent,
        self._last_sign_state: dict | None = None   # so an unchanged one is not sent again
        # The stage after tracking (see _ensure_result_consumer). Built on first use, inside
        # the running event loop it belongs to.
        self._consumer: asyncio.Task | None = None
        self._consumer_loop = None
        self._result_event: asyncio.Event | None = None
        self._results_lock: asyncio.Lock | None = None
        self._listening_to = None                   # the tracker currently told how to wake the consumer
        self.pending_candidates: list[tuple[str, float]] | None = None

    def _use_engine(self, pack) -> None:
        """Puts this connection on one language's whole engine. Every language-specific
        part comes from `pack` and nothing is borrowed from another language: a missing
        part is empty (no words, questions refused, nothing to play back).
        """
        self.pack = pack
        self.pending_candidates = None    # candidates offered by the previous language's model
        vocab = pack.vocab if pack is not None and pack.vocab is not None else Vocabulary(None)
        self.clips = pack.clips if pack is not None else None
        self.assembler = Assembler(vocab)
        self.segmenter = (pack.make_segmenter(SEGMENTATION_DEFAULTS) if pack is not None
                          else Segmenter(**SEGMENTATION_DEFAULTS))
        self.dialogue = Dialogue(
            pack.kb if pack is not None else None,
            CONFIG.kb_similarity_floor, CONFIG.claude_model, CONFIG.claude_timeout_s,
            CONFIG.dialogue_cloud_enabled, vocab,
            self.clips._entries if self.clips is not None else {},
        )

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
            if pack is None or not pack.selectable:
                # Never fall back to another language's model: a visitor who picked
                # American must not be recognised by the Jordanian model, or vice versa.
                # `selectable` also refuses a model trained only on generated data.
                await self.send({"type": "error", "code": "language_unavailable",
                                 "message": f"{code!r} has no trained model yet"})
                return
            self._use_engine(pack)
            self.session.language = code
            if code != self.capture.language:
                # The upload carries one language for the whole session. Anything signed
                # under the previous one would be stored, and later trained on, as this one.
                self.capture.items.clear()
            self.capture.language = code
            self.session.transition(SessionState.LANGUAGE, language=self.session.language)
            await self.send_state()
            in_flight = (self._perception_task is not None and not self._perception_task.done()
                         and self._perception_task_generation == self._perception_generation)
            if self.perception is None and not in_flight:
                self._perception_task_generation = self._perception_generation
                # The visit is identified now, not when the task first runs: a visit that
                # ends in between must not leave a tracker behind for the next person.
                self._perception_task = asyncio.create_task(
                    self._prepare_perception(self._perception_generation))
                _background_tasks.add(self._perception_task)
                self._perception_task.add_done_callback(_background_tasks.discard)

        elif msg_type == "consent":
            if message.get("processing") and (self.pack is None or not self.pack.selectable):
                # A connection starts on the default engine, so a client that never sent a
                # `language` message would otherwise be recognised by it. The rule that a
                # language not trained on real people is not offered has to hold on this
                # path too, not only when a card is pressed.
                await self.send({"type": "error", "code": "language_unavailable",
                                 "message": "no language with a trained model has been selected"})
                return
            self.session.consent_processing = bool(message.get("processing"))
            self.session.consent_retention = bool(message.get("retention"))
            self.capture.consent_retention = self.session.consent_retention
            if self.session.consent_processing:
                await self._start_active()
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

    async def _prepare_perception(self, generation: int | None = None) -> None:
        """Starts the tracker. Takes seconds on the Jetson, so it runs in a worker thread, and
        is begun when the visitor picks a language, while they read the consent screen, so
        that signing can start the moment they agree. It sees no frame until consent: frames
        are refused in `_handle_frame` unless the session is ACTIVE with consent given."""
        if generation is None:
            generation = self._perception_generation
        if self.perception is not None or generation != self._perception_generation:
            return   # already running, or the visit this was started for has ended
        started = time.perf_counter()
        try:
            attached = await asyncio.to_thread(self._build_perception, generation)
        except PerceptionUnavailable as exc:
            logger.warning("perception_unavailable", extra={"error": str(exc)})
            return   # self.perception stays None: the frontend gets framing errors, not a crash
        if attached:
            logger.info("perception_ready", extra={"seconds": round(time.perf_counter() - started, 2)})

    def _build_perception(self, generation: int) -> bool:
        """Worker-thread half of `_prepare_perception`: builds the tracker and attaches it,
        unless the page went away or the visit ended while it was starting. Then nobody
        would ever close it, so it is closed here: left to the garbage collector, a MediaPipe
        landmarker holds its memory and can deadlock when it is finally freed. The decision is
        made in this thread, under the lock, so it holds even if the waiting task is cancelled."""
        perception = HolisticPerception(stride=CONFIG.perception_stride)
        with self._perception_lock:
            wanted = (not self._closed and generation == self._perception_generation
                      and self.perception is None)
            if wanted:
                self.perception = perception
        if not wanted:
            perception.close()
        return wanted

    async def _start_active(self) -> None:
        task = self._perception_task
        if (task is not None and not task.done()
                and self._perception_task_generation == self._perception_generation):
            await task            # already on its way for this visit: wait, do not start a second one
        if self.perception is None:
            await self._prepare_perception()   # never started, or it was for a visit that ended

    def _note_frame_time(self, seconds: float) -> None:
        """Rolling cost of taking one frame in (decode + hand-off to the tracker), logged once
        every 100 frames, so a slow kiosk shows up in the log instead of only on the screen."""
        LIVE.note_frame(seconds)
        self._frame_seconds += seconds
        self._frame_count += 1
        if self._frame_count >= 100:
            brightness = self.perception.brightness if self.perception is not None else None
            logger.info("frame_timing", extra={
                "avg_ms": round(self._frame_seconds / self._frame_count * 1000, 1),
                "frames": self._frame_count,
                "tracker_fps": LIVE.snapshot()["tracker_fps"],
                # What SIGNOO_LIGHT_TOO_DARK / _TOO_BRIGHT are set from on a real site.
                "brightness": None if brightness is None else round(brightness)})
            self._frame_seconds = 0.0
            self._frame_count = 0

    async def _handle_frame(self, message: dict) -> None:
        # Checked before anything is decoded: the size of the text, then the size the
        # JPEG header claims. A small file can claim an enormous image, and decoding
        # it is what would exhaust the device's memory.
        data = message.get("data")
        if not isinstance(data, str) or not data or len(data) > CONFIG.max_frame_b64_chars:
            await self.send({"type": "error", "code": "bad_frame",
                             "message": "frame data is missing, not text, or too large"})
            return
        try:
            jpeg = decode_base64_frame(data)
            ts_ms = int(message.get("ts_ms", time.time() * 1000))
        except (ValueError, TypeError, OverflowError):
            await self.send({"type": "error", "code": "bad_frame",
                             "message": "frame is not base64 or its timestamp is not a number"})
            return
        size = jpeg_size(jpeg)
        if size is None or not 0 < size[0] * size[1] <= CONFIG.max_frame_pixels:
            await self.send({"type": "error", "code": "bad_frame",
                             "message": "frame is not a JPEG or its image is too large"})
            return

        if self.session.state != SessionState.ACTIVE or not self.session.consent_processing:
            return
        if self.perception is None:
            await self.send({"type": "framing", "ok": False, "reason": "perception_unavailable"})
            return

        self._ensure_result_consumer()   # before the frame goes in, so its result is not missed
        try:
            frame_started = time.perf_counter()
            self.perception.submit_frame(jpeg, ts_ms)
            self._note_frame_time(time.perf_counter() - frame_started)
        except PerceptionUnavailable as exc:
            await self.send({"type": "error", "code": "perception_error", "message": str(exc)})
            return

        # Anything the tracker has already finished is handled now. What it finishes later
        # is handled by the consumer the moment it is ready, not when the next frame arrives.
        if not self._results_lock.locked():
            await self._drain_results()

    def _ensure_result_consumer(self) -> None:
        """Starts the task that handles each tracking result as soon as it exists, and tells
        the tracker how to wake it.

        Tracking is asynchronous: a frame goes in, its result comes out about 100 ms later
        on MediaPipe's thread. Results used to be collected only when the next frame
        arrived, which delayed every hand position by a frame and, with the browser waiting
        for a result before sending again, stalled both sides until a timeout and threw away
        every second result (measured 2026-10-13). Capture, tracking and the rest are
        separate stages joined by a newest-wins slot (SYSTEM_DESIGN.md B1); this is the
        stage after tracking, running on its own."""
        loop = asyncio.get_running_loop()
        if self._consumer is None or self._consumer.done() or self._consumer_loop is not loop:
            self._consumer_loop = loop
            self._result_event = asyncio.Event()
            self._results_lock = asyncio.Lock()
            self._consumer = asyncio.create_task(self._consume_results(self._result_event))
            _background_tasks.add(self._consumer)
            self._consumer.add_done_callback(_background_tasks.discard)
            self._listening_to = None
        if self._listening_to is not self.perception:
            event = self._result_event

            def wake() -> None:      # called on MediaPipe's thread: only signal, never work
                try:
                    loop.call_soon_threadsafe(event.set)
                except RuntimeError:
                    pass             # the loop has closed: the connection is going away

            self.perception.set_listener(wake)
            self._listening_to = self.perception

    async def _consume_results(self, event: asyncio.Event) -> None:
        while not self._closed:
            await event.wait()
            event.clear()
            try:
                await self._drain_results()
            except (WebSocketDisconnect, RuntimeError) as exc:
                # Almost always the socket having gone, which the handler's `finally` cleans
                # up after. Results are still collected as frames arrive, and the next frame
                # starts a new consumer, so stopping here loses nothing.
                logger.info("result_consumer_stopped", extra={"reason": str(exc) or type(exc).__name__})
                return
            except Exception:
                logger.exception("result_handling_failed",
                                 extra={"correlation_id": self.session.correlation_id})

    async def _drain_results(self) -> None:
        """Handles the newest tracking result, if there is one. One at a time, in order:
        the lock is held across the model pass, so the segmenter never sees a later frame
        while an earlier word is still being classified."""
        async with self._results_lock:
            perception = self.perception
            landmarks = perception.latest() if perception is not None else None
            if landmarks is None:
                return
            if self.session.state != SessionState.ACTIVE or not self.session.consent_processing:
                return   # a result that finished after the visit ended belongs to nobody
            await self._process_landmarks(landmarks)

    async def _process_landmarks(self, landmarks) -> None:
        LIVE.note_tracker_result()

        if landmarks.left is not None or landmarks.right is not None:
            self.session.touch()   # a visitor signing is activity; an empty room is not
        await self.send({
            "type": "landmarks",
            "pose": landmarks.pose.tolist() if landmarks.pose is not None else None,
            "left": landmarks.left.tolist() if landmarks.left is not None else None,
            "right": landmarks.right.tolist() if landmarks.right is not None else None,
        })
        await self._send_if_changed("_last_framing", self._framing_message(landmarks))

        utterance = self.segmenter.feed(landmarks)
        if utterance is None:
            await self._send_sign_state("signing" if self.segmenter.buffering else "watching")
            return
        recognizer = self.pack.recognizer if self.pack is not None else None
        if recognizer is None:
            await self.send({"type": "error", "code": "model_unavailable",
                              "message": "no recognition model is loaded"})
            return

        missing = utterance.frames_without_hands
        frames = len(utterance.landmarks)
        if missing / (frames + missing) > CONFIG.max_hands_lost_share:
            # FR-K5 and Requirements §6.3: most of this sign was not seen. Saying so is
            # honest and tells the visitor what to change; classifying it would be a guess
            # dressed up as recognition.
            logger.info("recognition", extra={
                "language": self.pack.code, "status": "hands_lost", "gloss": None,
                "frames": frames, "frames_without_hands": missing,
                "span_ms": utterance.t_end - utterance.t_start})
            LIVE.note_outcome("hands_lost")
            await self.send({"type": "hypothesis", "status": "rejected", "reason": "hands_lost",
                             "gloss": None, "confidence": 0.0, "alternatives": []})
            await self._send_sign_state("watching")
            return

        # NFR-X1: "I am working on what you just signed" has to be visible, or people sign
        # again over the top of it. Sent at the word boundary, before the model runs.
        await self._send_sign_state("recognising")

        # classify() is a synchronous PyTorch forward pass — called inline, it
        # blocks this process's single asyncio event loop for its whole
        # duration, during which no frame, ping, or message for any session
        # gets handled. On hardware without a TensorRT-converted model (this
        # deployment), that's enough to read as a periodic stutter right when
        # a word boundary is detected. classify() is pure/read-only (model in
        # .eval() + torch.no_grad(), no shared state) so running it in a
        # worker thread is safe.
        started = time.perf_counter()
        hypothesis: Hypothesis = await asyncio.to_thread(recognizer.classify, utterance)
        await self._handle_hypothesis(hypothesis, utterance, time.perf_counter() - started)
        await self._send_sign_state("watching")

    def _framing_message(self, landmarks) -> dict:
        """FR-K5: what is wrong with the picture, if anything. Lighting is checked first
        because it is usually the cause: in a dark or washed-out frame the body is not
        found either, and "step back" would be the wrong advice."""
        reason = None
        brightness = self.perception.brightness if self.perception is not None else None
        if brightness is not None and brightness < CONFIG.light_too_dark:
            reason = "too_dark"
        elif brightness is not None and brightness > CONFIG.light_too_bright:
            reason = "too_bright"
        elif landmarks.pose is None:
            reason = "no_body"
        return {"type": "framing", "ok": reason is None, "reason": reason}

    async def _send_if_changed(self, slot: str, message: dict) -> None:
        """Sends a status message only when it differs from the last one sent in `slot`.
        These arrive with every tracked frame; repeating an unchanged one ten times a
        second is traffic and JSON work for nothing."""
        if getattr(self, slot) != message:
            setattr(self, slot, message)
            await self.send(message)

    async def _send_sign_state(self, state: str) -> None:
        """watching -> signing (hands up, a word is being collected) -> recognising (the
        word is closed, the model is running) -> watching."""
        await self._send_if_changed("_last_sign_state", {"type": "signing", "state": state})

    async def _handle_hypothesis(self, hypothesis: Hypothesis, utterance, classify_s: float = 0.0) -> None:
        from core.features import normalise_frame, resample

        raw_frames = resample(
            [normalise_frame(lm.pose, lm.left, lm.right) for lm in utterance.landmarks]
        ).tolist()

        # One line per recognised sign, so a live session can be judged afterwards: how sure
        # the model was, and against which bar. Never landmarks or identity. The words
        # themselves are what the visitor said, so they are written only for a visitor who
        # agreed to retention (NFR-S3: without it, nothing of the session outlives it); the
        # log keeps the confidences either way, which is what tuning needs.
        recognizer_now = self.pack.recognizer if self.pack is not None else None
        bar = (recognizer_now.commit_bar(hypothesis.gloss) if recognizer_now and hypothesis.gloss
               else CONFIG.confident_threshold)
        status = ("rejected" if hypothesis.gloss is None
                  else "committed" if hypothesis.confidence >= bar else "candidates")
        words_kept = self.session.consent_retention
        logger.info("recognition", extra={
            "language": self.pack.code if self.pack is not None else None,
            "status": status,
            "gloss": hypothesis.gloss if words_kept else None,
            "confidence": round(float(hypothesis.confidence), 3), "bar": round(float(bar), 3),
            "top3": [(g if words_kept else None, round(float(c), 3)) for g, c in hypothesis.alternatives[:3]],
            "words_logged": words_kept,
            "frames": len(utterance.landmarks),
            "frames_without_hands": utterance.frames_without_hands,
            "span_ms": utterance.t_end - utterance.t_start,
            "classify_ms": round(classify_s * 1000),
        })
        LIVE.note_outcome(status, classify_s)

        if hypothesis.gloss is None:
            await self.send({"type": "hypothesis", "status": "rejected", "reason": "not_recognised",
                              "gloss": None, "confidence": hypothesis.confidence,
                              "alternatives": hypothesis.alternatives})
            return

        recognizer = self.pack.recognizer if self.pack is not None else None
        confident = recognizer.commit_bar(hypothesis.gloss) if recognizer else CONFIG.confident_threshold
        if hypothesis.confidence >= confident:
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
        if not isinstance(gloss_id, str) or self.pending_candidates is None:
            return
        alternatives, raw_frames = self.pending_candidates
        if gloss_id not in [gloss for gloss, _ in alternatives]:
            return   # only a candidate that was just offered can be chosen, never an arbitrary word
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
            try:
                ok = await asyncio.wrap_future(future)
            except Exception:
                # A speech engine that fails must still end the "speaking" state on screen.
                logger.exception("tts_failed", extra={"lang": lang})
                ok = False
            try:
                await self.send({"type": "tts_done", "lang": lang, "ok": bool(ok)})
            except Exception:
                logger.debug("tts_done_not_sent: the connection was already closed")

        # Hold a reference: the loop keeps only a weak one, so an untracked task can be
        # collected before it finishes.
        task = asyncio.create_task(_await_and_notify())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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
            # This language's own clips; with no clip library there is nothing to play.
            words = self.clips.sequence_for_text(answer.ar, answer.en) if self.clips is not None else []
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
        with self._perception_lock:
            self._perception_generation += 1
            perception, self.perception = self.perception, None
        if perception is not None:
            perception.close()
        self._listening_to = None
        self._use_engine(registry.get(DEFAULT_LANGUAGE))
        self.pending_candidates = None
        self._last_framing = self._last_sign_state = None   # the next visitor's screen starts blank
        old_kiosk_id = self.session.kiosk_id
        self.session = Session(kiosk_id=old_kiosk_id)
        self.capture = SessionCapture(
            session_id=self.session.correlation_id, kiosk_id=old_kiosk_id, language=DEFAULT_LANGUAGE,
            consent_retention=False, started_at=datetime.now(timezone.utc).isoformat(),
        )

    def shut_down(self) -> None:
        """The connection is over. Frees the tracker, and marks the connection closed so that
        one still starting up in the background closes itself when it arrives."""
        with self._perception_lock:
            self._closed = True
            perception, self.perception = self.perception, None
        if perception is not None:
            perception.close()
        self._listening_to = None
        if self._consumer is not None and not self._consumer.done():
            try:
                self._consumer.cancel()
            except RuntimeError:
                pass   # its event loop has already closed, and took the task with it

    async def release(self) -> None:
        """Another connection has replaced this one: free its perception and close it."""
        self.shut_down()
        try:
            await self.ws.close(code=SUPERSEDED_CLOSE_CODE)
        except Exception:
            logger.debug("superseded_close_failed: the old socket was already gone")

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


SUPERSEDED_CLOSE_CODE = 4001
_active_connection: KioskConnection | None = None


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    global _active_connection
    await ws.accept()
    conn = KioskConnection(ws)
    # One kiosk has one visitor. A second connection (a reloaded page, a stray tab) takes
    # over and the older one is closed, so a MediaPipe instance cannot be left running
    # for a page nobody is looking at: the Jetson's memory has room for one.
    previous, _active_connection = _active_connection, conn
    if previous is not None:
        await previous.release()
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
            if not isinstance(message, dict):
                await conn.send({"type": "error", "code": "bad_message", "message": "a message must be a JSON object"})
                continue
            try:
                await conn.handle(message)
            except WebSocketDisconnect:
                raise
            except Exception:
                # A malformed field must cost the sender an error, not the visitor their session.
                logger.exception("message_failed", extra={"correlation_id": conn.session.correlation_id})
                await conn.send({"type": "error", "code": "bad_message",
                                 "message": "that message could not be handled"})
            # While the camera streams, the receive above never times out, so the
            # idle check has to run on the clock, not only on silence.
            if time.monotonic() - last_idle_check >= 2.0:
                await conn.check_idle()
                last_idle_check = time.monotonic()
    except WebSocketDisconnect:
        logger.info("session_disconnected", extra={"correlation_id": conn.session.correlation_id})
    except RuntimeError:
        # Raised by the receive after `release()` closed this socket from the other side.
        logger.info("session_superseded", extra={"correlation_id": conn.session.correlation_id})
    finally:
        if _active_connection is conn:
            _active_connection = None
        conn.shut_down()


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


# The frontend, last so it never shadows the API/WebSocket routes above. Only what the
# browser asks for is served: index.html, assets/, and the vocabulary files. The repo
# root also holds queued visitor sessions, the server's database, models and source,
# none of which may be reachable from the kiosk's port.
app.mount("/assets", _HttpOnlyStaticFiles(directory=str(REPO_ROOT / "assets")), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(REPO_ROOT / "index.html")


@app.get("/{path:path}", include_in_schema=False)
def vocabulary_file(path: str) -> FileResponse:
    """The default vocabulary assets/app.js starts with, and each ready language's
    `vocab_url` from /languages. The path is matched against that list, never opened as given."""
    allowed = {"core/vocab.json"} | {
        pack.vocab_path.as_posix() for pack in registry.packs.values() if pack.ready and pack.vocab_path
    }
    target = REPO_ROOT / path
    if path not in allowed or not target.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(target)

# kiosk/ — SIGNOO kiosk runtime (B1-B10)

FastAPI + WebSocket backend, paired with the browser frontend at the repo root
(`index.html`, `assets/*.js`). This directory is Phase 2 of the delegated plan —
`docs/SYSTEM_DESIGN.md` §1-2 and §4 are its spec; `docs/API_CONTRACTS.md` is what
it must interoperate with `server/` and `console/` on.

## Setup (Windows)

The `python` on PATH in this environment is unrelated to this project (no numpy/
torch). Build a dedicated venv from the real Python 3.11 install and use only
that venv's `python`/`pip` for everything below:

```bash
/c/Users/Abdel/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -r kiosk/requirements.txt
```

Windows' console defaults to cp1252, which crashes on Arabic text or a `→` the
moment anything prints it directly. `kiosk/session.py`'s `configure_logging()`
reconfigures stdout to UTF-8, so kiosk code is safe; if you run an `ml/` script
directly on Windows, set `PYTHONIOENCODING=utf-8` first (that latent issue is in
`ml/`, out of this phase's remit — see "Found in `ml/`/`core/`" below).

## Build a checkpoint

The kiosk needs a checkpoint at `models/signoo.pt` to boot recognition. Until real
LIU data exists (`docs/ROADMAP.md` Phase 6), build the synthetic one per
`models/README.md`:

```bash
export PYTHONIOENCODING=utf-8      # see the ml/ note above
.venv/Scripts/python.exe ml/make_synthetic.py --out data/synthetic --signs 20 --signers 12
.venv/Scripts/python.exe ml/train.py data/synthetic/manifest.json --split data/synthetic/split.json \
    --out models/signoo.pt --epochs 40
```

If it's missing, `kiosk/main.py` still starts (the frontend can serve a clear
out-of-service state per `SYSTEM_DESIGN.md §6`) but `/health` reports
`"model_loaded": false` and every "frame" message returns a `model_unavailable`
error instead of a hypothesis.

## Run

One process serves both the API/WebSocket and the static frontend:

```bash
.venv/Scripts/python.exe -m uvicorn kiosk.main:app --port 8000
```

Open `http://localhost:8000/` in a browser with a camera. `http://localhost:8000/health`
reports model-load status and telemetry (temperature/throttle are honestly
`"unavailable"` — this is a dev machine, not a Jetson).

## Languages (one isolated engine per sign language)

`core/languages.json` lists every card on the language screen. Each entry is that
language's whole engine: `model`, `vocab`, optional `confident_threshold` /
`reject_threshold`, optional `segmentation` tuning, `knowledge_base` and
`clip_library`, plus `status` (`live` | `pilot` | `coming_soon`) and `real_data` (was
the model trained on real human signing). `kiosk/languages.py` loads each one into a
`LanguagePack` at startup. A language with `real_data: false` must have the status
`coming_soon` (`kiosk/tests/test_live_feedback.py` fails otherwise); that is `liu` today.

Pressing a card switches the visitor to that pack as a unit
(`KioskConnection._use_engine`). Nothing is shared and nothing falls back across
languages:

- A language is **ready** only if its checkpoint and vocabulary load, and **selectable**
  only if it is also trained on real signing (or `SIGNOO_ALLOW_SYNTHETIC=1`). Selecting one
  that is not selectable returns `language_unavailable` and changes nothing. In
  `GET /languages`, `ready` is the selectable answer and `model_loaded` is the plain one.
- Every card in `index.html` starts disabled and "coming soon"; the page enables only what
  `GET /languages` reports ready. If the backend cannot answer, no card is pressable.
- A language with no `knowledge_base` (or an empty one, like `asl` today:
  `kiosk/data/knowledge_base_asl.json` is `[]`) refuses every question. It never
  answers from another language's knowledge base.
- A language with no `clip_library` sends an empty `playback` list and reports every
  answer word as unrenderable. It never looks up another language's clips.
- If the default language (`liu`) itself is not ready, a new connection holds an empty
  engine: no vocabulary, questions refused, frames answered with `model_unavailable`.

`GET /languages` reports `ready`, `has_knowledge_base` and `has_clip_library` per
language; the frontend only lets ready cards be pressed.

To bring a language live: train a checkpoint (`ml/train.py ... --out models/<name>.pt`),
add its vocabulary and clip library under `core/` and its knowledge base under
`kiosk/data/`, set the paths in `languages.json`, and restart. The env overrides
`SIGNOO_MODEL_PATH`, `SIGNOO_VOCAB_PATH`, `SIGNOO_KB_PATH` and
`SIGNOO_CLIP_LIBRARY_PATH` apply to the default language (`liu`) only.
`kiosk/replay.py` builds its segmenter from the same pack, so it mirrors the kiosk
per language.
The American pilot is `models/asl50.bars.pt`: 50 words trained on PopSign (CC BY 4.0)
phone-camera clips, with a per-word confirmation bar for the 45 words that have one
(`ml/word_bars.py`). It commits a word at 0.80 confidence (or the word's own higher bar),
offers the top three between 0.25 and that, and refuses below 0.25. It is shown as "Pilot"
and is not a Jordanian model. `models/asl_pilot.pt` (8 signs) is the earlier pilot, kept
but no longer configured.

## Wire protocol (`/ws/session`)

JSON text frames both ways; the full reference lives as a docstring at the top of
`kiosk/main.py`. Design choice: camera capture and rendering stay in the browser
(`getUserMedia`); every processed frame is sent to this backend as **base64
JPEG**, and *all* of B2 Perception, B3 Segmentation, B4 Recognition, B5 Assembly,
B7 Dialogue, and B8 Playback run server-side. The browser only draws what it's
told (corner marks round the hands and face, recognised words, candidates, the answer,
playback placeholders) — this keeps the model, the KB, and the clip library in one place
and matches `SYSTEM_DESIGN.md`'s "no camera handles, no model objects... crossing
a boundary" rule at the process boundary too, not just between Python modules.

## Architecture notes (one per block)

- **B1 Capture** (`kiosk/buffers.py`) — `SingleSlotBuffer`: newest-wins, always
  drops rather than queues. Used both for MediaPipe's async LIVE_STREAM results
  (`kiosk/perception.py`) and is unit-tested directly (`tests/test_buffers.py`).
  **A result is handled the moment the tracker produces it** (2026-10-13): the tracker
  wakes a task on the connection (`_ensure_result_consumer` in `kiosk/main.py`) which
  sends the landmarks and feeds the segmenter. Before, a result waited in the slot until
  the next frame arrived and asked for it, which delayed every hand position by a frame
  and, with the browser pacing its frames to the answers, lost every second result.
- **B2 Perception** (`kiosk/perception.py`) — MediaPipe Holistic in **LIVE_STREAM**
  mode (not `ml/extract_landmarks.py`'s VIDEO mode — that mode is for a finite
  recorded file; LIVE_STREAM is the Tasks API's mode for a live, one-frame-at-a-time
  source, and is asynchronous via `detect_async`/`result_callback`). One
  `HolisticPerception` instance per WebSocket connection, closed on session end —
  tracking state must not bleed between visitors, same reasoning as
  `ml/extract_landmarks.py`'s "fresh landmarker per clip". Degrades to a typed
  `PerceptionUnavailable` (never a crash) if MediaPipe/OpenCV/the model file
  aren't usable.
  The tracker takes seconds to start, so it is **started when a language card is
  pressed**, in a worker thread, while the visitor reads the consent screen; signing can
  then begin the moment they agree (measured on a PC: consent to active in 53 ms, tracker
  start 3.3 s). It is given no frame until consent. A tracker that finishes starting
  after the page has gone, or after the visit ended, is closed and not attached.
  **A tracker must be closed, never left to the garbage collector:** a MediaPipe
  landmarker freed by Python's cycle collector deadlocks in its own `__del__` (measured
  2026-10-13; it hung the test suite). `HolisticPerception` closes on every exit path and
  has a finalizer that hands a forgotten one to a thread of its own to close
  (`tests/test_perception_lifecycle.py`). It also reports the frame's mean brightness,
  for the lighting warning (FR-K5).
  On MediaPipe 0.10.x (the Jetson ships 0.10.18) a frame with a person in view and hands
  down aborts the whole process; `install_empty_packet_guard` patches that, and logs
  `mediapipe_empty_packet_guard_not_installed` if a later 0.10 release renames what it
  patches.
- **B3 Segmentation** (`kiosk/segmentation.py`) — the IDLE/ACTIVE/BOUNDARY state
  machine from `SYSTEM_DESIGN.md` B3, thresholds in `kiosk/config.py` (env-var
  overridable), unit-tested in `tests/test_segmentation.py` and
  `tests/test_live_feedback.py`.
  **A word's length is measured in time, not frames** (since 2026-10-13): at least 3
  frames spanning 150 ms, at most 4.5 s. The old rule, "at least 6 frames", is 0.2 s at 30
  tracked frames a second but 0.6 s at the Jetson's ~10, and dropped ordinary quick signs
  without a word; figures in `docs/LIVE_PIPELINE.md`. The segmenter also counts the frames
  where the hands vanished mid-word (`Utterance.frames_without_hands`); a word that was
  more than half holes is refused with the reason `hands_lost` and never classified.
- **B4 Recognition** (`kiosk/recognition.py`) — one `Recognizer` per language (see
  "Languages" above), reusing `ml/model.py` and `core/features.py` unchanged. Rejection
  (`Hypothesis.gloss=None`) is explicit and tested (`tests/test_recognition.py`)
  with a hand-built checkpoint whose head bias makes the output deterministic, so
  the test doesn't depend on training having converged. Each model takes one throwaway
  forward pass at start-up (`warm_up`), so the first sign of the day does not pay
  PyTorch's first-call cost; `/health` reports the seconds per language in `warm_up_s`,
  and `null` there means that model failed to run.
- **B5 Assembly** (`kiosk/assembly.py`) — template/dictionary rendering from
  `core/vocab.json`, the single word list shared with `server/`'s KB editor.
- **B6 Output** (`kiosk/output.py`) — `TTSEngine.speak_async` returns a `Future`
  immediately; synthesis runs on one background worker thread. Proven
  non-blocking in `tests/test_output_ordering.py` with a deliberately slow fake
  engine.
- **B7 Dialogue** (`kiosk/dialogue.py`) — keyword-overlap retrieval over
  `kiosk/data/knowledge_base.json` (`docs/API_CONTRACTS.md §3` shape — **this file
  is genuinely shared with `server/`'s B14**; during this build another agent's
  process was live-editing it through what looks like a KB CRUD path, which is
  the intended integration point, not a bug — see "Found during integration"
  below). No match above `kb_similarity_floor` → refusal (never improvised).
  Above it: a real Claude API call (`claude-haiku-4-5`, 4s timeout, `max_retries=0`
  so a failure surfaces fast) phrases the retrieved fact bilingually; any
  exception at all falls through to the local-fallback template with a visible
  `source: "local_fallback"` flag.
- **B8 Sign playback** (`kiosk/playback.py`) — looks up each word in
  `core/clip_library.json`; every `clip_path` is `null` today, which renders as
  the "not yet recorded" placeholder (`.pw-placeholder` in the frontend),
  sequenced and highlighted the same way a real clip would be.
- **B9 Session/telemetry** (`kiosk/session.py`) — `Session` state machine
  (attract/language/consent/active/ended), a correlation id per session, JSON
  structured logs via stdlib `logging` (UTF-8-safe stream). `/health`'s `telemetry`
  carries the hottest thermal zone and this process's resident memory where the platform
  reports them (the Jetson; `"unavailable"` on Windows), plus `LiveStats`: `tracker_fps`
  (tracked frames a second over the last 5 s, `null` with nobody at the camera),
  `frame_ms`, `classify_ms`, and counts of outcomes (`committed`, `candidates`,
  `rejected`, `hands_lost`). `throttled` is still always `"unavailable"`.
  Three log lines are for judging a live session afterwards: `recognition` (one per sign:
  status, word, confidence, the bar it was held to, top 3, frames, frames without hands,
  span, classify time), `frame_timing` (every 100 frames: average frame cost, tracker
  rate, brightness) and `perception_ready` (tracker start-up seconds). None holds
  landmarks or identity.
- **What the visitor is told while signing** (`kiosk/main.py`, `assets/app.js`):
  `signing` messages move the camera badge between "I can see you", "signing…" and
  "reading your sign…" (NFR-X1); `framing` says when the picture is too dark, too
  bright, or shows no body (FR-K5); a rejection says why (`not_recognised` or
  `hands_lost`). `framing` and `signing` are sent when they change, not with every frame.
  The Arabic of these messages is unreviewed and waits for a Deaf reviewer.
- **B10 Data capture** (`kiosk/data_capture.py`) — on `consent_retention=true`,
  buffers landmarks/predictions/corrections in exactly `docs/API_CONTRACTS.md §4`'s
  shape to `kiosk/data/pending_uploads/<session_id>.json`. **Upload is wired
  (2026-10-10)** as store-and-forward: set `SIGNOO_SERVER_URL` (e.g.
  `http://localhost:8001`) and each finished session, plus anything left over at
  startup, is POSTed to `server/`'s `/ingest/sessions` off the event loop
  (`flush_pending_uploads`, stdlib `urllib`). A file is deleted only after a 2xx;
  offline or rejected files stay queued. Left blank, nothing leaves the device.
  Every item carries an anonymous `visitor-<8 hex>` `signer_id` (one visit = one
  signer) so the server's signer-independent split works without identifying anyone.
  With `SIGNOO_SERVER_URL` set the kiosk also registers with the fleet and sends a
  heartbeat every `SIGNOO_HEARTBEAT_INTERVAL_S` (30 s) with uptime and, where the board
  reports it, the hottest thermal-zone temperature (`kiosk/fleet.py`). Set
  `SIGNOO_KIOSK_TOKEN` to the server's token; a production server refuses requests without it.
  A server in `per_kiosk` mode wants this kiosk's own token, not the master key: get it with
  `python -m server.scripts.kiosk_token <your SIGNOO_KIOSK_ID>` on the server.
  **From 2026-10-12 each heartbeat tick also asks the server for two things per language:**
  the assigned model (`kiosk/model_sync.py`) and the knowledge base (`kiosk/kb_sync.py`).
  A model the server later withdraws (a rollback) makes the kiosk **revert to the checkpoint
  it shipped with** — the only model it can fall back to offline. Synced knowledge bases are
  written under `SIGNOO_SYNCED_MODELS_DIR/kb/<code>.json` (never over the source-controlled
  seed in `kiosk/data/`), survive a restart, and a download that does not parse leaves the
  working one untouched.
- **A language trained only on generated data is not offered.** `core/languages.json` marks
  each language `real_data` true or false; a false one is refused (`language_unavailable`,
  card shows "coming soon") unless `SIGNOO_ALLOW_SYNTHETIC=1`. That is currently the case for
  `liu`, so **a development machine or demo must set `SIGNOO_ALLOW_SYNTHETIC=1` to see it** —
  the kiosk test suite does this itself (`kiosk/tests/conftest.py`). A deployment must not.
- **One browser connection at a time.** A new `/ws/session` connection replaces the old one
  (closed with code 4001) so a reloaded page or stray tab cannot leave a second MediaPipe
  instance running on the Jetson.

## Frontend

`index.html` is the six-screen visitor flow (hello → language → consent → capture →
answer → thank-you). `hello` (camera-based presence auto-detect) and `thank-you` are
client-side only: the backend's `SessionState` still has just
`attract|language|consent|active|ended` (see `docs/ROADMAP.md` drift item 6).
`assets/app.js` is the WebSocket client driving it, `assets/vision.js` trimmed to pure canvas drawing (perception moved
server-side, so the old in-browser MediaPipe loader is gone), `assets/kiosk.css`
added for the new screens.

**The overlay is corner marks, not a skeleton (owner's decision, 2026-10-12).** FR-K4
and the `deaf-kiosk-ux` skill both say "tracked skeleton overlay". The dots and lines
covered the visitor's own hands and face, so `vision.js` draws four white rounded
corners round each hand and round the face (pose points 0–10) and eases them between
tracker updates; the skeleton code is still there behind `SHOW_HANDS` / `SHOW_BODY`. It
answers the same question, "does the camera see me", but it is a departure from the
written requirement that a Deaf reviewer has not yet judged.

**Frames are paced to the tracker.** The page captures a frame every 70 ms, shrinks it to
480 px, and does not send one while the last is still unanswered (it gives up waiting
after 200 ms). Nothing queues and no frame is encoded only to be dropped. `assets/recognizer.js` (the old client-side
nearest-neighbour classifier) was deleted — confirmed unused by `record.js`
first, so the recording studio (`record.html`, untouched) is unaffected.
`assets/vocab.js` is kept only for `record.html`; the kiosk frontend fetches
`core/vocab.json` directly (the one list `docs/API_CONTRACTS.md §1` says is
shared), so gloss labels can't drift between the two.

**Stated gaps, not hidden** (FR-K1/FR-K3/FR-X4 call for sign video):
`.sign-video-note` placeholders mark every spot a real signed instructional clip
belongs — there is no such video yet, so these are explicit text labels, never a
faked video element.

## Replaying recordings through the live path (NFR-N1)

```bash
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -m kiosk.replay data/popsign/raw --language asl \
    --signers 7002 8012 --signs apple book dog --tracker-fps 10
```

Feeds each video to the same tracker, segmenter and recogniser the kiosk uses, the way
the browser would send it, and reports per clip what the kiosk would have done:
committed (right or wrong), offered candidates (with or without the true word), refused,
or produced no word at all. `--tracker-fps` gives the tracker only that many frames a
second, which is how a development machine shows what the Jetson's ~10 a second does to
a sign. Use validation signers only; the test signers stay untouched until a final model.

## Playing recordings to a running kiosk, as the browser would

```bash
export PYTHONIOENCODING=utf-8
# starts a backend itself:
.venv/Scripts/python.exe -m kiosk.live_check data/popsign/raw --language asl \
    --signers 7002 8012 --signs apple book dog
# or a kiosk that is already running, such as the Jetson:
.venv/Scripts/python.exe -m kiosk.live_check data/popsign/raw --language asl \
    --signers 7002 --url ws://192.168.100.60:8000/ws/session
```

`kiosk/replay.py` drives the tracker, segmenter and recogniser directly. This goes
through everything between them and the visitor as well: the JPEG the page sends and its
pacing, the WebSocket, the consent gate, and when results come back. It found two faults
on 2026-10-13 that the unit tests and the replay tool could not see. It prints, per clip,
the outcome a visitor would have seen, and at the end how many frames were sent against
how many results came back; those two should be close. It takes over the kiosk's one
connection, so run it against a kiosk only when no visitor is there. Consent is sent with
retention off. **Run this, not only the unit tests, after any change to the frame path.**

## Sending it to the Jetson

```bash
scripts/deploy-jetson.sh --dry-run    # what would be sent
scripts/deploy-jetson.sh              # send, restart, check /health
```

Saves the Jetson's current copy first and prints the command that puts it back. Exits 2,
having changed nothing, when the Jetson cannot be reached. **Only the dry run and the
unreachable path have been run** (the Jetson was off the network on 2026-10-13); the
backup, send, restart and health-check steps are untested until it is used for real.

## Verification performed

- `.venv/Scripts/python.exe -m pytest kiosk/tests -q` — all passing (run it for the
  current count; 16 at the first build, over a hundred by 2026-10-13):
  buffer semantics, segmentation state transitions, recognition rejection vs.
  commit (deterministic toy checkpoint), TTS non-blocking ordering, dialogue
  refusal/fallback, and `core/features.py` train/inference bit-for-bit parity
  (`tests/test_features_parity.py` — the check `SYSTEM_DESIGN.md §7` demands).
- `kiosk/tests/test_smoke_pipeline.py` — builds a synthetic clip the same way
  `ml/make_synthetic.py` does, feeds it through the real `Segmenter` and the real
  `models/signoo.pt` checkpoint, and prints the resulting `Hypothesis`. **Caveat
  stated plainly**: a synthetic "sign" is an abstract per-axis sine pattern, not
  real Jordanian Sign Language — in the run recorded during this build it came
  back a **rejection** (confidence 0.276, below the 0.55 floor), which is exactly
  the expected, exercised outcome for made-up input, not the "committed" path.
  This proves the pipeline wiring (frames → landmarks → features → hypothesis →
  text → answer → playback), not recognition accuracy — no accuracy claim is
  possible before Phase 6's real data (`slr-methodology` skill).
- Live end-to-end run: started `uvicorn kiosk.main:app`, confirmed
  `GET /health` (`model_loaded: true`), `GET /` and `GET /assets/app.js` serve
  through the mounted static root, and drove the full WebSocket protocol by hand
  (`kiosk/tests/manual_ws_check.py`) — language select → consent → several
  frames (blank frames correctly produced `framing: {ok: false, reason:
  "no_body"}`, proving the honest "can't see you" path, not a silent hang) →
  `control: clear` → `ask` (no signed words → the dialogue path correctly
  **refused**, not guessed) → `control: end_session` → `session_reset`. This
  also exercises the consent gate (frames are ignored until `consent.processing`
  is true) and that declining retention (`retention: false`, sent in this same
  run) still completes a full session.
- `test_output_ordering.py` is the code-level proof that text-shown always
  precedes TTS completion (`speak_async` returns in <50ms against a fake engine
  that takes 300ms to "speak").

### What I could not verify

No physical camera or speaker was available in this environment, so the actual
`getUserMedia` capture path, the skeleton overlay rendering, and audible TTS
output were not eyeballed in a real browser — only the server-side halves
(perception on synthetic/blank frames, `pyttsx3` invoked via a fake engine in the
ordering test) were exercised. `pyttsx3.init()` itself was not called for real in
this Linux/CI-like sandbox (no host TTS engine); it is wrapped in `output.py`'s
try/except so a missing engine degrades to a logged failure, never a crash, but
that specific path was not driven end-to-end.

## Found during integration, outside this phase's remit

- `kiosk/data/knowledge_base.json` was being live-edited by another process
  during this build (entries appearing/changing while this work was in
  progress, one with garbled non-UTF-8-safe Arabic text) — almost certainly
  `server/`'s B14 KB CRUD path pointed at the same shared file, which is the
  correct integration per `docs/API_CONTRACTS.md §3`. Left as-is per instruction
  not to fight over a file another agent owns; flagging only that whichever
  write path produced the `"?????"` entry should ensure UTF-8 encoding on write.
- `ml/make_synthetic.py`, `ml/train.py` (and by inspection every `print()` in
  `ml/`) crash on Windows with `UnicodeEncodeError` on `→`/Arabic text unless
  `PYTHONIOENCODING=utf-8` is set first — this is the exact latent issue named in
  this task's brief as pre-existing in `ml/`. Not fixed here (out of `kiosk/`'s
  remit); worked around with the env var in every command above.
- Training synthetic data on this machine hit a hard `MemoryError` mid-run
  (`ml/train.py`, epoch ~15) — the environment is memory-constrained enough that
  even routine shell commands intermittently stalled or failed during this
  build. The checkpoint saved at the best epoch before the crash (100% synthetic
  val accuracy — expected and meaningless per `slr-methodology`, since synthetic
  signs are trivially separable) was sufficient for pipeline verification. Not a
  `kiosk/` or `ml/` code defect as far as could be determined — looked like host
  memory pressure, not a leak in the training loop.

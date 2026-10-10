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

## Languages (one model per sign language)

`core/languages.json` lists every card on the language screen with its `model`,
`vocab`, `status` (`live` | `pilot` | `coming_soon`) and `real_data` (was the model
trained on real human signing). `kiosk/languages.py` loads each model that exists at
startup. A language is **ready** only if its checkpoint and vocabulary load;
`GET /languages` reports which are ready, and the frontend only lets those cards be
pressed. Pressing a card puts that visitor on that language's model and vocabulary;
selecting a language that is not ready returns `language_unavailable` and changes
nothing, so one language is never quietly served by another's model.

To bring a language live: train a checkpoint (`ml/train.py ... --out models/<name>.pt`),
add its vocabulary file under `core/`, set `model` and `vocab` in `languages.json`, and
restart. `SIGNOO_MODEL_PATH` still overrides the Jordanian (`liu`) checkpoint.
`models/asl_pilot.pt` is the American pilot, trained on PopSign (CC BY 4.0) phone-camera
clips, a handful of signs only; it is shown as "Pilot" and is not a Jordanian model.

## Wire protocol (`/ws/session`)

JSON text frames both ways; the full reference lives as a docstring at the top of
`kiosk/main.py`. Design choice: camera capture and rendering stay in the browser
(`getUserMedia`); every processed frame is sent to this backend as **base64
JPEG**, and *all* of B2 Perception, B3 Segmentation, B4 Recognition, B5 Assembly,
B7 Dialogue, and B8 Playback run server-side. The browser only draws what it's
told (skeleton overlay, recognised words, candidates, the answer, playback
placeholders) — this keeps the model, the KB, and the clip library in one place
and matches `SYSTEM_DESIGN.md`'s "no camera handles, no model objects... crossing
a boundary" rule at the process boundary too, not just between Python modules.

## Architecture notes (one per block)

- **B1 Capture** (`kiosk/buffers.py`) — `SingleSlotBuffer`: newest-wins, always
  drops rather than queues. Used both for MediaPipe's async LIVE_STREAM results
  (`kiosk/perception.py`) and is unit-tested directly (`tests/test_buffers.py`).
- **B2 Perception** (`kiosk/perception.py`) — MediaPipe Holistic in **LIVE_STREAM**
  mode (not `ml/extract_landmarks.py`'s VIDEO mode — that mode is for a finite
  recorded file; LIVE_STREAM is the Tasks API's mode for a live, one-frame-at-a-time
  source, and is asynchronous via `detect_async`/`result_callback`). One
  `HolisticPerception` instance per WebSocket connection, closed on session end —
  tracking state must not bleed between visitors, same reasoning as
  `ml/extract_landmarks.py`'s "fresh landmarker per clip". Degrades to a typed
  `PerceptionUnavailable` (never a crash) if MediaPipe/OpenCV/the model file
  aren't usable.
- **B3 Segmentation** (`kiosk/segmentation.py`) — the IDLE/ACTIVE/BOUNDARY state
  machine from `SYSTEM_DESIGN.md` B3, thresholds in `kiosk/config.py` (env-var
  overridable), unit-tested in `tests/test_segmentation.py`.
- **B4 Recognition** (`kiosk/recognition.py`) — loads `models/signoo.pt`, reuses
  `ml/model.py`'s `SignTransformer` and `core/features.py` unchanged. Rejection
  (`Hypothesis.gloss=None`) is explicit and tested (`tests/test_recognition.py`)
  with a hand-built checkpoint whose head bias makes the output deterministic, so
  the test doesn't depend on training having converged.
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
  structured logs via stdlib `logging` (UTF-8-safe stream). Temperature/throttle
  are reported as the string `"unavailable"`.
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

## Frontend

`index.html` is the six-screen visitor flow (hello → language → consent → capture →
answer → thank-you). `hello` (camera-based presence auto-detect) and `thank-you` are
client-side only: the backend's `SessionState` still has just
`attract|language|consent|active|ended` (see `docs/ROADMAP.md` drift item 6).
`assets/app.js` is the WebSocket client driving it, `assets/vision.js` trimmed to pure canvas drawing (perception moved
server-side, so the old in-browser MediaPipe loader is gone), `assets/kiosk.css`
added for the new screens. `assets/recognizer.js` (the old client-side
nearest-neighbour classifier) was deleted — confirmed unused by `record.js`
first, so the recording studio (`record.html`, untouched) is unaffected.
`assets/vocab.js` is kept only for `record.html`; the kiosk frontend fetches
`core/vocab.json` directly (the one list `docs/API_CONTRACTS.md §1` says is
shared), so gloss labels can't drift between the two.

**Stated gaps, not hidden** (FR-K1/FR-K3/FR-X4 call for sign video):
`.sign-video-note` placeholders mark every spot a real signed instructional clip
belongs — there is no such video yet, so these are explicit text labels, never a
faked video element.

## Verification performed

- `.venv/Scripts/python.exe -m pytest kiosk/tests -q` — 16 tests, all passing:
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

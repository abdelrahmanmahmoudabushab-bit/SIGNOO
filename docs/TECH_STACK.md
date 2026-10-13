# SIGNOO — Tech Stack

**Version** 0.1 · companion to `SYSTEM_DESIGN.md`, organised by the five blocks
(`SYSTEM_DESIGN.md` §1.1a). States what's decided, what's running today, and what's
still open — an undecided row is not a gap in this document, it's a gap in the
project, named rather than hidden.

---

## 1. Presence & Capture

| Component | Choice | Status |
|---|---|---|
| Camera capture | Browser `getUserMedia` (dev/kiosk frontend); CSI camera on Jetson in prod | Running (dev) |
| Landmark extraction | MediaPipe Holistic (upper-body pose + 2×21 hand landmarks) | Running server-side in `kiosk/perception.py`, LIVE_STREAM mode, CPU build. **About 10 tracked frames a second on the Jetson (measured 2026-10-12)**, a third of the 30 the design assumes. The Jetson ships MediaPipe 0.10.18; training clips were extracted with 1.0.1, and what that difference costs is not measured |
| Edge inference runtime | TensorRT (target), plain PyTorch and CPU MediaPipe (current) | **Not yet converted** — Phase 0's perception criterion (30 fps hot, on-device) is unmet, and the rate above shows by how much. The largest open engineering item on the kiosk; needs the device |
| Presence/attract detection | **Decided 2026-10-08**: frame-difference motion heuristic on a low-res camera sample (`assets/app.js`'s `startPresenceWatch`) — no extra model, works offline. A dedicated PIR/ultrasonic sensor remains a possible hardware upgrade later, not required | Built, browser/Electron-side; not yet verified on the actual Jetson camera path. It keeps the camera on at the idle screen, before consent, with an on-screen notice and nothing sent or stored; that has had no privacy review and reads against FR-K3's "before the camera starts" (`SYSTEM_DESIGN.md` §8, question 9) |
| Target hardware | NVIDIA Jetson Orin Nano Super, 8 GB shared LPDDR5 | The kiosk runs on it as two systemd user services. Not yet bench-tested hot (`jetson-realtime` skill has the budget). It could not be reached on 2026-10-13: the development PC was on a different network (`192.168.1.x`) from the Jetson's (`192.168.100.60`) |
| Getting code onto it | `scripts/deploy-jetson.sh`: backup on the device, send `kiosk/`, `core/`, `assets/`, `index.html` and the configured models, restart, check `/health` | Written 2026-10-13. Only its dry run and its "cannot reach the Jetson" path have been run |

## 2. Understanding

| Component | Choice | Status |
|---|---|---|
| Segmentation | Pause-based state machine (`kiosk/segmentation.py`). Kiosk-wide default thresholds in `kiosk/config.py`; since 2026-10-11 each language may override them with a `segmentation` entry in `core/languages.json`. Since 2026-10-13 a word's length is a time (≥3 frames over 150 ms, ≤4.5 s), not a frame count, so it means the same at any tracker speed | Running, tested. No language sets its own values yet, so all run on the defaults; the pause and motion defaults were tuned on synthetic data. Measured with `kiosk/replay.py --tracker-fps` (`docs/LIVE_PIPELINE.md` §15) |
| Recognition model | Custom Transformer encoder, ~1.27M params (`ml/model.py`). **Revised 2026-10-11**: one complete model per language, each its own ~5.5 MB checkpoint with its own confidence bars, listed in `core/languages.json`. This replaces "shared backbone + per-language head" (`SYSTEM_DESIGN.md` §1.1b, B4) | Two checkpoints load today: `liu` (`models/signoo.pt`, synthetic data only, marked `coming_soon` and not offered to visitors) and `asl` (`models/asl50.bars.pt`, 50 words, marked `pilot`, real data; about 71% top-1 and 87% top-3 on validation signers, under NFR-Q1 and NFR-Q3). Memory with several languages loaded is not measured on the Jetson |
| Feature representation | Body-relative, scale/speed-normalised landmark features (`core/features.py`), shared bit-for-bit between `ml/` (training) and `kiosk/` (inference) | Running, parity-tested (`test_features_parity.py`) |
| Assembly (gloss → text) | Template + dictionary rendering (`kiosk/assembly.py`, `core/vocab.json`) — deliberately not a learned model yet | Running |
| Serving | FastAPI + WebSocket, one process per kiosk (`kiosk/main.py`) | Running |

## 3. Response

| Component | Choice | Status |
|---|---|---|
| Text-to-speech | Piper (local, offline), Arabic + English | Wrapped non-blocking in `kiosk/output.py` (`TTSEngine.speak_async`); not audibly verified on real hardware yet |
| Dialogue / Q&A | Keyword-overlap retrieval over the selected language's own knowledge base (2026-10-11: one file per language, named in `core/languages.json`; `kiosk/data/knowledge_base.json` is `liu`'s, and a language without one refuses every question), phrased bilingually by Claude (`claude-haiku-4-5`, 4s timeout) when above the similarity floor; local-fallback template on any failure | Running, refusal path tested |
| Sign playback / avatar | **Revised 2026-10-07**: was a flat per-word video clip library (every `clip_path` still `null` — no clips recorded); now a 3D rigged character (face/body/hands) driven by **retargeted recorded landmark motion**, never generative synthesis (`SYSTEM_DESIGN.md` B8) | **Not built.** Retargeting (landmarks → rig) and the renderer are the open spike |
| 3D rig / renderer | **Undecided.** Kiosk frontend is already browser-based (served by `kiosk/main.py`'s static mount), so a WebGL renderer (e.g. Three.js + a rigged GLTF/GLB humanoid) is the natural fit — avoids adding a second runtime process. Not yet chosen or prototyped | **Not built** |
| Retargeting method | **Undecided.** Needs a mapping from MediaPipe Holistic landmarks (what's already recorded) onto an arbitrary rig's skeleton, robust to differing signer body proportions | **Not built**, flagged in `SYSTEM_DESIGN.md` B8 as needing a Deaf-reviewer correctness check before any word is trusted live |

## 4. Kiosk Experience

| Component | Choice | Status |
|---|---|---|
| Frontend | Plain HTML/CSS/vanilla JS (`index.html`, `assets/app.js`, `assets/kiosk.css`) — no framework, deliberate per repo convention | Running, six screens (hello/language/consent/capture/answer/thank-you), 9-card language grid. Every card starts disabled and is enabled from the backend's `GET /languages`: since 2026-10-13, 1 pressable on a production kiosk (`asl`, pilot) and 8 "coming soon", `liu` among them. The camera overlay is corner marks round the hands and face, not the skeleton FR-K4 names (owner's decision 2026-10-12); the camera badge shows watching / signing / reading, and banners say too dark, too bright, no body, and "couldn't see your hands". None of the 2026-10-13 screen changes has been seen in a browser, and their Arabic is unreviewed |
| App packaging | Electron shell (`electron/`) wraps the above as an installed fullscreen kiosk app — spawns the backend, auto-grants camera permission, shows an honest "starting up" screen (not a browser error page) until `/health` answers, and keeps watching + respawning for the app's whole life, not just at launch | Built and hardened 2026-10-08 (self-healing: backend crash → auto-respawn → auto-recover, no restart needed); not yet packaged for distribution (see `electron/README.md`'s "not solved here") |
| Session state machine | `kiosk/session.py`, JSON-structured logs, correlation ID shared with `server/` | Running. Corrected 2026-10-08: no backend gap here after all — the frontend's `hello`/`thankyou` screens map directly onto the existing `ATTRACT`/`ENDED` states (hello *is* attract; thank-you is the moment between `_end_session()`'s `ENDED` transition and the next visitor's fresh `Session()`), so no new enum values were needed |
| Wire protocol | JSON over one WebSocket (`/ws/session`), documented in `kiosk/main.py`'s module docstring and `docs/API_CONTRACTS.md` | Running |
| Local data capture | Opt-in landmark buffering to `kiosk/data/pending_uploads/*.json` (`kiosk/data_capture.py`) | Running. Corrected 2026-10-11: upload is wired. With `SIGNOO_SERVER_URL` set, queued sessions are POSTed to the server's `/ingest/sessions` with the `X-Kiosk-Token` header and deleted only after a 2xx (`kiosk/data_capture.py`); left blank, nothing leaves the device |
| Language menu | **Rebuilt, corrected 2026-10-11.** `core/languages.json` lists nine language engines; `kiosk/languages.py` loads each one and `GET /languages` tells the frontend which cards are ready. Pressing a card switches the connection to that language's whole engine; an unready language is refused with `language_unavailable` | Running. Fixed 2026-10-12/13: a language whose model was not trained on real signing is not offered (unless `SIGNOO_ALLOW_SYNTHETIC=1`), and consent is refused on a language that may not be offered. Remaining: a connection still sits on the `liu` engine until a card is pressed (`SYSTEM_DESIGN.md` §1.1b) |
| Telemetry (B9) | `/health` on the kiosk: temperature and resident memory where the platform reports them, tracker rate, stage timings, outcome counts, each model's start-up pass (`kiosk/session.py::LiveStats`) | Running since 2026-10-13. Not sent in the heartbeat, so not in the console, and nothing alerts on it |

## 5. Learning Loop (server + console)

| Component | Choice | Status |
|---|---|---|
| API framework | FastAPI (`server/main.py`), routers per concern (`auth`, `fleet`, `ingest`, `kb`, `registry`, `review`, `training`) | Implemented, committed. 28/28 tests passing on 2026-10-08; the suite has since grown to 12 test files (88 test functions counted 2026-10-13) and was last run in full on 2026-10-12. Use `server/.venv`, a separate venv from `kiosk/`'s, per `server/requirements.txt`'s own header comment; using the wrong one is what once looked like a missing dependency |
| Database | SQLAlchemy + Alembic migrations; SQLite dev (`server/signoo_dev.db`), Postgres intended for prod | Implemented |
| Auth | JWT via `python-jose`, role-scoped (`server/auth.py`) | Implemented |
| Promotion gate | Reuses `ml/evaluate.py` / `ml/compare_splits.py` logic as subprocesses (`server/gate.py`) against a frozen benchmark | Implemented; **the frozen benchmark itself doesn't exist yet** — blocks switching on the 3-day auto-retrain cadence (`docs/ROADMAP.md` R3) |
| Console (admin UI) | React + TypeScript + Vite, `vitest` for tests | Implemented and committed (`a7ffa3f`); 17 tests passing when last confirmed (2026-10-08). Its `KBEntry` type has no `language` field yet, so it cannot say which language a knowledge-base entry is for |
| Fleet transport | mTLS, store-and-forward (design). Corrected 2026-10-11: store-and-forward upload, registration and heartbeat are implemented over plain HTTP with a shared `X-Kiosk-Token` secret (`docs/API_CONTRACTS.md` §9). mTLS is not implemented. Since 2026-10-12 a `per_kiosk` mode gives each kiosk its own token | Partly implemented. Per-language model and knowledge-base distribution to kiosks is built and tested on both sides (`docs/API_CONTRACTS.md` §10), including a rollback that returns kiosks to their previous model; none of it has run against a real fleet |
| Server deployment | Docker Compose on a DigitalOcean droplet: Postgres, the API, Caddy for HTTPS and the console (`deploy/`) | Written 2026-10-12. **Never built or run**: no Docker build, no Postgres migration, no droplet |

## Cross-cutting

- **Language**: Python 3.11 (`kiosk/`, `server/`, `ml/`, `core/`), TypeScript (`console/`)
- **Training**: PyTorch, synthetic data generator (`ml/make_synthetic.py`) for pipeline tests only — never a stand-in for a real accuracy claim (`slr-methodology` skill)
- **Testing**: `pytest` (`kiosk/tests`, `server/tests`), `vitest` (`console`)
- **Windows dev gotcha**: default console codepage (cp1252) crashes on Arabic text; `PYTHONIOENCODING=utf-8` required for any `ml/` script run directly (`kiosk/session.py`'s logger already handles this for the kiosk process)
- **No cloud dependency for core service, with one bootstrap caveat**: the B2 Perception model (`kiosk/perception.py`) is vendored at `vendor/models/holistic_landmarker.task` (already present in this repo) and loaded from disk — zero network calls once provisioned. If that file is ever missing, `perception.py` falls back to downloading it from `storage.googleapis.com` on the spot; a kiosk provisioned without it would need network for that one-time fetch (`scripts/fetch-vendor.sh` is the intended provisioning step). Beyond that, the only other network call anywhere in the kiosk path is the optional Claude API call in B7 dialogue, which has a local-fallback template on any failure. `index.html` deliberately does not load fonts from a CDN (removed 2026-10-08) for the same reason — an offline app shouldn't spend startup time on a doomed connection

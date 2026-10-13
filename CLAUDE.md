# SIGNOO

Jordanian Sign Language (LIU) kiosk translator, expanding to a multi-language sign
kiosk. Read this file every session — it is the short version; the long version is
`docs/`.

## What this is

A public kiosk a Deaf visitor walks up to, signs at, and gets understood and answered
in sign. Nothing here is decorative accessibility tooling — sign languages are full
languages, and the system is held to the standard of getting them right, not the
standard of "better than nothing."

## Source of truth, in order

1. `docs/REQUIREMENTS.md` — what the system must do (FR-/NFR- numbered)
2. `docs/SYSTEM_DESIGN.md` — the architecture: 5 blocks, 16 B-numbers underneath (§1.1a
   has the 5-block summary; §2/§3 have the B1–B16 detail)
3. `docs/TECH_STACK.md` — what's built with what, and what's still undecided
4. `docs/ROADMAP.md` — phases, current state vs. plan, the next two weeks
5. `docs/API_CONTRACTS.md` — the shapes crossing `kiosk/` ↔ `server/` ↔ `console/`
6. `docs/DATA_COLLECTION.md`, `docs/TRAINING.md` — dataset and model specifics
7. `docs/LIVE_PIPELINE.md` — how one sign travels from camera to caption as built, what
   has been measured on it, and a backlog of ideas (ideas, not commitments)

If a doc and the code disagree, say so out loud rather than picking one silently —
both `docs/ROADMAP.md` and `kiosk/README.md` have sections written specifically to
surface that kind of drift; add to them, don't paper over them.

## Non-negotiables

These come from `docs/SYSTEM_DESIGN.md` and the skills below, repeated here because
getting them wrong is worse than most bugs:

- **No generative sign synthesis.** B8 (sign playback / the avatar) only ever plays
  back real recorded motion, retargeted onto a rig. A system that generates
  plausible-but-wrong signing is actively harmful to a Deaf visitor who has no way to
  detect the error. See the `signoo-avatar` skill before touching B8.
- **Split by signer, never randomly**, in every train/val/test split, no exceptions.
  See the `slr-methodology` skill.
- **Landmarks by default, not video**, for anything retained past a session. Video
  retention is a separate, explicit, time-boxed opt-in.
- **Every kiosk state must be visible** — no audio-only feedback, ever. See the
  `deaf-kiosk-ux` skill.
- **A language card that isn't trained on real data says "coming soon."** Never run an
  untrained or synthetic-only model live and call it recognition.
- **The automatic retraining gate (B13) cannot be bypassed.** 3-day cadence is
  automatic; promotion is not.

## Skills (load before the matching work)

- `deaf-kiosk-ux` — any kiosk screen, flow, wording, or error state
- `signoo-engineering` — module boundaries, error handling, testing, "done" means ran it
- `jetson-realtime` — anything running on-device: latency, memory, TensorRT, thermal
- `slr-methodology` — training, evaluating, or reporting on any recognition model
- `signoo-avatar` — B8, retargeting, the signing character, anything avatar/rig-related
- `signoo-language-scaling` — adding or prioritising a language pack

## Commands

```bash
# kiosk (needs its own venv — see kiosk/README.md, system python has no numpy/torch)
.venv/Scripts/python.exe -m pytest kiosk/tests -q
.venv/Scripts/python.exe -m uvicorn kiosk.main:app --port 8000

# kiosk, as the installed app (spawns the backend above itself — see electron/README.md)
cd electron && npm install && npm start

# server — has its OWN venv, separate from kiosk/'s (server/requirements.txt's
# own header comment says so). Using the wrong venv/bare `python` here looks
# exactly like a missing-dependency failure and isn't one — confirmed 2026-10-08.
server/.venv/Scripts/python.exe -m pytest server/tests -q
server/.venv/Scripts/python.exe -m uvicorn server.main:app --port 8001

# console
cd console && npm test -- --run
cd console && npm run dev

# everything at once (run alone; the server's gate tests train models and need free RAM)
scripts/test-all.sh          # or: scripts/test-all.sh fast

# recorded clips played to the kiosk over its real WebSocket, as the browser would
# (validation signers only). Run this after any change to the frame path: it found two
# faults on 2026-10-13 that every unit test passed over. Add --url ws://<kiosk>:8000/ws/session
# to put the clips through a running kiosk such as the Jetson.
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m kiosk.live_check data/popsign/raw \
    --language asl --signers 7002 8012 --signs apple book dog hello milk

# the tracker, segmenter and model alone, at the Jetson's tracker speed
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m kiosk.replay data/popsign/raw \
    --language asl --signers 7002 8012 --tracker-fps 10

# send the kiosk to the Jetson (backup, restart, health check); --dry-run to preview
scripts/deploy-jetson.sh
```

A kiosk test that leaves a tracker (`HolisticPerception`) unclosed can hang the whole
run: MediaPipe deadlocks when Python's garbage collector frees one. Close what you open.

Deployment settings that matter (details in `server/README.md` and `kiosk/README.md`):
server refuses to start without `SIGNOO_JWT_SECRET` and `SIGNOO_KIOSK_TOKEN` (or
`SIGNOO_ENV=dev`); a kiosk reports to the server only when `SIGNOO_SERVER_URL` is set,
sending `SIGNOO_KIOSK_TOKEN`. Each sign language has its own model, listed in
`core/languages.json`; a language is pressable only when its checkpoint loads.

Windows: set `PYTHONIOENCODING=utf-8` before running anything in `ml/` directly — the
console defaults to cp1252 and crashes on Arabic text. `kiosk/session.py`'s logger
already handles this; raw scripts do not.

## Repo layout

`kiosk/` B1–B10 on-device runtime · `electron/` kiosk app shell wrapping the frontend
below · `server/` B11–B16 backend · `console/` admin UI ·
`core/` shared features/contracts/vocab between `kiosk/` and `ml/` · `ml/` training
pipeline · `index.html`/`assets/` the kiosk's browser frontend, served by
`kiosk/main.py` · `data/`, `models/` datasets and checkpoints, both gitignored. For
Jordanian (`liu`) they are synthetic only: there is no real Jordanian signer data yet,
and that is the project's actual bottleneck. American (`asl`) has real public data
(PopSign, CC BY; ASL Citizen, research-only licence) and a 50-word pilot model.

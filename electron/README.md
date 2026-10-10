# electron/ — SIGNOO kiosk app shell

Wraps the existing browser frontend (`../index.html`, `../assets/*.js`, served by
`kiosk/main.py`) as an installed, fullscreen kiosk application instead of a browser
tab. The frontend itself is untouched — this is packaging, not a rewrite.

## What it does

1. Spawns the kiosk backend (`uvicorn kiosk.main:app`) as a child process, using the
   venv path documented in `kiosk/README.md`.
2. Opens a fullscreen, chrome-less (`kiosk: true`) window immediately, showing
   `offline.html` — a styled "starting up" screen in the kiosk's own visual
   language, not a generic browser error page — until the backend actually answers.
3. Polls `GET /health` in the background; the moment it answers, swaps the window
   over to the real kiosk UI automatically.
4. Keeps polling for the app's entire lifetime (every 5s), not just at startup — if
   the backend ever dies mid-session, it falls back to `offline.html` and switches
   back once healthy again, with no restart needed.
5. Respawns the backend process itself (2s after it exits or fails to start,
   whichever happens) rather than leaving a dead process dead — otherwise
   "retrying automatically" on the offline screen would be true of the health
   check but not of anything actually coming back.
6. Auto-grants the camera permission request — there's no person present on a public
   kiosk to click "Allow".

## Setup

```bash
cd electron
npm install
npm start
```

Needs the Python venv described in `kiosk/README.md` to already exist at
`<repo root>/.venv` (or point `SIGNOO_BACKEND_CMD`/`SIGNOO_BACKEND_ARGS` at wherever
it actually lives).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SIGNOO_KIOSK_PORT` | `8000` | Port the backend is told to listen on, and the port this app loads |
| `SIGNOO_BACKEND_CMD` | `<repo>/.venv/Scripts/python.exe` (Win) / `.venv/bin/python` (else) | The Python interpreter to run the backend with |
| `SIGNOO_BACKEND_ARGS` | `-m uvicorn kiosk.main:app --port <PORT>` | Args passed to that interpreter |
| `SIGNOO_SKIP_BACKEND_SPAWN` | unset | Set to `1` when the backend runs as its own service (the Jetson's systemd units); this app then only waits for it and never spawns a second one |

Any other `SIGNOO_*` variable in this app's environment is passed straight to the
backend it spawns. When the backend runs as a separate service instead, set them in that
service's environment. The ones a deployed kiosk needs:

| Variable | Purpose |
|---|---|
| `SIGNOO_SERVER_URL` | Server base URL. Without it the kiosk keeps consented sessions on disk and sends no heartbeat |
| `SIGNOO_KIOSK_TOKEN` | Shared secret the server requires (a production server refuses to start without one) |
| `SIGNOO_KIOSK_ID` | Name this kiosk reports; unique per kiosk |
| `SIGNOO_MODEL_PATH` | Overrides the Jordanian (`liu`) checkpoint; other languages are in `core/languages.json` |

Full list and behaviour: `kiosk/README.md` and `server/README.md`.

## A gotcha specific to VSCode-integrated terminals

Launching from a terminal that's itself inside an Electron app (VSCode, or an agent
running as a VSCode extension) can silently fail with `TypeError: Cannot read
properties of undefined (reading 'whenReady')`. Cause: VSCode sets
`ELECTRON_RUN_AS_NODE=1` in its own process environment, child terminals inherit it,
and Electron treats that as "run as plain Node, not as the Electron app" — so
`require('electron')` returns a path string instead of the real `app`/`BrowserWindow`
API. Fix: unset it before launching —

```bash
env -u ELECTRON_RUN_AS_NODE npm start      # bash
```
```powershell
Remove-Item Env:\ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue; npm start
```

A real double-click launch (not through a dev terminal) never hits this.

## Exiting kiosk mode

`Ctrl+Shift+Q` quits the app. There is no other way out of a fullscreen kiosk window
with no menu bar — document this key combo physically at the deployment site, not
just here.

## Not solved here (stated plainly, not hidden)

- **Packaging/distribution.** This runs against a dev-machine Python venv; bundling a
  real Python runtime so the app is a single installable artifact (e.g. via
  `electron-builder` plus a bundled interpreter, or PyInstaller-ing the backend
  separately) hasn't been done.
- **Auto-launch on boot.** Wiring this as the thing a Jetson/kiosk PC boots straight
  into (replacing a desktop session entirely) is an OS-level systemd/autostart
  concern, not an Electron one, and isn't configured here.
- **Backend respawn has no backoff or retry cap.** A backend that fails every time
  (e.g. a genuinely broken venv) respawns every 2 seconds forever rather than giving
  up or backing off — noisy in the logs, but not unsafe. Fine for now; revisit if it
  ever matters in practice.
- **Presence auto-detect** lives in `assets/app.js` (camera-based motion heuristic on
  the hello screen) and works the same whether loaded in a browser or this shell —
  nothing kiosk-app-specific was needed for it beyond the camera permission grant
  above.

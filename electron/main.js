// SIGNOO kiosk — Electron shell.
//
// Wraps the existing browser frontend (../index.html, served by kiosk/main.py's
// FastAPI process) as a real installed app: fullscreen, no browser chrome,
// launches on its own. The frontend itself (HTML/CSS/vanilla JS) is unchanged —
// this file's only job is: start the backend, wait until it's actually serving,
// open a kiosk window pointed at it, and auto-grant the camera permission a
// kiosk has no human present to click "Allow" on.
//
// signoo-engineering: "prefer the boring choice" — this spawns the same
// `uvicorn kiosk.main:app` command documented in kiosk/README.md rather than
// re-implementing the backend lifecycle; the backend's own code is untouched.

const { app, BrowserWindow, session, globalShortcut } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const http = require('node:http');

const REPO_ROOT = path.resolve(__dirname, '..');
const KIOSK_PORT = process.env.SIGNOO_KIOSK_PORT || 8000;
const KIOSK_URL = `http://127.0.0.1:${KIOSK_PORT}/`;

// Matches kiosk/README.md's documented venv layout. Override with
// SIGNOO_BACKEND_CMD/SIGNOO_BACKEND_ARGS if the venv lives somewhere else (e.g.
// a packaged build with its own bundled Python — not solved here, see README).
const DEFAULT_PYTHON = process.platform === 'win32'
  ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
  : path.join(REPO_ROOT, '.venv', 'bin', 'python');
const BACKEND_CMD = process.env.SIGNOO_BACKEND_CMD || DEFAULT_PYTHON;
const BACKEND_ARGS = process.env.SIGNOO_BACKEND_ARGS
  ? process.env.SIGNOO_BACKEND_ARGS.split(' ')
  : ['-m', 'uvicorn', 'kiosk.main:app', '--port', String(KIOSK_PORT)];

let backendProcess = null;
let mainWindow = null;
let appQuitting = false;

function startBackend() {
  backendProcess = spawn(BACKEND_CMD, BACKEND_ARGS, {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },   // kiosk/README.md's Windows note
  });
  backendProcess.stdout.on('data', d => process.stdout.write(`[backend] ${d}`));
  backendProcess.stderr.on('data', d => process.stderr.write(`[backend] ${d}`));

  // Without a listener, a missing/renamed venv (ENOENT) is an *unhandled*
  // 'error' event on the ChildProcess — Node treats that as an uncaught
  // exception and crashes the whole Electron main process. Both failure
  // modes (a bad path, 'error' only; a process that started and later
  // crashed, 'exit' only) land in onBackendDown so the offline page's
  // "retrying automatically" is true either way, not just on crash-after-
  // start. `handled` guards against 'error' and 'exit' both firing for the
  // same failure on some platforms, which would otherwise schedule two
  // respawns fighting over the same port.
  let handled = false;
  const onBackendDown = reason => {
    if (handled) return;
    handled = true;
    console.error(reason);
    backendProcess = null;
    if (!appQuitting) {
      console.error('respawning backend in 2s...');
      setTimeout(startBackend, 2000);
    }
  };
  backendProcess.on('error', err =>
    onBackendDown(`backend failed to start (${err.message}) — is the venv set up? see kiosk/README.md`));
  backendProcess.on('exit', code => onBackendDown(`backend exited (code ${code})`));
}

// A single GET /health, resolving true/false — never throws. kiosk/main.py
// always answers 200 here regardless of model-load state (status: "ok" or
// "degraded" is in the body), so this only answers "is the process up and
// talking at all" — the model/degraded distinction is the kiosk's own
// frontend's job (SYSTEM_DESIGN.md §6), not this shell's.
function pingHealth(timeoutMs = 2000) {
  return new Promise(resolve => {
    const req = http.get(`${KIOSK_URL}health`, res => { res.resume(); resolve(true); });
    // destroy() with no argument emits no 'error' event (only destroy(err)
    // does) — without passing one here, a timed-out request would never hit
    // the handler below and this promise would hang forever, silently
    // stalling the retry loop on every slow/dead connection.
    req.setTimeout(timeoutMs, () => req.destroy(new Error('health check timed out')));
    req.on('error', () => resolve(false));
  });
}

function waitForBackend(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise(resolve => {
    const tryOnce = async () => {
      if (await pingHealth(2000)) return resolve(true);
      if (Date.now() > deadline) return resolve(false);
      setTimeout(tryOnce, 500);
    };
    tryOnce();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    fullscreen: true,
    kiosk: true,
    autoHideMenuBar: true,
    backgroundColor: '#fdf3f5',   // kiosk.css's paper tone — avoids a white flash on load
    webPreferences: { contextIsolation: true },
  });
  // `mainWindow?.foo()` below only guards null/undefined — a destroyed-but-
  // still-referenced BrowserWindow is neither, and calling a method on one
  // throws synchronously. Nulling the reference here is what makes those
  // optional-chained calls (from the health watchdog's async interval) an
  // actual no-op instead of an unhandled rejection that could take the whole
  // process down.
  mainWindow.on('closed', () => { mainWindow = null; });

  // No window chrome means no obvious way out for whoever maintains the
  // physical kiosk — an escape hatch that fails *visibly* (a key combo
  // documented here) beats one that silently traps the operator.
  globalShortcut.register('Control+Shift+Q', () => app.quit());
}

// Was: load KIOSK_URL unconditionally even if the backend never came up,
// which showed Chromium's generic "can't reach this page" error — accurate,
// but not the kiosk's own honest out-of-service screen, and gave a visitor
// no indication anyone would ever come fix it. Now: the window always shows
// something from the moment it opens (offline.html first), and a background
// watchdog swaps to the real app the moment the backend answers — including
// recovering automatically if the backend ever dies *after* a successful
// start, which matters for an unattended kiosk (ROADMAP.md NFR-R2).
let showingOffline = false;
const HEALTH_POLL_MS = 5000;

function showOfflinePage() {
  showingOffline = true;
  mainWindow?.loadFile(path.join(__dirname, 'offline.html'));
}

function showKiosk() {
  showingOffline = false;
  mainWindow?.loadURL(KIOSK_URL);
}

// Real bug, found live on the Jetson: flipping to offline.html on a single
// failed health check tore down the real, live, in-progress WebSocket
// session every time one check was merely slow — not actually down, just
// contending with camera encode + perception + inference on real hardware
// under real load. The backend log showed sessions starting and dying every
// 4-6 seconds, repeatedly — that was this, not a person. A single missed
// beat must never cost a visitor their session; only sustained failure
// should. Requiring several consecutive misses before declaring it down
// (while flipping back to the real app on the very first success) fixes
// this without giving up the original point of the watchdog — recovering
// automatically from a backend that's actually, lastingly dead.
let consecutiveFailures = 0;
const FAILURE_THRESHOLD = 3;   // ~15s of sustained failure (3 x 5s poll), not one slow beat

function startHealthWatchdog() {
  setInterval(async () => {
    if (!mainWindow) return;
    const healthy = await pingHealth(4000);
    if (healthy) {
      consecutiveFailures = 0;
      if (showingOffline) showKiosk();
    } else {
      consecutiveFailures += 1;
      if (consecutiveFailures >= FAILURE_THRESHOLD && !showingOffline) showOfflinePage();
    }
  }, HEALTH_POLL_MS);
}

app.whenReady().then(async () => {
  // A kiosk has no person present to click "Allow" on a camera permission
  // prompt (B1 Capture depends on it). This is the one blanket grant in the
  // whole app, and it's scoped to 'media' only.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media');
  });

  createWindow();
  showOfflinePage();   // something is on screen immediately, before the backend exists at all

  // Real deployments don't all look like a dev laptop: on the Jetson, the
  // backend runs as its own systemd service (process supervision, restart-
  // on-crash, survives reboot — all things worth having independently of
  // this app) rather than a `.venv` this app spawns itself. Without this
  // escape hatch, main.js would spend forever respawning a `.venv/bin/python`
  // that was never meant to exist here, fighting a backend it doesn't own.
  if (process.env.SIGNOO_SKIP_BACKEND_SPAWN !== '1') startBackend();

  const healthy = await waitForBackend();
  if (healthy) showKiosk();
  startHealthWatchdog();   // keeps watching for the rest of the app's life either way
});

app.on('window-all-closed', () => app.quit());
app.on('will-quit', () => {
  appQuitting = true;
  globalShortcut.unregisterAll();
  backendProcess?.kill();
});

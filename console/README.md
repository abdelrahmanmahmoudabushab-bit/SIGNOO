# SIGNOO console (Phase 4)

React + TypeScript operator console for SIGNOO (FR-W1–FR-W9 in
`docs/REQUIREMENTS.md` §3.4). Built against the shapes fixed in
`docs/API_CONTRACTS.md` so it can be pointed at the real `server/` (B16 Console
API) with a one-line config change, whether or not that API existed yet at
build time.

## Run it

```bash
cd console
npm install
npm run dev       # http://localhost:5173
```

Log in with a demo account (mock mode, no real server needed):

| Username | Password | Role |
|---|---|---|
| `operator` | `operator` | operator |
| `reviewer` | `reviewer` | reviewer |
| `engineer` | `engineer` | engineer |
| `admin` | `admin` | admin |

## Pointing at the real server/

Copy `.env.example` to `.env.local` and set:

```
VITE_API_BASE_URL=http://localhost:8000
```

Every API call in `src/api/endpoints.ts` tries that base URL first and falls
back to seeded mock data (`src/api/mock.ts`) on any failure (network error,
404, wrong shape) — so partial server availability doesn't break the console,
it just quietly shows mock data for whatever endpoint isn't up yet. Each
screen displays a small tag (`live server data` / `mock data (server
unreachable)`) so it's never ambiguous which one you're looking at. Set
`VITE_USE_MOCKS=1` to force mock mode even with a reachable server (useful for
demos).

**`server/` did land partway through this build, so one real integration was
verified**: started it locally (`server/.venv/Scripts/python -m uvicorn
server.main:app --port 8001`), seeded an admin user with its own
`python -m server.seed <email> <password>` script, logged in, and confirmed
`GET /kb` and `GET /fleet/kiosks` respond (both returned `[]` — no data
seeded yet, but the routes, auth middleware, and JSON responses all work).

That said, **every console screen below is still wired to mock data at
runtime**, because the live server's actual shapes differ from what this
console assumes in two places discovered during that check, neither of which
`docs/API_CONTRACTS.md` pins down explicitly:
- Login is `POST /auth/login` with `{email, password}` → `{access_token,
  token_type, role}` (no `user.username`), not the `{username, password}` →
  `{token, user: {username, role}}` shape `src/api/types.ts` assumes.
- Route paths differ slightly from what this console guessed for endpoints
  `API_CONTRACTS.md` doesn't itself specify (e.g. `/kb` not `/kb/entries`,
  `/fleet/kiosks` matches but `KioskOut`'s fields — `current_model_version`,
  `last_heartbeat_at`, `uptime_s`, `online` — don't line up with this
  console's `KioskSummary` field names).

Because `endpoints.ts` catches any fetch failure and falls back to mocks,
this mismatch fails soft today (screens just show mock data with the "mock"
tag) rather than breaking — but real integration needs either this console's
`login`/`fetchFleet` calls adjusted to the server's actual field names, or
the server's routes adjusted to match `API_CONTRACTS.md`. Left as a follow-up
since resolving it means picking one side to change, which is a call for
whoever owns reconciling the two phases, not something to silently pick here.

## What's built

- **Auth** (`src/auth/`): login screen, JWT-shaped token in `localStorage`,
  role stored alongside it. `src/auth/permissions.ts` is the single
  permission matrix for all four roles (`operator | reviewer | engineer |
  admin`) — every nav link and route checks it via
  `src/components/RequirePermission.tsx`, so access control (FR-W9) is
  enforced consistently rather than screen-by-screen.
- **FR-W1 Fleet** (`src/pages/FleetView.tsx`) — kiosk table: health, model
  version, uptime, temperature (`null` renders as "unavailable", never a
  faked number, matching the kiosk's own behaviour on non-Jetson hardware),
  last contact.
- **FR-W2 Live session** (`src/pages/LiveSessionView.tsx`) — recognition +
  confidence stream for in-progress sessions. The camera-imagery control is a
  genuinely `disabled` button with an explanatory tooltip (elevated
  authorisation + audit, no video is ever retained per NFR-S2) rather than a
  toggle that does nothing.
- **FR-W3 Review workspace** (`src/pages/ReviewWorkspace.tsx`) — one
  recognition per screen, fully keyboard-driven: `C` confirm, `X` correct
  (opens a text field, `Enter` submits), `F` flag for linguist, auto-advance
  after every decision. Built for volume since reviewer time is this
  project's scarcest resource.
- **FR-W4 Dataset management** (`src/pages/DatasetManagement.tsx`) — coverage
  by sign (signer/clip counts vs. target, gaps highlighted), recorded
  sessions with a retire action.
- **FR-W5 Training visibility** (`src/pages/TrainingVisibility.tsx`) —
  training curves per run, diff against the champion, and every one of the 7
  DR-6 gate criteria rendered individually (not collapsed into a pass/fail
  badge).
- **FR-W6 Model registry** (`src/pages/ModelRegistry.tsx`) — every version
  with metrics/gate/status; promote, roll back, and pin-per-kiosk actions
  (gated to `engineer`/`admin` via the permission matrix).
- **FR-W7 Knowledge base** (`src/pages/KnowledgeBaseEditor.tsx`) — edit
  keywords/answers, with a signed-rendering preview that flags words missing
  from the clip library. Since `core/clip_library.json` currently has
  `clip_path: null` for every word, the UI treats that as the calm, expected
  state (a labelled placeholder per word, a short explanatory note) rather
  than an alarming wall of warnings.
- **FR-W8 Logs & alerts** (`src/pages/LogsAlerts.tsx`) — searchable
  structured-log table, alert list with severities and acknowledge action.
- **FR-W9 Access control** — see Auth above.

## Data layer

`src/api/client.ts` (fetch wrapper, base URL, auth header) +
`src/api/endpoints.ts` (one function per operation, mock-fallback baked in) +
`src/api/types.ts` (types mirroring `docs/API_CONTRACTS.md` exactly) +
`src/api/mock.ts` (seeded fixtures for every view, clearly separated from
real data). No component calls `fetch` directly — pointing the whole app at a
different server is changes to `client.ts`/`endpoints.ts` only.

## Tests

```bash
npm test        # vitest run — 17 tests across 11 files
npm run build   # tsc -b && vite build — verified clean
```

One smoke test per major view (renders against mock data, asserts the
requirement-specific behaviour: gate criteria all shown, camera control
disabled-with-tooltip, keyboard-driven review advance, etc.), plus a
permission-matrix unit test and an app-level routing smoke test.

`vite.config.ts` sets `pool: 'forks'` + `fileParallelism: false` for the test
runner — the default fully-parallel/fully-isolated config spawned enough
concurrent jsdom workers to OOM on this machine; this trades some test speed
(~30s instead of ~6s) for reliability. Worth revisiting on a machine with
more headroom.

## Deliberately deferred

- Real integration against `server/` (see above) — types are ready, only
  needs a base URL once `server/` exposes B16's routes.
- Camera-imagery viewing itself: no endpoint exists anywhere in the system
  yet (no video is retained), so there's nothing to wire even for `admin`;
  the control is present and explained, not implemented.
- Charting for training curves is a plain table for now — swapping in a
  chart library is cosmetic, not structural, and the data shape already
  supports it.
- Server-side enforcement of the same role matrix — this console only
  enforces FR-W9 in the UI layer; `server/`'s B16 must independently enforce
  it (never trust a client-side check alone), which is out of this phase's
  scope (`server/` belongs to a different agent).

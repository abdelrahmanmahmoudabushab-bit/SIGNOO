# Running SIGNOO on a DigitalOcean droplet

**Status: written and reviewed, not yet run.** No Docker daemon or droplet was available
where this was written, so none of `deploy/` has been built or started. What *was* run is
listed under "What was verified". Expect to fix a typo or two on the first
`docker compose up`.

## What goes on the droplet, and what does not

```
                 internet
                    │  443
   ┌────────────────▼────────────────── droplet ───────────────────────────┐
   │  web (Caddy)    HTTPS certificate, the console's static files,        │
   │       │         /api/*  ──►  api                                      │
   │  api  (FastAPI) B11–B16: ingest, review, training, registry, KB, fleet │
   │       │         + scheduler loop + ml/train.py subprocesses            │
   │  db   (Postgres)                                                      │
   │  volumes: pgdata · runs (checkpoints, frozen benchmark) · kbdata      │
   └───────────────────────────────────────────────────────────────────────┘
        ▲  https://<domain>/api            ▲  https://<domain>
        │  kiosk token (per kiosk)         │  console login (JWT)
   Jetson kiosks (B1–B10, offline-capable) You, in a browser
```

- **On the droplet:** everything in `server/`, and the console. This is the *learning loop*.
  It can be down for a day and no visitor notices: every kiosk keeps serving from its own
  models and its own knowledge base.
- **Not on the droplet:** the kiosk (`kiosk/`, `electron/`). It runs on the Jetson beside the
  camera. Putting recognition on a droplet would add network latency to every sign and make
  the kiosk useless when the network drops.
- **Not distributed yet:** the sign clip library. The kiosk plays back real recorded motion
  from its own disk, and nothing sends clips to kiosks. There are no recorded clips yet, so
  nothing is lost today; once there are, they need the same ETag sync that models and
  knowledge bases now have.

## Sizing

| Droplet | What it can do |
|---|---|
| 1–2 GB RAM | API + Postgres + console only. Set `SIGNOO_AUTO_TRAIN=0` and train somewhere else. Training next to Postgres on this size runs out of memory. |
| **4 GB RAM, 2 vCPU** (recommended start) | Everything, including the automatic 3-day training. CPU only: the landmark model (~1.3 M parameters) is small, but a run will be slower than on a desktop. |
| 8 GB+ | Several languages trained in turn without contention. |

A droplet has no GPU. If a model outgrows CPU training, run `ml/train.py` on a GPU machine
rather than resizing the droplet for it.

## First-time setup

1. **Create the droplet** (Ubuntu 24.04), add your SSH key, and point a domain's A record at
   its IP. Caddy cannot get a certificate until DNS resolves to the droplet.
2. **Provision it** (installs Docker, firewall, swap, automatic security updates). Copy the
   script over and read it before running it as root; it is short:
   ```bash
   scp deploy/provision-droplet.sh root@<ip>:
   ssh root@<ip> bash provision-droplet.sh
   ```
3. **Get the code and configure it:**
   ```bash
   git clone <your repo> /opt/signoo && cd /opt/signoo
   cp deploy/.env.example deploy/.env
   # fill in every value; generate each secret with:  openssl rand -hex 32
   # set SIGNOO_CODE_COMMIT to:  git rev-parse --short HEAD
   ```
4. **Start it:**
   ```bash
   docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
   docker compose -f deploy/docker-compose.yml logs -f api   # migrations run, then uvicorn starts
   curl https://<domain>/api/health                           # {"status":"ok"}
   ```
   The server **refuses to start** with a short or default `SIGNOO_JWT_SECRET` or a blank
   `SIGNOO_KIOSK_TOKEN`. If `api` keeps restarting, read its log first.
5. **Create the first admin** (there is deliberately no endpoint for it):
   ```bash
   docker compose -f deploy/docker-compose.yml exec api python -m server.seed you@example.com 'a-long-password'
   ```
   Then sign in at `https://<domain>` and create reviewer/engineer accounts from there.

## Connecting a kiosk

Each kiosk gets **its own credential**, derived from the server's master key. A token stolen
from one kiosk opens nothing belonging to another.

```bash
docker compose -f deploy/docker-compose.yml exec api python -m server.scripts.kiosk_token kiosk-lobby-01
```

On that kiosk set (see `kiosk/README.md`):

```
SIGNOO_SERVER_URL=https://<domain>/api
SIGNOO_KIOSK_ID=kiosk-lobby-01
SIGNOO_KIOSK_TOKEN=<the printed value>
```

The kiosk then registers, heartbeats every 30 s, uploads consented sessions, and pulls the
model and knowledge base the server assigns it. With `SIGNOO_SERVER_URL` unset it works as
before and sends nothing anywhere.

> **Read this before updating a kiosk that is already running.** From this change on, a
> language whose model was trained only on generated data is **refused to visitors** unless
> the kiosk has `SIGNOO_ALLOW_SYNTHETIC=1`. Today that is `liu`, so an existing Jetson
> service that is not given that variable will show `liu` as "coming soon" after the update.
> That is the intended rule (a model that recognises a generator is not recognition), but
> if you are demoing the synthetic model, set the variable on that kiosk. A deployment that
> serves the public should not set it.

## The path a model takes to a kiosk

1. Visitors who consent upload clips → the **review queue**. Nothing trains on an unreviewed clip.
2. Reviewers confirm or correct them in the console.
3. Once, deliberately: an engineer **freezes the benchmark** with `POST /training/benchmark`,
   naming the validation and test signers. They must be *enrolled* people from a
   data-collection session, not anonymous kiosk visits (a visit is one id per visit, so the
   same person returning would leak from training into the benchmark). It cannot be changed
   afterwards. **Until two enrolled signers have reviewed data, the automatic cycle cannot
   start, however many kiosk visits arrive.**
4. Every 3 days (`SIGNOO_AUTO_TRAIN_HOURS`) the server trains a **candidate** for each
   language in `SIGNOO_AUTO_TRAIN_LANGUAGES`. One run per language at a time.
5. An engineer runs `shadow-complete`, then `promote`. The gate cannot be bypassed.
6. `POST /fleet/rollouts` targets kiosks. They pick it up on their next heartbeat, verify its
   SHA-256, and swap. A rollout becomes `completed` once every target reports the version.
7. **Rollback** (`POST /registry/<version>/rollback`) stops serving that version; kiosks
   revert to the previous promoted model, or to the model they shipped with.

## Day-two operations

- **Update:** `git pull`, set `SIGNOO_CODE_COMMIT`, re-run the `up -d --build` command.
  Migrations run on start. Take a backup first.
- **Backups:** `deploy/backup.sh` dumps the database and the `runs` volume. Put it in cron and
  set `SIGNOO_BACKUP_REMOTE` to a DigitalOcean Spaces rclone remote. **A backup that lives on
  the droplet it protects will not help when the droplet is gone.** Restore into a scratch
  droplet once before you need it.
- **Logs:** `docker compose -f deploy/docker-compose.yml logs -f api`.
- **Retention:** unreviewed clips are deleted after `SIGNOO_RETENTION_DAYS` (default 30).
  `DELETE /review/sessions/<session_id>` (admin) erases one session now. Neither can remove
  data from a model that has already trained on it.

## What was verified, and what was not

**Verified by running it (2026-10-12):**

- The migration chain on a scratch SQLite database: upgrade, the copy of existing rollout
  targets into their new table (a duplicate target was dropped), downgrade (the JSON column
  came back with the targets), and upgrade again.
- Tests for: per-kiosk tokens, declined consent refused by the server, rollback withdrawing a
  model and falling back, rollout completion, the download hash, the per-language training
  lock (including taking over a dead holder's lock), the enrolled-signer benchmark, retention,
  erasure, per-language knowledge bases, their delivery to a kiosk, the kiosk reverting to its
  factory model, a corrupt download never replacing a working one, a synthetic-only language
  not being offered, and a second browser connection replacing the first.

**Not verified. Do these before trusting it:**

- `docker compose up` itself, the CPU-torch install (`torch==2.14.0` must exist on PyTorch's
  CPU index; if the image build fails there, that line is the cause), and Caddy's certificate.
- The migration on **Postgres**. It ran on SQLite. It uses Alembic's batch mode, which is
  correct on both, but a Postgres run is the real check.
- A real kiosk talking to a real droplet over the internet.
- Load: nothing here has been measured with more than a handful of simulated kiosks.

## Known limits

- **One droplet is one point of failure for the learning loop** (not for visitors). Backups
  and rebuilding from `deploy/` are the mitigation, not high availability.
- **No rate limiting** on login or the kiosk endpoints. Use DigitalOcean's cloud firewall or a
  managed load balancer in front, or add a rate-limit module to Caddy.
- **Landmarks are stored as JSON in Postgres.** Fine for thousands of clips. At hundreds of
  thousands, move them to object storage (DigitalOcean Spaces) and keep a pointer in the row.
- **`POST /training/trigger` still trains inside the HTTP request.** A second trigger for the
  same language is refused with 409, but one trigger holds its connection for the length of
  the run. Prefer the scheduler, or call it with a long timeout.
- **The console can show seeded mock data when the API is unreachable in a development build.**
  The production image forces mocks off (`VITE_USE_MOCKS=0`), but `console/src/api/endpoints.ts`
  still has the fallback path; check every screen discloses its source before an operator
  relies on it.
- **Bearer tokens, not mutual TLS.** Per-kiosk tokens stop one stolen kiosk exposing the
  fleet, but revoking a single kiosk means retiring its id or rotating the master key.

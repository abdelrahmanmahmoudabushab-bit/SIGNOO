# Shared contracts between kiosk, server, and console

Written once, up front, so `kiosk/`, `server/`, and `console/` can be built in parallel
without inventing incompatible shapes for the same thing. Companion to
`docs/SYSTEM_DESIGN.md §1.1` (which defines the in-process block contracts in
`core/contracts.py`) — this file defines the contracts that cross a process boundary.

## 1. Vocabulary (`core/vocab.json`)

One list, shared by kiosk B5 (assembly) and server B14 (knowledge-base editor, for the
"unrenderable word" warning) and console (dataset/KB views). Ported from the existing
`assets/vocab.js` starter list.

```json
[{"id": "marhaba", "ar": "مرحبا", "en": "hello"}, ...]
```

Amended 2026-10-11: there is one vocabulary file per language, all in this shape.
`core/vocab.json` is the `liu` vocabulary and `core/vocab_asl.json` is the `asl` one;
each language's entry in `core/languages.json` names its own (§7). "One list, shared"
above is true of the kiosk and server for `liu` only. The server and console read
`core/vocab.json` alone (`server/config.py`), so their warnings and views cover `liu`
and no other language.

## 2. Sign clip library manifest (`core/clip_library.json`)

What B8 (sign playback) looks up per word. Empty of real clips until
`docs/ROADMAP.md` Phase 6 — every entry exists with `clip_path: null`, which is the
normal state, not an error state (`Requirements.md` FR-W7).

```json
{"marhaba": {"clip_path": null, "recorded_by": null, "recorded_at": null}}
```

B8 and the console's KB editor both treat `clip_path: null` as "render the
placeholder / flag this word", never as a missing key to guard against.

Amended 2026-10-11: there is one manifest per language, all in this shape
(`core/clip_library.json` for `liu`, `core/clip_library_asl.json` for `asl`), named by
`clip_library` in the language's entry (§7). The kiosk looks a word up only in the
selected language's manifest. The server's `GET /kb/clip-library` returns the `liu`
manifest only.

## 3. Knowledge base entry (server B14, read by kiosk B7)

```json
{
  "id": "kb_001",
  "site_id": "default",
  "keywords": ["افتتاح", "hours", "opening"],
  "answer_ar": "نفتح من التاسعة صباحًا حتى الخامسة مساءً.",
  "answer_en": "We are open from 9am to 5pm.",
  "updated_at": "2026-09-04T00:00:00Z"
}
```

Retrieval (B7) is keyword/embedding match over `keywords` + both answer fields; no
match above a similarity floor is a **refusal** (`Answer.refused=True`), not a guess
(FR-A1).

Two corrections, 2026-10-11:

- Retrieval as built is keyword overlap over `keywords` only (`kiosk/dialogue.py`).
  There is no embedding match and the answer fields are not searched.
- A knowledge base is per language on the kiosk: each language's entry in
  `core/languages.json` names its own file (§7), a JSON array of entries in the shape
  above. The entry has no language field, and the server has none either: `/kb` stores
  one set of entries and writes them all to `kiosk/data/knowledge_base.json`, the `liu`
  file. A language whose file is missing or empty refuses every question.

## 4. Session upload (kiosk B10 -> server B11)

One POST per session, on consent. Landmarks only — never video (`NFR-S2`).

```json
{
  "session_id": "uuid",
  "kiosk_id": "kiosk-dev-01",
  "language": "liu",
  "consent_retention": true,
  "started_at": "...", "ended_at": "...",
  "items": [
    {
      "utterance_id": "uuid",
      "landmarks": [[...203-dim float array per frame...]],
      "predicted_gloss": "marhaba",
      "confidence": 0.82,
      "correction": null,
      "signer_id": "visitor-1a2b3c4d"
    }
  ]
}
```

`signer_id` is an anonymous per-visit id (`visitor-` plus 8 hex characters of the
session id). Visitors are never identified, so one visit stands in for one signer;
that is what lets the server keep a signer-independent train/validation/test split
(DR-4). A returning visitor becomes a new "signer". Rows with no `signer_id`
(`"unknown"`) are never exported for training.

The kiosk delivers each session as store-and-forward: set `SIGNOO_SERVER_URL` and a
queued file is deleted only after the server answers 2xx; blank means nothing leaves
the device.

`correction` is filled in when the visitor picked a different candidate from the
uncertainty UI (FR-K8/FR-K9) — a **weak label** (FR-L2), never treated as verified
ground truth until a Deaf reviewer confirms it in the console (FR-L3).

## 5. Roles (server B16, console auth)

`operator | reviewer | engineer | admin` — least privilege (FR-W9). Camera-imagery
endpoints (none exist yet, since no video is retained) are reserved for `admin` and
must be audited with a reason if ever added (`NFR-S4`).

## 6. Model registry entry (server B13)

```json
{
  "version": "2026-09-04T00:00:00Z-abcd123",
  "code_commit": "abcd123",
  "data_snapshot": "manifest-sha256:...",
  "metrics": {"top1": 0.0, "top5": 0.0, "per_signer": {}, "per_sign": {}},
  "gate": {"passed": false, "criteria": {"...": "pass|fail"}},
  "status": "candidate | shadow | promoted | rejected | rolled_back"
}
```

Produced by wrapping `ml/train.py` + `ml/evaluate.py` output — see `models/README.md`.

Added 2026-10-11, because the code returns more than the example above shows
(`ModelVersionOut` in `server/schemas.py`):

- `language`: the one language this version belongs to. Every version has exactly one
  (`SYSTEM_DESIGN.md` §1.1b). `GET /registry?language=<code>` filters on it.
- `checkpoint_path`: where the server keeps the checkpoint file, or `null`.
- `gate` also carries `details` (one human-readable sentence per criterion) and
  `measurements` (numbers the gate needs again later, such as
  `candidate_latency_ms`). Criterion values are the strings `"pass"` or `"fail"`.

## 7. Language entry (`core/languages.json`)

One JSON array, one object per card on the kiosk's language screen. Each object is one
language's whole engine (`SYSTEM_DESIGN.md` §1.1b). Read at kiosk start by
`LanguageRegistry` in `kiosk/languages.py`. The file path can be changed with
`SIGNOO_LANGUAGES_PATH`.

```json
{
  "code": "asl",
  "status": "pilot",
  "real_data": true,
  "model": "models/asl50.bars.pt",
  "vocab": "core/vocab_asl.json",
  "knowledge_base": "kiosk/data/knowledge_base_asl.json",
  "clip_library": "core/clip_library_asl.json",
  "confident_threshold": 0.80,
  "reject_threshold": 0.25,
  "segmentation": {"pause_ms": 500, "motion_threshold": 0.02, "hands_absent_ms": 800,
                   "min_frames": 3, "min_ms": 150, "max_ms": 4500, "max_frames": 150}
}
```

The `segmentation` object in the example shows the kiosk defaults; the real `asl` entry
sets none of them.

These are all the keys the loader reads. Any other key is ignored.

| Key | Required | Meaning |
|---|---|---|
| `code` | yes | The language code used everywhere else: the `language` websocket message, session uploads, the registry, the model endpoint |
| `status` | yes | `live`, `pilot` or `coming_soon`. Passed through to the frontend for the card's badge. It does not decide whether the card can be pressed; `ready` does (§8). A language with `real_data: false` must be `coming_soon` (checked by `kiosk/tests/test_live_feedback.py`) |
| `real_data` | no, default `false` | Whether the model was trained on real human signing. **Enforced since 2026-10-12:** a language where this is `false` is not offered to a visitor unless the kiosk runs with `SIGNOO_ALLOW_SYNTHETIC=1` (development and tests only) |
| `model` | no | Checkpoint path, relative to the repository root. `null` or absent means no model |
| `vocab` | no | Vocabulary file (§1 shape), relative to the repository root |
| `knowledge_base` | no | Knowledge-base file (a JSON array of §3 entries) for this language only |
| `clip_library` | no | Clip-library manifest (§2 shape) for this language only |
| `reject_threshold` | no | Below this probability the recogniser returns no hypothesis. Default: the kiosk's `SIGNOO_REJECT_THRESHOLD` (0.55) |
| `confident_threshold` | no | Below this probability the visitor is offered candidates instead of a committed word. Default: the kiosk's `SIGNOO_CONFIDENT_THRESHOLD` (0.80) |
| `segmentation` | no | Object with any of `pause_ms`, `motion_threshold`, `hands_absent_ms`, `min_frames`, `min_ms`, `max_ms`, `max_frames`. A key left out takes the kiosk default. Any other key inside it raises an error when a connection is built for that language. A word must have at least `min_frames` frames spanning at least `min_ms`, and is closed as it stands after `max_ms` (or `max_frames`, the memory bound) |

Loading rules, as the code has them:

- A language is **configured** only when both `model` and `vocab` are set. Without
  both, nothing is loaded for it, including its knowledge base and clip library.
- A configured language is **ready** when its vocabulary and its checkpoint both load.
  If either fails, the language is not ready and the reason is logged; the kiosk still
  starts.
- A ready language is **selectable**, meaning a visitor may be put on it, only when
  `real_data` is `true` or the kiosk was started with `SIGNOO_ALLOW_SYNTHETIC=1`. Both the
  `language` message and the `consent` message refuse a language that is not selectable.
- The knowledge base and the clip library are loaded only for a ready language. If one
  is absent or fails to load, the language stays ready and goes without that part.
- `SIGNOO_MODEL_PATH`, `SIGNOO_VOCAB_PATH`, `SIGNOO_KB_PATH` and
  `SIGNOO_CLIP_LIBRARY_PATH` replace the paths of the default language (`liu`) only,
  and only when `liu` is configured in the file.
- A model the server rolled out earlier (§10) is preferred over `model` at start. If it
  does not load, the kiosk uses `model`.

## 8. Kiosk language surfaces (kiosk process, port 8000)

### 8.1 `GET /languages`

No authentication; it is served to the kiosk's own frontend. Returns a JSON array with
one object per entry in `core/languages.json`, in file order
(`LanguagePack.summary`):

```json
[
  {
    "code": "asl",
    "status": "pilot",
    "ready": true,
    "model_loaded": true,
    "real_data": true,
    "labels": 50,
    "model_version": null,
    "vocab_url": "/core/vocab_asl.json",
    "has_knowledge_base": true,
    "has_clip_library": true
  }
]
```

| Field | Meaning |
|---|---|
| `code`, `status`, `real_data` | Copied from the entry (§7) |
| `ready` | `true` when this language may be offered to a visitor: its model and vocabulary are loaded **and** it is selectable (§7). The frontend enables a card only when this is `true`; every card starts disabled in `index.html` |
| `model_loaded` | `true` when the model and vocabulary loaded, whether or not the language may be offered. A language can be `model_loaded` and not `ready`: `liu` on a production kiosk today |
| `labels` | Number of labels in the loaded model, `0` when no model is loaded. The count includes the `__other__` label if the model has one |
| `model_version` | The version the server rolled out to this kiosk for this language (§10), or `null` when the kiosk is running the checkpoint it shipped with. Added on 2026-10-11 |
| `vocab_url` | Where the frontend fetches this language's vocabulary, or `null` when the language is not ready or its vocabulary file is outside the served directory |
| `has_knowledge_base` | `true` when this language's own knowledge-base file loaded. An empty file still counts as `true`; such a language refuses every question |
| `has_clip_library` | `true` when this language's own clip library loaded. It says nothing about whether any clip is recorded |

The numbers in the example are illustrative.

### 8.2 `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "labels": 20,
  "languages": [ ...the same array as GET /languages... ],
  "telemetry": {
    "temperature_c": 52.5, "throttled": "unavailable", "memory_mb": 912.4,
    "tracker_fps": 10.2, "frame_ms": 3.1, "classify_ms": 14.0,
    "outcomes": {"committed": 12, "candidates": 5, "rejected": 2, "hands_lost": 1}
  },
  "warm_up_s": {"liu": 0.026, "asl": 0.008}
}
```

`telemetry` (the numbers above are illustrative): `temperature_c` and `memory_mb` are
real where the platform reports them and the string `"unavailable"` where it does not
(Windows); `throttled` is always `"unavailable"`, nothing reads it yet. `tracker_fps` is
tracked frames a second over the last five seconds and `null` when no frame has arrived
in that time, which means "nobody at the camera", not "stalled". `frame_ms` is the average
cost of taking one frame in, `classify_ms` the last model pass, and `outcomes` counts
recognitions since the process started. None of it identifies a visitor or a word.

`warm_up_s` is the seconds each loaded model took for its start-up test pass. `null` for
a language means the pass failed and that model may not run at all.

`status`, `model_loaded` and `labels` describe the default language (`liu`) only:
`status` is `"ok"` when `liu` is ready and `"degraded"` otherwise, even if other
languages are ready. Use the `languages` array for the per-language picture. This
is a leftover from the single-language kiosk and does not match
`SYSTEM_DESIGN.md` §1.1b.

### 8.3 Selecting a language on `/ws/session`

The full websocket protocol is the docstring at the top of `kiosk/main.py`. The parts
that concern language:

```json
{"type": "language", "lang": "asl"}
```

If the language is ready, the connection moves onto that language's whole engine
(model, vocabulary, confidence bars, segmentation, knowledge base, clip library) and
the server replies with a `state` message whose `session_state` is `"language"`.
Session uploads from that session carry this code in `language` (§4).

If the code is unknown or the language is not ready, nothing changes and the server
replies:

```json
{"type": "error", "code": "language_unavailable", "message": "'esl' has no trained model yet"}
```

The kiosk never substitutes another language. A client must treat
`language_unavailable` as final for that card and show it as not available.

Other error codes on this socket: `model_unavailable` (a word boundary was detected
but the connection's language has no model loaded), `perception_error`, `bad_json`,
`unknown_message`, `unknown_control`.

Two behaviours to know about. A connection starts on the default language's engine
(`liu`) and returns to it when a session ends. A client that never sends `language` is
therefore on `liu`, and its `consent` is refused with `language_unavailable` unless `liu`
is selectable (§7), so it cannot be recognised by a model that may not be offered. And the
frontend falls back to `/core/vocab.json` for its labels when a selected card has no
`vocab_url`.

Pressing a card also starts the tracker in the background, so it is ready by the time
consent is given. It receives no frame before consent.

### 8.4 What the visitor is told while signing

Sent by the kiosk during an active session; the frontend shows each one (FR-K5, NFR-X1).
`framing` and `signing` are sent when their content changes, not with every frame.

```json
{"type": "framing", "ok": false, "reason": "too_dark"}
{"type": "signing", "state": "signing"}
{"type": "hypothesis", "status": "rejected", "reason": "hands_lost",
 "gloss": null, "confidence": 0.0, "alternatives": []}
```

| Message | Values | Meaning |
|---|---|---|
| `framing.reason` | `too_dark`, `too_bright`, `no_body`, `perception_unavailable`, `null` | What is wrong with the picture. Lighting is checked first, since a dark or washed-out frame is usually also why no body is found. `ok` is `true` only when `reason` is `null` |
| `signing.state` | `watching`, `signing`, `recognising` | Nothing being signed; hands up and a word being collected; the word is closed and the model is running |
| `hypothesis.reason` | `not_recognised`, `hands_lost` | Present on a rejection only. `hands_lost`: tracking lost both hands for more than `SIGNOO_MAX_HANDS_LOST_SHARE` (0.5) of the word's frames, and nothing was classified |

A client must ignore a `reason` or `state` it does not know rather than show it blank.

## 9. Kiosk authentication (`X-Kiosk-Token`)

Kiosk-to-server calls carry one shared secret in an HTTP header:

```
X-Kiosk-Token: <value of SIGNOO_KIOSK_TOKEN>
```

Checked by `require_kiosk_token` in `server/auth.py` on:

| Call | Purpose |
|---|---|
| `POST /ingest/sessions` | Session upload (§4) |
| `POST /fleet/kiosks/register` | Body `{"kiosk_id", "site_id" (default "default"), "language" (default "liu")}`. Returns 201 with the kiosk record, also when the kiosk was already registered |
| `POST /fleet/kiosks/{kiosk_id}/heartbeat` | Body: any of `uptime_s`, `temperature_c`, `throttled`, `current_model_version`, `model_versions` (§10). An omitted field keeps its last value. 404 if the kiosk is not registered |

Rules:

- A missing or wrong token gets `401` with `{"detail": "invalid kiosk token"}`. The
  comparison is constant-time.
- If the server's `SIGNOO_KIOSK_TOKEN` is blank, the check is off. A server refuses to
  start with a blank token unless `SIGNOO_ENV=dev`.
- The kiosk sends the header only when its own `SIGNOO_KIOSK_TOKEN` is set, and makes
  these calls only when `SIGNOO_SERVER_URL` is set.
- In the default mode (`SIGNOO_KIOSK_TOKEN_MODE=shared`) the token is one secret for the
  whole fleet. It proves "this is one of our kiosks", not which one; the `kiosk_id` in
  the path or body is not verified against it.
- In `per_kiosk` mode (2026-10-12) each kiosk presents its own token,
  HMAC-SHA256 of its `kiosk_id` under the server's `SIGNOO_KIOSK_TOKEN`, and a call for
  one kiosk made with another kiosk's token gets `401`. Issue a kiosk's token on the
  server with `python -m server.scripts.kiosk_token <kiosk_id>`.
- `SYSTEM_DESIGN.md` specifies mTLS for this link. mTLS is not implemented, and this
  header is what exists in its place.
- Console calls do not use this header. They use a bearer JWT and roles (§5).

## 10. Model and knowledge-base distribution

**Status on 2026-10-13:** built on both sides and committed (`kiosk/model_sync.py`,
`kiosk/kb_sync.py`, `get_model` and `get_knowledge_base` in `server/routers/fleet.py`),
with tests on each side. This section was re-read against that code on this date. It has
not run against a real fleet.

A kiosk pulls its models. One request asks about one language, because each language's
model is a separate engine (`SYSTEM_DESIGN.md` §1.1b).

```
GET /fleet/kiosks/{kiosk_id}/model?language=<code>
X-Kiosk-Token: <token>
If-None-Match: "<version the kiosk has installed for this language>"    (optional)
```

**Desired version.** The model of the most recent served rollout (`active` or
`completed`) that targets this kiosk, whose model's language matches `<code>` and whose
model is still `promoted`. A rollout of a model for another language is never an answer
to this request, and a rolled-back version stops being an answer at once.

| Status | When | Body and headers |
|---|---|---|
| `404` | The kiosk is not registered | |
| `204` | Nothing is assigned to this kiosk for this language | No body |
| `304` | `If-None-Match: "<version>"` equals the desired version | No body |
| `200` | A version is assigned and the kiosk does not have it | Body: the checkpoint file's bytes. Headers: `X-Model-Version: <version>`, `X-Model-Sha256: <hex sha256 of the body>`, `ETag: "<version>"` |
| `409` | A version is assigned but the server has no checkpoint file for it | |

`401` applies as for every `X-Kiosk-Token` call (§9).

Also true of the code: the `304` carries the `ETag` header; `409` covers both "no
checkpoint path recorded" and "the recorded file is missing"; and
`POST /fleet/rollouts` refuses a version that is not `promoted` with `409`, so only a
gated model can become a desired version.

What the kiosk does with a `200` is its own business, but the contract depends on it
never installing bytes whose sha256 differs from `X-Model-Sha256`, and on a failed
install leaving that language's running model in place. A `204` for a language the kiosk
had been given a model for means that model was withdrawn (a rollback): the kiosk goes
back to the checkpoint it shipped with.

**Knowledge base.** The same pattern, per language:

```
GET /fleet/kiosks/{kiosk_id}/kb?language=<code>
X-Kiosk-Token: <token>
If-None-Match: "<etag of the copy the kiosk has>"    (optional)
```

`200` returns a JSON array of §3 entries for that language and the kiosk's site (plus the
`default` site), with an `ETag`; `304` when the kiosk's copy is current; `204` when the
language has no entries, which tells the kiosk to keep what it has; `404` when the kiosk
is not registered. The kiosk writes what it receives beside its synced models, never over
the seed file in the repository, and keeps its working copy if a download does not parse.

**Heartbeat.** A heartbeat (§9) may carry the versions the kiosk has installed, one per
language:

```json
{"uptime_s": 1234.5, "model_versions": {"asl": "2026-10-11T09:00:00Z-abcd123"}}
```

A language running the checkpoint the kiosk shipped with has no entry. The older
single-valued `current_model_version` field still exists beside it and predates
per-language engines.

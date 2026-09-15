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

## 2. Sign clip library manifest (`core/clip_library.json`)

What B8 (sign playback) looks up per word. Empty of real clips until
`docs/ROADMAP.md` Phase 6 — every entry exists with `clip_path: null`, which is the
normal state, not an error state (`Requirements.md` FR-W7).

```json
{"marhaba": {"clip_path": null, "recorded_by": null, "recorded_at": null}}
```

B8 and the console's KB editor both treat `clip_path: null` as "render the
placeholder / flag this word", never as a missing key to guard against.

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
      "correction": null
    }
  ]
}
```

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

# SIGNOO — System Design

**Version** 0.1 (draft for approval) · Companion to `REQUIREMENTS.md`

---

## 1. Shape of the system

Three tiers, deliberately separated by what fails when the network does.

```
┌─ KIOSK (Jetson Orin Nano Super) ──────────────────────────────┐
│  B1 Capture → B2 Perception → B3 Segmentation → B4 Recognition│
│                                        ↓                       │
│  B7 Dialogue ← B6 Output ← B5 Assembly                        │
│       ↓                                                        │
│  B8 Sign Playback      B9 Session/Telemetry   B10 Data Capture│
│                                                                │
│  Fully functional offline: B1–B6, B8                          │
└──────────────────────┬─────────────────────────────────────────┘
                       │ mTLS, store-and-forward
┌──────────────────────▼─────────────────────────────────────────┐
│  SERVER                                                        │
│  B11 Ingest/Review · B12 Training · B13 Registry+Gate          │
│  B14 Knowledge Base · B15 Fleet Control · B16 Console API      │
└──────────────────────┬─────────────────────────────────────────┘
┌──────────────────────▼─────────────────────────────────────────┐
│  CONSOLE (web) — fleet, review, datasets, training, models     │
└────────────────────────────────────────────────────────────────┘
```

**The load-bearing decision:** the kiosk owns the entire visitor experience and never
depends on the server to serve a visitor. The server owns learning. A severed link
degrades *improvement*, never *service*.

### 1.1a The five blocks (plain-language view)

The sixteen B-numbers below are the engineering-level breakdown; day to day, the system
is five blocks. This grouping doesn't rename or renumber anything — every B-number
below is still the unit of code, tests, and the comments that cite it.

| # | Block | Does | B-numbers |
|---|---|---|---|
| 1 | **Presence & Capture** | notices a visitor, turns on the camera, produces landmarks | B1, B2, + presence/attract detection |
| 2 | **Understanding** | turns the landmark stream into recognised words | B3, B4, B5 |
| 3 | **Response** | speaks the result, answers questions, signs the answer back | B6, B7, B8 |
| 4 | **Kiosk Experience** | the on-screen session itself; local data capture with consent | B9, B10, + UI flow |
| 5 | **Learning Loop** | server + console: review, retrain, gate, roll out, admin | B11–B16 |

Blocks 1–4 are entirely on-device and offline. Block 5 is the only one that needs the
network — severing it degrades *improvement*, never *service* (§1's load-bearing
decision, restated).

### 1.1 Contracts between blocks

Every block takes and returns plain data — no camera handles, no model objects, no
database sessions crossing a boundary. This is what makes **NFR-N1** possible: a
recorded session replays through B2–B6 off-device and reproduces the output exactly.
When a kiosk misbehaves in a mall three cities away, that replay is the only debugging
tool that works.

```python
Frame        = {ts_ms: int, image: ndarray}                    # B1 → B2
Landmarks    = {ts_ms: int, pose: ndarray|None,
                left: ndarray|None, right: ndarray|None}       # B2 → B3
Utterance    = {clips: list[Clip], t_start: int, t_end: int}   # B3 → B4
Hypothesis   = {gloss: str, confidence: float,
                alternatives: list[tuple[str, float]]}          # B4 → B5
Rendered     = {gloss_seq: list[str], ar: str, en: str,
                uncertain: list[int]}                           # B5 → B6/B7
```

---

## 2. Kiosk blocks

### B1 — Capture

**Does:** owns the camera, produces timestamped frames.

60 fps capture, but **not** 60 inferences per second — past ~30 fps the extra frames
carry almost no additional linguistic information, while costing the entire compute
budget. The high frame rate buys reduced motion blur and a responsive preview, and
lets downstream stages pick the *freshest* frame instead of waiting.

**Single-slot buffer, newest wins, older frames dropped.** A queue converts a
throughput problem into permanently growing latency, and the visitor ends up watching
the system respond to what they signed five seconds ago. Dropping frames is the
correct behaviour; queueing them is not.

Threads: capture, inference, and UI are separate, communicating through single-slot
buffers.

### B2 — Perception

**Does:** frames → body and hand landmarks.

MediaPipe Holistic (upper-body pose + 21 landmarks per hand), converted to TensorRT.
Runs on every second frame → 30 fps. This is the largest item in the latency budget
(~20 ms) and the first place to optimise.

Emits `None` for absent hands rather than zeros, so downstream can distinguish "hand
not visible" from "hand at origin" — a distinction that silently corrupts features if
collapsed.

**Deliberate limitation:** hands and upper body only. Facial expression carries grammar
in sign languages, and this stack cannot see it (Requirements §7.1). Face landmarks are
a planned Phase 4 addition, not an oversight.

### B3 — Segmentation

**Does:** decides where an utterance starts and ends. **The block most likely to be
underestimated.**

Isolated recognition assumes someone hands you a clip containing exactly one sign.
Reality is a continuous stream with signs running into each other (co-articulation),
hands resting, and people scratching their nose mid-sentence.

v1 uses **pause-based segmentation** with a state machine:

```
IDLE ──hands enter frame──> ACTIVE ──motion below threshold for N ms──> BOUNDARY
  ↑                            │                                            │
  └────hands leave frame───────┴────────────────────────────────────────────┘
```

Thresholds are per-language and tuned on real recordings, never guessed — signing
tempo differs between languages and between people.

**Honest about the limit:** this reads *deliberate, paused signing* well and fluent
connected signing poorly. Fluent signers do not pause between words.

**Fluent recognition is the product requirement, not an optional nice-to-have** — but
it cannot be built before there is continuous-signing data to train it on, and none
exists yet for LIU or ESL (Requirements §5.4). v1 therefore asks users to sign
word-by-word as an interim state, disclosed rather than hidden, while continuous data
collection runs alongside it. True continuous recognition (CTC over the landmark
stream, Roadmap Phase 4.5) replaces this the moment enough continuous clips exist —
the gap is a data problem with a named phase attached to close it, not a permanent
ceiling.

### B4 — Recognition

**Does:** utterance clip → gloss hypothesis with confidence.

Transformer encoder over landmark sequences (see `docs/TRAINING.md`). One **shared
backbone** with a **per-language head**, selected by the visitor's language choice.

Rationale: sign languages differ in vocabulary but share the underlying structure of
human signing — handshape, movement, spatial relationships. A shared backbone lets
ASL's larger public data improve encoders that start with little or no public data of
their own. Heads stay separate because the vocabularies are genuinely different
languages, not dialects.

**Language packs (2026-10 revision).** The visitor-facing language menu now lists nine
cards — Jordan (LIU), Saudi, UAE, Omani, Qatari, French (LSF), American (ASL), British
(BSL), International Sign — each its own head, dataset, and signer-recruitment effort,
not a caption-language toggle. Only a language with a real (non-synthetic) dataset and
a trained head is enabled; every other card opens to an explicit "recording in
progress" state rather than running an untrained or synthetic-only head live. Which
language is recruited next, after LIU, is an open decision (`docs/ROADMAP.md`) — the
architecture does not need that decision made to support the menu.

Rejection is a first-class output: anything beyond the rejection threshold returns *no
hypothesis*, and the UI asks the visitor to sign again. A confident mistranslation is
worse than an admitted failure (Requirements §6.3).

### B5 — Assembly

**Does:** gloss sequence → Arabic and English text.

Two stages, kept separate because they fail differently:

1. **Gloss sequence** — what was actually recognised, in sign-language order. Shown to
   the visitor as-is so they can see whether the machine understood them.
2. **Rendered text** — Arabic and English sentences.

For v1, rendering is **template and dictionary based**, not a learned translation
model: sign-language grammar differs from Arabic, but with a bounded vocabulary and
short utterances, templates are predictable and debuggable, and they never hallucinate.
A learned gloss→text model is Phase 5, once there is enough paired data to train and
*evaluate* one.

Low-confidence words are marked in the output structure so B6 can render them visibly
uncertain rather than presenting a guess as fact.

### B6 — Output

**Does:** shows text, speaks it.

Arabic and English shown **together**, not behind a toggle — a companion or staff
member may read one and not the other, and the pairing is the point of the product.

Text renders the instant it is ready. **Speech is strictly outside the latency
budget** and never blocks display: the visitor cannot hear it, and making them wait for
audio they will not perceive is the kind of design error that comes from a hearing
default. Piper for offline Arabic and English TTS.

### B7 — Dialogue

**Does:** utterance → grounded answer.

Retrieval over a curated per-site knowledge base, then a small language model to phrase
the answer. **Retrieval-grounded, not open-ended**: out-of-scope questions get a
visible refusal and a route to a human (FR-A5), because a plausible invented answer is
the worst possible output for someone who cannot easily verify it.

The answer is constrained at generation time to short, concrete sentences using
vocabulary present in the sign clip library, so B8 can render it completely. Words
outside the library are flagged and fingerspelled.

**Sizing, and the tightest constraint in the system:** the Jetson's 8 GB is shared
between CPU and GPU with no separate VRAM. A 7B model at 4-bit spends ~4.5 GB plus KV
cache — effectively the entire remaining budget — and the failure mode is swapping,
which on this board means multi-second stalls rather than graceful slowdown.

**v1 is cloud-primary with automatic local fallback.** The kiosk prefers a larger
cloud-hosted model whenever the network is up and the site's data policy allows
sending the utterance off-device, giving noticeably better answers than the 3B class
can produce alone. The moment the network is slow or absent, it drops to the local
3B-class quantised model (~2.5 GB resident) with a visible notice — never a spinner,
never a failed request the visitor has to retry. This is why B7 sits fully inside the
offline guarantee in §1 despite preferring the network: the fallback is not a degraded
mode bolted on afterward, it is the always-available floor the cloud path sits above.

### B8 — Sign playback

**Does:** answer text → a signing character on screen.

**Still not a generative avatar — the safety reasoning that ruled that out stands.**
Generating grammatical sign language from text is an open research problem; a system
that generates plausible-looking but wrong signing is actively harmful to a Deaf
visitor relying on it for correct information, with no way to detect the error. That
conclusion doesn't change with this revision.

**What changes (2026-10 revision): retargeted motion instead of flat video clips.**
Each word's library entry is no longer a video file — it is the same landmark sequence
`kiosk/data_capture.py` (B10) already records from a real signer, retargeted onto a
single persistent 3D rigged character (face, body, two hands) instead of played back
as that signer's own video. Every individual sign is still a real recording, played
back faithfully — correct by construction, same guarantee as the clip-library design —
but the visitor sees one consistent character regardless of which real signer
contributed which word, and sequences blend between words instead of cutting between
clips. This reuses B10's existing capture pipeline and `core/features.py`'s landmark
representation; the new work is the retargeting (landmarks → rig) and the renderer, not
a new capture or recognition system.

Cost, stated plainly: every word in every answer must still be recorded from a real
signer, per language — recording a landmark sequence is cheaper than filming a clean
video take, but the coverage problem (every word needed) is unchanged. The console
still warns when knowledge-base text contains words with no recorded sequence (FR-W7).

Playback highlights each word as it plays, tying the signed and written answer
together, same as before.

**Open engineering question, not yet resolved:** the retargeting step (mapping a real
signer's hand/body landmarks onto a rig that was not their own body) is nontrivial —
proportions differ per signer, and getting it wrong reintroduces exactly the
"plausible but wrong" risk this design otherwise avoids. Treat the first retargeting
attempt as a spike with an explicit correctness check (a Deaf reviewer confirms the
retargeted playback still reads as the right sign) before it is trusted for any word
shown live.

### B9 — Session and telemetry

**Does:** session state machine, structured logs, metrics.

Every state is visible on screen (`deaf-kiosk-ux` skill) and every transition is
logged with a correlation ID shared with the server. Telemetry covers latency per
stage, frame rate, temperature, throttle state, memory, and interaction outcomes.

Temperature and throttling are logged specifically because a kiosk slowing down in
August is otherwise indistinguishable from a software regression.

**Visitor-facing flow (2026-10 revision).** The state machine gains two states and the
`active` state gains an internal split; this is a UI/flow change on top of the
existing blocks, not a new block:

```
attract (presence auto-detected, no tap-to-start)
   → hi (greeting)
   → language (card grid, one card per language pack — §B4 "Language packs";
      unready packs open to "recording in progress", not a live session)
   → consent
   → active:translate (self-view camera box; two live caption boxes — the visitor's
      chosen language and English — fed by the existing B1→B6 pipeline, unchanged)
   → [visitor presses "Next"] → active:ask (B7 question → B8's signing character +
      a synced text box, replacing the old single "answer" screen)
   → thank_you
   → attract (reset, NFR-S3 erasure unchanged)
```

**Presence detection (`attract`) is new engineering, not yet built.** Today the kiosk
only turns the camera on when a visitor explicitly starts a session (`getUserMedia` on
button press); nothing watches the idle screen for an approaching visitor. Candidate
approaches: a low-fps face/person detector on the same camera feed used for B2 (cheapest,
no new hardware, but means the camera is always live, which needs its own
privacy-review sentence since today's design assumes the camera is off until consent);
or a dedicated presence sensor (PIR/ultrasonic/radar) gating when the camera turns on at
all (more hardware, cleaner privacy story — camera stays off until someone is actually
there). Pick one in Phase 2 alongside the Jetson bring-up; it is a camera/sensor
decision, not a model decision, so it does not block on any language's dataset.

### B10 — Data capture

**Does:** with opt-in consent, records what the learning loop needs.

**Landmarks by default, not video.** They carry the linguistic content, are what
training actually consumes, and do not carry the visitor's face. Video retention is a
separate, time-boxed, separately-justified opt-in.

Session data is erased on return to idle. Someone who walks away has consented to
nothing.

---

## 3. Server blocks

### B11 — Ingest and review

Store-and-forward from kiosks; everything lands in a **review queue**, never in
training. The queue is prioritised by expected value: low-confidence cases, corrected
cases, and under-represented signs first, because reviewer time is the scarcest
resource in the whole system.

Deaf reviewers work in a keyboard-driven interface built for volume — play, confirm,
correct, next.

### B12 — Training

Candidate training on validated data **every 3 days**, off-device on a GPU; kiosks
never train. The cadence is fixed and fully automatic — what is not automatic is
whether a candidate ships, which is entirely B13's job. A fast retraining cadence
without an equally firm gate is how a system quietly teaches itself its own mistakes
(see B13).

### B13 — Registry and promotion gate

**The safety-critical block.** Every model version is stored with the data snapshot and
code commit that produced it (NFR-N2).

The gate enforces the seven criteria in Requirements §5.3 against a **frozen,
human-verified benchmark** that the training process cannot modify, extend, or select
into.

> This is the mechanism that makes "the system retrains itself" safe. A model retrained
> on its own confident predictions drifts toward its own biases and grows *more
> confident as it grows more wrong* — no self-generated metric detects this, because
> the metric drifts with the model. **The training is automatic; the judgement is not.**

Passing candidates shadow-run against live traffic before promotion. Rollback is one
action, and automatic on live-metric degradation.

### B14 — Knowledge base

Per-site content for B7, edited in the console with preview of the signed rendering and
warnings for unrenderable words.

### B15 — Fleet control

Staged rollout of software and models, health monitoring, remote diagnostics,
automatic rollback. Never all kiosks at once.

### B16 — Console API

Role-scoped API behind the web console. Camera imagery requires elevated authorisation
and is audited with a reason (NFR-S4).

---

## 4. Budgets

### 4.1 Latency — end of sign to word on screen

```
capture + colour convert      ~5 ms
landmark extraction          ~20 ms   ← largest; optimise here first
segmentation decision         ~1 ms
feature assembly              ~2 ms
classification                ~5 ms
UI render                    ~16 ms   (one frame at 60 Hz)
                            ───────
                             ~49 ms  → NFR-P1 budget 300 ms
```

Wide headroom, deliberately: these are bench figures, and the real device runs hot,
shares memory with a language model, and drives a display.

**Outside this budget:** speech synthesis, answer generation, sign playback. They
happen after text appears.

### 4.2 Memory — the binding constraint

```
OS + display + application        ~1.5 GB
perception                        ~0.6 GB
recognition (one active language) ~0.2 GB
answering model (3B quantised)    ~2.5 GB
speech synthesis                  ~0.2 GB
buffers + clip library + slack    ~1.0 GB
                                 ────────
                                  ~6.0 GB  of 8 GB shared
```

Models load once at startup. Memory growth is monitored, not assumed absent — a kiosk
runs for weeks, and leaks surface on day nine, not in a demo.

---

## 5. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Kiosk runtime | Python 3.11 | Requirement C-6; hot paths go through TensorRT rather than a rewrite |
| Perception | MediaPipe Holistic → TensorRT | Mature, runs on-device, no training needed |
| Recognition | PyTorch → ONNX → TensorRT | Train off-device, deploy converted; engines are build artefacts, never committed as source |
| Kiosk UI | Local web app in kiosk-mode browser | Reuses existing SIGNOO UI work; touch and RTL are solved problems in a browser |
| Services | FastAPI | Typed, async, minimal |
| Console | React + TypeScript | Long-lived operational UI |
| Datastore | PostgreSQL + object storage | Boring and correct |
| Fleet transport | mTLS, store-and-forward | Kiosks must tolerate a bad link without losing data |

---

## 6. Failure modes

Designed for, not discovered later.

| Failure | Response |
|---|---|
| Network down | Recognition, translation, voice, playback continue. Answering falls back to local model with a visible notice. Captured data queues locally |
| Answering service down | Same fallback path |
| Camera disconnects | Visible out-of-service screen; alert to console; automatic recovery on reconnect |
| Perception too slow (thermal) | Drop to lower frame rate, keep serving, alert. Never queue frames |
| Memory pressure | Unload the answering model first — recognition is the core service, answering is the enhancement |
| Model file corrupt | Refuse to start with that model; fall back to the previously known-good version |
| Promotion gate rejects candidate | Champion stays; reason recorded; alert if rejections repeat, since that means the review pipeline is producing bad data |
| Bad model promoted anyway | Automatic rollback on live-metric degradation; one-action manual rollback |
| Visitor walks away mid-session | Timeout with visible countdown; session erased; return to clean idle |
| Reviewer queue backs up | Alert; training simply uses less new data. Never auto-approves to clear a backlog |

---

## 7. Repository layout

```
signoo/
├── kiosk/          B1–B10   runtime, UI, state machine
├── core/                    shared: features, contracts, protocol types
├── ml/             B12      training, evaluation, promotion gate
├── server/         B11–B16  ingest, review, registry, fleet, API
├── console/                 React operator console
├── data/                    datasets (gitignored)
├── docs/                    requirements, design, protocols
└── .claude/skills/          engineering, jetson, SLR, kiosk-UX standards
```

`core/` exists so the kiosk and the training pipeline compute features with **the same
code**. Divergence between training-time and inference-time features is a classic,
silent, and extremely expensive bug in this class of system.

---

## 8. Open design questions

Carried into the question round rather than silently decided:

1. ~~Answering model~~ — **resolved:** cloud-primary, automatic local fallback.
2. **Clip library production** — who records the sign playback library, per language,
   and how large is the initial word list?
3. **Reviewer capacity — now the sharpest open question.** Fluent recognition (Q4,
   resolved as the target) and originally-recorded LIU/ESL data (§5.4) both need
   human validation, and none is currently staffed. Zero reviewer hours is workable
   only if it is temporary: the near-term action is a data-sharing partnership with
   an institution that already employs Deaf staff (Holy Land Institute for the Deaf;
   Zayed Higher Organization) rather than building a reviewer team from nothing.
4. ~~Fluent vs word-by-word~~ — **resolved:** fluent is the target. Sequenced as
   Roadmap Phase 4.5, gated on continuous-signing data volume, not on a date.
5. **Deployment sites** — one site or several, and do they share a knowledge base?

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
connected signing poorly. Fluent signers do not pause between words. v1 therefore asks
users to sign word-by-word, which is a real usability cost, disclosed rather than
hidden. True continuous recognition (CTC over the landmark stream) is Phase 4 and
depends on data volumes that do not yet exist for LIU or ESL.

### B4 — Recognition

**Does:** utterance clip → gloss hypothesis with confidence.

Transformer encoder over landmark sequences (see `docs/TRAINING.md`). One **shared
backbone** with a **per-language head**, selected by the visitor's language choice.

Rationale: the three sign languages differ in vocabulary but share the underlying
structure of human signing — handshape, movement, spatial relationships. A shared
backbone lets ASL's larger public data improve the LIU and ESL encoders, which is the
single most valuable transfer available given LIU has no public dataset at all. Heads
stay separate because the vocabularies are genuinely different languages, not dialects.

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
which on this board means multi-second stalls rather than graceful slowdown. **v1 uses
a 3B-class quantised model locally (~2.5 GB), with an optional cloud model when the
network is available and the site permits it.** This is a product decision, not a
config value: it bounds how sophisticated answers can be.

### B8 — Sign playback

**Does:** answer text → signed video.

**A recorded clip library, sequenced by word — not a generative avatar.** Generating
grammatical sign language from text is an open research problem; a system that
generates plausible-looking but wrong signing is actively harmful to the person relying
on it. Recorded clips of real signers are correct by construction.

Cost, stated plainly: every word in every answer must be recorded, per language.
Coverage is a content problem, and the console warns when knowledge-base text contains
words the library cannot render (FR-W7).

Playback highlights each word as its clip plays, tying the signed and written answer
together.

### B9 — Session and telemetry

**Does:** session state machine, structured logs, metrics.

Every state is visible on screen (`deaf-kiosk-ux` skill) and every transition is
logged with a correlation ID shared with the server. Telemetry covers latency per
stage, frame rate, temperature, throttle state, memory, and interaction outcomes.

Temperature and throttling are logged specifically because a kiosk slowing down in
August is otherwise indistinguishable from a software regression.

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

Nightly candidate training on validated data. Runs off-device on a GPU; kiosks never
train.

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

1. **Answering model** — local 3B only, or cloud-primary with local fallback? Bounds
   answer quality against privacy and connectivity.
2. **Clip library production** — who records the sign playback library, per language,
   and how large is the initial word list?
3. **Reviewer capacity** — the learning loop's throughput is entirely gated by Deaf
   reviewer hours. How many hours per week are actually available?
4. **Fluent vs word-by-word signing** — is asking visitors to pause between signs
   acceptable for v1, or does the product require fluent recognition from day one?
   This single answer moves the timeline by many months.
5. **Deployment sites** — one site or several, and do they share a knowledge base?

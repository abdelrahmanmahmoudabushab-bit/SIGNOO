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

### 1.1b One isolated engine per language

Decided by the owner on 2026-10-11: every language has its own engine, and pressing a
language card takes the visitor to that language's wholly isolated system. This
replaces the earlier design of one shared recognition backbone with a small head per
language (B4 records what that design was avoiding and what isolation costs).

An engine is six parts, all owned by one language and listed in one entry of
`core/languages.json` (loaded by `kiosk/languages.py`):

| Part | Block | Entry key | If the language does not have it |
|---|---|---|---|
| Model checkpoint | B4 | `model` | The language is not ready. Selecting it is refused |
| Vocabulary | B5 | `vocab` | The language is not ready. Selecting it is refused |
| Confidence bars (reject, confident) | B4 | `reject_threshold`, `confident_threshold` | The kiosk defaults apply |
| Word-boundary tuning | B3 | `segmentation` | The kiosk defaults apply |
| Knowledge base | B7 | `knowledge_base` | Every question is refused |
| Clip library | B8 | `clip_library` | Nothing is played back |

The rules:

1. **The card selects the whole engine as a unit.** One call
   (`KioskConnection._use_engine` in `kiosk/main.py`) replaces the connection's
   recogniser, vocabulary, segmenter, dialogue and clip library together. There is no
   state in which a visitor has one language's model and another language's vocabulary.
2. **No fallback across languages.** A language with no trained model is refused with
   the error `language_unavailable` and is never served by another language's model. A
   language with no knowledge base of its own refuses questions. A language with no
   clip library plays nothing back. Defaults for confidence bars and word-boundary
   tuning are kiosk-wide numbers, not another language's tuning.
3. **Nothing learned is shared.** No shared backbone, no shared head, no shared
   vocabulary, no shared training data, no shared benchmark.
4. **The server keeps the same separation.** Each language has its own reviewed data,
   its own frozen benchmark signers, its own champion, its own gate verdicts and its own
   automatic 3-day candidate cycle (B12, B13).

What stays common to all languages, because it is code and not learned language
content: camera capture (B1), landmark extraction (B2), the feature encoding
(`core/features.py`), the model architecture code (`ml/model.py`, one class, different
weights per language), speech output (B6), the session state machine (B9) and data
capture (B10). B7's phrasing model is also common; it only rewords facts retrieved from
the selected language's own knowledge base.

**Where the code does not yet meet this rule** (read on 2026-10-11; each is repeated in
the block it belongs to):

- **Fixed 2026-10-12:** `liu` is marked `"status": "live"` with `"real_data": false` and
  runs `models/signoo.pt`, a model trained on synthetic data. A language is now pressable
  only if it is `selectable` (`kiosk/languages.py`): its model and vocabulary load **and**
  it was trained on real people, unless the deployment sets `SIGNOO_ALLOW_SYNTHETIC=1`.
  `GET /languages` reports `ready` as that answer and `model_loaded` separately. The
  consequence is deliberate: a production kiosk shows `liu` as "coming soon" until real
  signer data exists. The kiosk test suite and a dev machine opt in explicitly. (The
  `core/languages.json` `status` value `"live"` for `liu` is still wrong by the same rule
  and should become `"pilot"` or `"coming_soon"`; it is data, not changed here.)
- **Fixed 2026-10-12:** the server's knowledge base (B14) had no language field. Entries
  now carry a `language`, are checked against that language's own vocabulary and clip
  library, are written to that language's own file named in `core/languages.json`, and are
  delivered to a kiosk per language over the network (`GET /fleet/kiosks/{id}/kb`,
  `kiosk/kb_sync.py`). A language with no `knowledge_base` file listed refuses new entries.
- The server's fleet record (B15) still holds one `language` and one
  `current_model_version` per kiosk beside the new per-language `model_versions`, and
  the kiosk always registers as `liu`. Per-language model distribution is being built
  on 2026-10-11 and exists only as uncommitted work (B15).
- The automatic cycle runs only for the languages named in
  `SIGNOO_AUTO_TRAIN_LANGUAGES`, which defaults to `liu` alone.
- A kiosk connection holds the `liu` engine from the moment it opens until a card is
  pressed, and returns to it when the session ends. A client that skips the `language`
  message is served by the `liu` engine. The kiosk's `/health` status also reflects
  `liu` only.
- The frontend loads `/core/vocab.json` (the `liu` vocabulary) for its on-screen labels
  if `GET /languages` gave it no `vocab_url` for the selected card
  (`assets/app.js`). This is a cross-language fallback in the browser, for labels only.

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

Word-boundary tuning is part of each language's engine (§1.1b), because signing tempo
differs between languages and between people. A language's `segmentation` entry in
`core/languages.json` may set any of `pause_ms`, `motion_threshold`, `hands_absent_ms`
and `min_frames`. A key it leaves out takes the kiosk default from `kiosk/config.py`
(500 ms, 0.02, 800 ms, 6 frames). Each connection builds a fresh segmenter from the
selected language's values when the card is pressed (`LanguagePack.make_segmenter`).

The design requires these values to be tuned on real recordings of that language and
never guessed. The code does not meet that yet: no language sets a `segmentation` entry
today, so `liu` and `asl` both run on the kiosk defaults, and `kiosk/config.py` says
those defaults were tuned on synthetic data.

Word-boundary tuning is part of a language's engine (§1.1b). A language's `segmentation`
entry in `core/languages.json` may set any of `pause_ms`, `motion_threshold`,
`hands_absent_ms` and `min_frames`; a key it leaves out takes the kiosk default from
`kiosk/config.py`. Each connection builds its segmenter from the selected language's
pack (`LanguagePack.make_segmenter`), so the segmenter changes together with the model.
The state machine code is common to all languages; only the numbers differ.

As built on 2026-10-11, no language sets a `segmentation` entry, so every language runs
on the kiosk defaults, and `kiosk/config.py` says those defaults were tuned on
synthetic data. The "tuned on real recordings" promise above is not met yet for any
language.

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

Transformer encoder over landmark sequences (see `docs/TRAINING.md`). **One complete
model per language** (2026-10-11 revision, §1.1b). Each language has its own checkpoint
holding every weight, encoder and classification layer alike, and its own label list.
Languages share the model code (`ml/model.py`) and the feature encoding
(`core/features.py`). They share no learned parameters.

**As built.** At process start `kiosk/languages.py` builds one `Recognizer` for every
entry in `core/languages.json` that names both a `model` and a `vocab`. All of them
stay loaded for the life of the process, and the language card decides which one a
connection uses. A checkpoint that is missing, corrupt or missing required keys makes
that one language not ready. The process still starts and the other languages are
unaffected.

**Confidence bars are per language.** Each recogniser is built with its own reject
threshold (below it there is no hypothesis) and confident threshold (below it the
visitor is offered candidates instead of a committed word, FR-K8). A language sets them
with `reject_threshold` and `confident_threshold` in its entry. Otherwise the kiosk
defaults apply (0.55 and 0.80 in `kiosk/config.py`). Today `asl` sets a confident
threshold of 0.88 and `liu` uses the defaults.

**What this replaces, and what it costs.** The earlier revision of this section
specified one shared backbone with a small per-language head. It had two reasons:

1. *Memory.* One backbone plus N small heads is resident instead of N full models.
2. *Transfer.* Sign languages share the structure of human signing (handshape,
   movement, spatial relationships), so a shared backbone would let ASL's larger public
   data improve languages that start with little or none of their own.

Isolation gives up both. The cost in numbers, for the model as it is today:

- The model has about 1.27 million parameters (1,267,436 plus 193 per label, counted
  from `ml/model.py` at its default sizes with the 406-value input). A per-language head
  would have been those 193 parameters per label. Everything else is the part the old
  design shared.
- A checkpoint is about 5.5 MB on disk (`models/signoo.pt` is 5,495,417 bytes,
  `models/asl_pilot.pt` is 5,489,463 bytes).
- As 32-bit floats the weights are about 5.1 MB per language in memory. Nine languages
  are about 46 MB of weights, against about 5 MB for one shared backbone with nine
  heads. Isolation therefore costs roughly 40 MB at nine languages, about 0.5% of the
  8 GB budget.

These are arithmetic from the parameter count and file sizes, not measurements. Nobody
has measured recognition memory on the Jetson, with one language or with several, and
a TensorRT engine per language carries working memory that this arithmetic does not
cover. §4.2 says what must be measured. The cost also grows with the model: a model ten
times larger makes isolation cost ten times more, so this decision must be revisited if
the architecture grows.

The transfer benefit is not replaced by anything today. Every model is trained from
scratch (`ml/train.py` has no option to start from existing weights). Whether a
language's own model may be initialised from weights pre-trained on another language's
data, and then trained, gated and shipped as its own separate checkpoint, is an open
decision for the owner (§8).

What isolation buys: a change to one language cannot regress another, each gate verdict
(B13) covers exactly one deployable file, a language can be added, retrained, rolled
back or removed without touching the others, and there is no code path that can attach
the wrong head to a visitor.

**Language packs (2026-10 revision, amended 2026-10-11).** The visitor-facing language
menu lists nine cards — Jordan (LIU), Saudi, UAE, Omani, Qatari, French (LSF), American
(ASL), British (BSL), International Sign — each its own engine, dataset, and
signer-recruitment effort, not a caption-language toggle. Only a language with a real
(non-synthetic) dataset and a trained model of its own is enabled; every other card
says "coming soon" rather than running an untrained or synthetic-only model live.
Which language is recruited next, after LIU, is an open decision (`docs/ROADMAP.md`) —
the architecture does not need that decision made to support the menu.

The state of the nine entries on 2026-10-11: `liu` is `live`, `asl` is `pilot`, the
other seven are `coming_soon` with no model. The `asl` model is marked as trained on
real data (`kiosk/README.md` gives the source as PopSign). The `liu` model is
`models/signoo.pt`, trained on synthetic data, and its entry says `"real_data": false`
while its status is `live`. That contradicts the paragraph above and the project's
non-negotiable rule, and the code does not prevent it: a card is pressable when its
model and vocabulary load, whatever `real_data` says. The owner has to decide whether
`liu` goes back to "coming soon" until real data exists.

Rejection is a first-class output: anything beyond the rejection threshold returns *no
hypothesis*, and the UI asks the visitor to sign again. A confident mistranslation is
worse than an admitted failure (Requirements §6.3). A model may also carry an
`__other__` label meaning "none of my signs"; when that label wins, the result is a
rejection, and the label is never shown or offered as a candidate
(`kiosk/recognition.py`). Selecting a language that is not ready is refused in the same
spirit, with the error `language_unavailable`, and the connection stays as it was.

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

The dictionary is per language (§1.1b). Each engine has its own vocabulary file, named
by `vocab` in its `core/languages.json` entry, and the assembler for a connection is
rebuilt from that file when the card is pressed. A gloss that is not in the selected
language's vocabulary is not looked up anywhere else.

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

**The knowledge base is per language** (§1.1b). Each engine names its own file with
`knowledge_base` in `core/languages.json`, and retrieval for a connection only ever
searches the selected language's file. A language with no knowledge base, or an empty
one, refuses every question with the same visible refusal as an out-of-scope question.
It never answers from another language's content. This is the case for `asl` today:
`kiosk/data/knowledge_base_asl.json` is an empty list. The phrasing model is common to
all languages and is only ever given the facts retrieved from the selected language's
knowledge base.

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

The local 3B model is not in the code. The fallback in `kiosk/dialogue.py` today returns
the retrieved knowledge-base answer through a fixed template, with no local language
model loaded, so the 2.5 GB reserved for it in §4.2 is not yet in use.

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

**The clip library is per language** (§1.1b). Each engine names its own library with
`clip_library` in `core/languages.json`, and playback for a connection only looks up
the selected language's recordings. A sign recorded for one language is never played
for another, even where the written word is the same, because the signs are different.
A language with no clip library plays nothing back and reports every answer word as
unrenderable. As built, every entry in both existing libraries (`core/clip_library.json`
for `liu`, `core/clip_library_asl.json` for `asl`) has no recording, so playback shows
the "not yet recorded" placeholder for every word in every language.

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
      pressing a ready card moves the session onto that language's whole engine,
      §1.1b; an unready card says "coming soon" and cannot start a live session)
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

**Training is per language** (§1.1b). A candidate for a language is trained only on
that language's reviewed data, from scratch, into its own checkpoint. No run reads
another language's data or weights.

As built (`server/training.py`, `server/scheduler.py`):

- Only review items of that language with status `confirmed` or `corrected` are
  exported. Items with no signer identity are left out.
- The split is by signer. The first export for a language freezes its validation and
  test signers in `server/runs/frozen_split.json` under that language's code. Later
  exports keep them and send every new signer to train. Training refuses to run if a
  frozen signer's data has disappeared, or if fewer than three signers exist.
- The automatic cycle is real: the server starts a scheduler that checks hourly and
  trains a candidate for a language when 72 hours have passed since that language's
  last model. It runs once per language named in `SIGNOO_AUTO_TRAIN_LANGUAGES`. That
  list defaults to `liu` only, so another language gets no automatic cycle until an
  operator adds its code.
- The scheduler will not start a language's cycle until that language's benchmark has
  been frozen by a deliberate first run (`POST /training/trigger`).
- A candidate is never promoted by this block.

The module docstring at the top of `server/training.py` still says no automatic cycle
exists. That comment is out of date; the scheduler is in the code.

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

**The registry and the gate are per language** (§1.1b). Every model version belongs to
exactly one language. Each language has its own champion, and a candidate is judged
only against the champion of its own language, on its own language's frozen benchmark
signers, with a shadow evaluation on its own language's reviewed live items. A
language's first model has no champion to beat, but must still pass the latency and
shadow criteria. Passing the gate for one language says nothing about any other.

As built (`server/gate.py`, `server/shadow.py`, `server/routers/registry.py`):

- The champion lookup, the shadow window and the registry listing all filter by the
  candidate's language.
- Promotion is a separate action by an engineer or admin, allowed only for a version in
  `shadow` status whose gate passed. Nothing automatic can promote.
- `POST /training/trigger` accepts an explicit `champion_version`. In the committed
  code it is not checked against the candidate's language. Uncommitted work on
  2026-10-11 adds that check (a champion of another language is refused with 409
  before any training starts); confirm it is merged before relying on it.
- The false-accept criterion (DR-6 item 5) is not measured. `server/gate.py` treats it
  as passed when either figure is missing, which is always, because the benchmark has
  no non-signing clips.
- The latency criterion is measured on the server's CPU, not on the Jetson.
- The shadow evaluation is not a parallel run on live kiosks. It scores the candidate
  on reviewed items newer than the candidate and compares that with what the kiosk's
  model said at the time.
- Automatic rollback on live-metric degradation is not built. Manual rollback changes
  the version's status and does not move any kiosk back by itself.

### B14 — Knowledge base

Per-site content for B7, edited in the console with preview of the signed rendering and
warnings for unrenderable words.

Under §1.1b this content is per language as well as per site: each language's engine
answers only from its own knowledge base, and the unrenderable-word warning must be
computed against that language's own vocabulary and clip library.

The server does not do this yet. A knowledge-base entry has a `site_id` and no language
(`server/models.py`). Every entry is written to one file,
`kiosk/data/knowledge_base.json`, which is the `liu` engine's knowledge base, and the
warning is computed from `core/vocab.json` and `core/clip_library.json`, which are the
`liu` files (`server/config.py`, `server/kb.py`). So the console can only edit `liu`
content, and any other language's knowledge base is a file edited by hand on the kiosk.

### B15 — Fleet control

Staged rollout of software and models, health monitoring, remote diagnostics,
automatic rollback. Never all kiosks at once.

**Model rollout is per language** (§1.1b). A kiosk runs one model per ready language,
so its fleet state is one installed version per language, and a rollout of a model
moves only the engine of that model's language on the targeted kiosks. A new `asl`
model must never replace or disturb the `liu` engine on the same kiosk.

State of the code on 2026-10-11. This block is being built by two engineers today. The
model-distribution parts below were read in the uncommitted working tree, were not run
for this document, and are still changing, so check the code before relying on them:

- Registration, heartbeat, the kiosk list and rollout creation exist
  (`server/routers/fleet.py`). A rollout names a model version and a list of kiosk ids.
  Uncommitted: only a `promoted` version can be rolled out, so a rollout cannot put an
  ungated model on a kiosk.
- The kiosk side of model distribution is written (`kiosk/model_sync.py`): after each
  delivered heartbeat it asks the server, once per language that has a model
  configured, whether a different version is assigned. It installs a download only if
  the sha256 matches the server's header and the file loads as a checkpoint, then swaps
  that one language's recogniser and records the version so a restart keeps it. A bad
  download leaves the running model untouched.
- The server endpoint that answers that request is written, uncommitted
  (`GET /fleet/kiosks/{kiosk_id}/model?language=<code>`). It answers with the model of
  the newest active rollout that targets the kiosk and whose model is of the requested
  language, so a rollout of one language's model never changes what a kiosk is told to
  run for another. The contract is in `docs/API_CONTRACTS.md` §10.
- The kiosk sends `model_versions` (language code to installed version) in its
  heartbeat and the server stores it per kiosk.
- The older single-language fields remain: a kiosk record has one `language` and one
  `current_model_version`, and the kiosk always registers as `liu`. A rollout updates
  `current_model_version` only when the model's language equals the kiosk's registered
  language.
- A rollout is never closed or superseded in code: nothing sets a rollout's status to
  anything but `active`, so "most recent active rollout" is in practice "most recent
  rollout", and moving a kiosk back means creating a new rollout of the older version.
- Staging, health-based automatic rollback and remote diagnostics are not built.
- The transport is not mTLS. Kiosk calls carry a shared secret in the `X-Kiosk-Token`
  header (`docs/API_CONTRACTS.md` §9), which identifies the fleet, not the individual
  kiosk.

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
recognition (all ready languages) ~0.2 GB
answering model (3B quantised)    ~2.5 GB
speech synthesis                  ~0.2 GB
buffers + clip libraries + slack  ~1.0 GB
                                 ────────
                                  ~6.0 GB  of 8 GB shared
```

Models load once at startup. Memory growth is monitored, not assumed absent — a kiosk
runs for weeks, and leaks surface on day nine, not in a demo.

**Recognition under one engine per language (2026-10-11).** The recognition line used
to read "one active language". The kiosk as built keeps every ready language's model
loaded at once, so the line now covers all of them. What that means in plain numbers:

| | Per language | Nine languages |
|---|---|---|
| Checkpoint on disk | about 5.5 MB | about 50 MB |
| Weights in memory (32-bit floats, 1.27 million parameters) | about 5.1 MB | about 46 MB |
| Share of the 8 GB budget | about 0.06% | about 0.6% |

The inference runtime (PyTorch today) is loaded once however many languages there are,
and it is most of the 0.2 GB. Each additional language adds about 5 MB of weights. By
this arithmetic nine isolated models fit inside the existing 0.2 GB line, the total
stays at about 6.0 GB, and isolation costs about 40 MB more than the shared-backbone
design it replaced (B4).

**None of this has been measured on the Jetson.** Every figure in this section is an
estimate, and the per-language figures are arithmetic from the parameter count and the
file sizes on a development machine. Before a second language goes live on a device,
measure resident memory on the Jetson, at thermal steady state, with every ready
language loaded. The things the arithmetic does not cover:

- **TensorRT.** The design converts recognition to TensorRT (§5), and that conversion
  has not been done. Each language would be its own engine with its own execution
  context and working memory. That overhead per engine is unknown and may be larger
  than the weights.
- **Clip libraries.** Every language has its own (B8). Once real motion is recorded,
  their size grows with words times languages and will probably exceed the models. The
  1.0 GB line was written for one library.
- **Knowledge bases and vocabularies.** Per language, and small.
- **The answering model.** The 2.5 GB line is a reservation. No local model is loaded
  today (B7).

If the measurement shows that all ready languages do not fit, the fallback is to load a
language's engine when its card is pressed and unload the least recently used one. That
costs a visible wait on the language screen and is not built.

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
| Memory pressure | Unload the answering model first — recognition is the core service, answering is the enhancement. Never free memory by serving one language with another language's model. Not built: nothing in the kiosk unloads anything today |
| Model file corrupt | Refuse to load that model. Only that language becomes not ready; its card cannot be pressed and the other languages keep working. Fall back to the previously known-good version *of the same language*, never to another language's model. As built: a rolled-out model that fails to load falls back to that language's factory checkpoint, and a factory checkpoint that fails leaves the language not ready. No older rolled-out version is kept |
| Visitor selects a language with no trained model | Refused with `language_unavailable`; the session stays where it was. The card should already be disabled and say "coming soon" |
| Selected language has no knowledge base | Every question gets the visible refusal and the route to a human. Never answered from another language's knowledge base |
| Selected language has no clip library, or a word has no recording | Nothing is played for it; the word is shown as not yet recorded. Never another language's sign, never a generated one |
| Downloaded model fails its sha256 or will not load | Discarded. The running model for that language is untouched; other languages are not involved |
| Promotion gate rejects candidate | That language's champion stays; reason recorded; alert if rejections repeat, since that means the review pipeline is producing bad data. Other languages' champions are not affected either way |
| Bad model promoted anyway | Automatic rollback on live-metric degradation; one-action manual rollback. Both act on one language. As built: the automatic rollback does not exist, and the manual one changes the registry status without moving any kiosk |
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
   Since 2026-10-11 a knowledge base is also per language (§1.1b), so the question is
   per site and per language.
6. **Cross-language pre-training (opened 2026-10-11).** With one isolated model per
   language, may a language's model start from weights pre-trained on another
   language's data, as long as it is then trained, gated and shipped as its own
   checkpoint? The shared backbone existed partly for this transfer (B4). Nothing in
   the code does it today.
7. **LIU's card (opened 2026-10-11).** `liu` is live on a model trained on synthetic
   data, against the rule that such a language says "coming soon" (B4). Take it off the
   menu until real data exists, or accept it as a development-only state?
8. **Memory with every language loaded (opened 2026-10-11).** Not measured on the
   Jetson (§4.2). If it does not fit, loading on selection is the fallback, at the
   price of a wait on the language screen.

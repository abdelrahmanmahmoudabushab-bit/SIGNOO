# SIGNOO — Delivery Plan

**Version** 0.1 (draft for approval)

Sequenced so that the riskiest assumption is tested first and cheaply. The largest risk
is not the software — it is **whether a dataset can be collected at all**. Everything
downstream is worthless if signer recruitment fails, so it starts in week one and never
stops.

---

## Phasing principle

Each phase ends with a **decision point**, not a demo. If the exit criteria are not met,
the correct action is to change the plan, not to proceed and hope.

One language (LIU) is taken end to end before the second is started. Building three
languages in parallel triples the data burden while learning nothing three times over.

---

## Phase 0 — Foundations · ~3 weeks

**Goal:** everything needed to collect data, and proof the hardware can do the job.

- Repository restructure (`core/`, `kiosk/`, `ml/`, `server/`, `console/`)
- Shared feature code in `core/`, used by both training and kiosk
- Recording studio extended: three languages, session management, direct upload
- Jetson bring-up: JetPack, TensorRT, camera, power mode, thermal baseline
- **Perception spike:** MediaPipe Holistic → TensorRT on device, measured hot
- Ethics and consent pack: forms in Arabic, English, and sign video; legal review
- Deaf reviewer and signer recruitment begins

**Exit criteria**

- Perception sustains ≥ 30 fps for 20 minutes at thermal steady state
- Latency for capture → landmarks measured and within budget
- Consent pack approved by legal review
- ≥ 5 signers committed to recording sessions

**Kills the project early if:** perception cannot hold 30 fps on this board, or signer
recruitment finds no willing participants. Both are better discovered in week three
than month six.

---

## Phase 1 — LIU word recognition · ~6 weeks

**Goal:** honest signer-independent recognition of a real vocabulary.

- Collect **50 signs × 10 signers × 5 repetitions** (LIU)
- Deaf-reviewer validation of every recording
- Frozen benchmark set defined; test signers fixed and quarantined
- Train, evaluate, iterate — on validation signers only
- Promotion gate implemented and unit-tested, including rejection of a deliberately
  worse model

**Exit criteria**

- ≥ 85% top-1, signer-independent (NFR-Q1)
- ≥ 70% worst-signer (NFR-Q2)
- ≥ 95% top-3 (NFR-Q3)
- Gate demonstrably rejects a worse model

**Decision point.** If accuracy falls short, the answer is almost always *more signers*,
not more model. Adding signers extends this phase; it does not move the problem
downstream.

---

## Phase 2 — Kiosk runtime · ~6 weeks

**Goal:** a person can walk up and be understood.

- B1–B6 on device: capture, perception, segmentation, recognition, assembly, output
- Segmentation state machine tuned on real recordings
- Kiosk UI: attract → language → consent → capture → result
- Piper TTS, Arabic and English
- Session state machine, telemetry, crash recovery
- Offline operation throughout

**Exit criteria**

- NFR-P1 (≤ 300 ms) met on device, hot
- NFR-M1 (≤ 6 GB) met
- Five Deaf users complete a session unaided
- 72 hours unattended without intervention

**Decision point.** Segmentation quality decides whether the word-by-word interaction is
acceptable. This is where Open Question 4 gets answered by evidence rather than
opinion.

---

## Phase 3 — Answering and signed replies · ~6 weeks

**Goal:** the visitor gets an answer, and can see it in sign.

- B7 dialogue: retrieval over a site knowledge base, 3B local model
- Refusal and escalation paths
- **Sign clip library recorded** for the answer vocabulary (LIU) — content work,
  starts in Phase 2 because it is slow
- B8 playback with per-word highlighting
- Knowledge base editing with unrenderable-word warnings

**Exit criteria**

- Answer within NFR-P3 (≤ 3 s local)
- Memory still within NFR-M1 with the language model resident
- Out-of-scope questions refused, not invented
- Ten Deaf users report the signed answer was understood (≥ 8 of 10)

---

## Phase 4 — Learning loop and console · ~8 weeks

**Goal:** the system improves from use, safely, under human control.

- B10 capture with opt-in consent; landmarks by default
- B11 ingest and prioritised review queue
- Reviewer workspace built for volume
- B12 nightly candidate training
- B13 registry, gate, shadow-running, rollback
- Console: fleet view, datasets, training visibility, model registry, logs, alerts,
  roles
- B15 staged fleet rollout

**Exit criteria**

- One full cycle observed: capture → review → candidate → gate → shadow → promotion
- Gate rejects a deliberately degraded candidate in the live pipeline
- Rollback exercised end to end
- Deletion request honoured through to training snapshots (NFR-S6)

---

## Phase 5 — Second and third languages · ~8 weeks each, partly parallel

**Goal:** ESL and ASL on the same rails.

- Data collection per language (ESL is the harder one — ESL127 is a seed, not a
  training set; ASL has substantial public data)
- Per-language heads on the shared backbone
- Clip libraries per language
- Language selection wired end to end

Running these after Phase 4 means each new language inherits a working learning loop
rather than needing its own.

---

## Phase 6 — Production hardening · ~6 weeks

- 14-day unattended soak (NFR-R2)
- Thermal validation at 35 °C ambient in the real enclosure
- Security review, penetration test, disk encryption
- Privacy review sign-off per deployment jurisdiction
- Operator runbooks and training
- Full acceptance run against Requirements §8

---

## Beyond v1

Ordered by value, not ease:

1. **Face landmarks** — recovers grammatical expression, the largest single accuracy
   and correctness gain available (Requirements §7.1)
2. **Continuous recognition** — CTC over the landmark stream, removing the word-by-word
   constraint. Needs data volumes that do not exist yet for LIU or ESL
3. **Learned gloss → text** — fluent Arabic instead of templates
4. **Public dataset release** — with consent, the first open LIU dataset; the project's
   most durable contribution
5. **Two-way** — voice input for hearing users, answered in sign

---

## Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **Too few signers recruited** | Fatal — no dataset, no model | Start week one; partner with Deaf clubs and institutes; compensate; treat signer count as the primary project metric |
| R2 | Accuracy misses NFR-Q1 | Timeline slip | More signers before more model. Budgeted as a Phase 1 extension, not a surprise |
| R3 | **Reviewer capacity limits the loop** | Learning loop stalls | Prioritised queue so scarce hours go to the highest-value samples; measure hours in Phase 0, not Phase 4 |
| R4 | Fluent signing unsupported (Open Q4) | Product acceptability | Test with Deaf users in Phase 2, before committing to Phase 3 |
| R5 | 8 GB budget exceeded | Redesign of B7 | Budget written down in Phase 0; measured every phase; answering model is the first thing unloaded under pressure |
| R6 | Thermal throttling in enclosure | Latency and stability | Measure hot in Phase 0; validate the real enclosure, not a bench board |
| R7 | Clip library production underestimated | Phase 3 slip | Start recording in Phase 2; console flags coverage gaps |
| R8 | Regional variant disputes | Wrong signs shipped | Deaf reviewers are authoritative; variants recorded, not normalised by hearing judgment |
| R9 | Privacy/regulatory blocker | Deployment blocked | Legal review in Phase 0, per jurisdiction, before any collection |
| R10 | Self-training degrades models | Silent quality collapse | Frozen human-verified benchmark the training cannot touch; shadow run; automatic rollback |
| R11 | Public-site abuse or misuse | Reputational | Escalation paths; no medical/legal/emergency use; visible uncertainty |

---

## What decides the schedule

Not engineering effort. **Signer recruitment and Deaf reviewer hours** set the pace of
Phases 1, 4, and 5. Every estimate above assumes those are staffed; if they are not,
the phases stretch regardless of how fast the code is written.

Treat *number of signers recorded* and *clips reviewed per week* as the two headline
project metrics from week one.

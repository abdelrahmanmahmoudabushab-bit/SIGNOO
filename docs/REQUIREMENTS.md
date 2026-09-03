# SIGNOO — Requirements

**Version** 0.1 (draft for approval) · **Status** awaiting sign-off

A public kiosk that lets a Deaf person walk up, sign in their own sign language, be
understood, get an answer, and see that answer explained back in sign.

---

## 1. Scope

### 1.1 In scope

A self-contained kiosk running on NVIDIA Jetson Orin Nano Super with a 60 fps camera
and a touchscreen, plus a web console for operators, and a training service that
improves the models from real use under human supervision.

Three sign languages, each a distinct language with its own models and vocabulary:

| Code | Language | Written outputs |
|---|---|---|
| `liu` | Jordanian Sign Language · لغة الإشارة الأردنية | Arabic + English |
| `esl` | Emirati Sign Language · لغة الإشارة الإماراتية | Arabic + English |
| `asl` | American Sign Language | English (+ Arabic optional) |

### 1.2 Explicitly out of scope for v1

Naming these now prevents a slow, silent expansion of scope later.

- **Free-form conversational sign translation.** v1 recognises a defined vocabulary,
  not arbitrary signing.
- **Generative signing avatar.** Sign output is assembled from a recorded clip
  library, not synthesised from text (§4.7).
- **Facial-expression grammar.** The perception stack is hands and upper body. Signs
  distinguished only by expression cannot be separated (§7.1).
- **Voice input from hearing users.** One-way: sign in, text and voice out.
- **Fingerspelling of arbitrary words** beyond the trained alphabet sets.
- **Medical, legal, or emergency interpretation.** The system must not be presented as
  a substitute for a qualified human interpreter (§6.3).

---

## 2. Users

| Role | Needs | Constraint that drives design |
|---|---|---|
| **Deaf visitor** | Ask a question in their sign language, get an answer they can read *and* see signed | Cannot hear any prompt, chime, or error. All state must be visual |
| **Hearing companion / staff** | Follow the exchange | Reads Arabic or English, not sign |
| **Operator** | Keep kiosks running, watch quality, act on failures | Not an ML engineer. Needs plain answers, not metrics dumps |
| **Deaf reviewer / linguist** | Validate recognitions, correct labels, approve vocabulary | The authority on correctness. Their throughput gates retraining |
| **ML engineer** | Train, evaluate, promote models | Needs the honest number, not the flattering one |

---

## 3. Functional requirements

### 3.1 Kiosk — visitor flow

- **FR-K1 Idle attract.** Idle screen invites approach in all three sign languages by
  video, without text-only instructions.
- **FR-K2 Language selection.** First interaction is choosing sign language, presented
  as three large targets labelled by flag, name, and a short signed clip. Selection is
  reachable within one touch from idle.
- **FR-K3 Consent gate.** Before the camera starts, state in sign video and text what
  is captured and kept. Using the kiosk requires consent to *processing*; consent to
  *retention for training* is a separate, optional choice, and declining it must not
  degrade the service (§6.1).
- **FR-K4 Live capture.** Camera preview with tracked skeleton overlay, visible at all
  times while capturing, so the user can see the system sees them.
- **FR-K5 Framing feedback.** When hands leave frame, the body is not detected, or
  lighting is inadequate, show it visually and immediately.
- **FR-K6 Sign recognition.** Recognise signs from the active language's vocabulary
  and display each recognised word as it is produced, not only at the end.
- **FR-K7 Sentence assembly.** Group recognised signs into an utterance using pause
  segmentation, and render it as Arabic and English text side by side.
- **FR-K8 Uncertainty handling.** Below the confidence threshold, present the top
  candidates as large touch targets rather than committing to a guess. Never silently
  substitute.
- **FR-K9 Correction.** Remove the last word, or clear the utterance, from any screen
  where text is shown.
- **FR-K10 Voice output.** Speak the assembled text in Arabic and English on demand.
  Text must appear before, and independently of, audio.
- **FR-K11 Answer.** Send the assembled utterance to the answering service and display
  the reply as Arabic and English text.
- **FR-K12 Signed answer.** Play the reply back as sign, assembled from the clip
  library for the active language, with the matching word highlighted as each clip
  plays (§4.7).
- **FR-K13 Session end.** Return to idle on completion, on a visible countdown after
  inactivity, or on one touch. Clear all session content on return to idle.
- **FR-K14 Offline operation.** Recognition, translation, and voice continue with no
  network. Answering degrades to the local model and a clear notice.

### 3.2 Answering service

- **FR-A1 Grounded answers.** Answer from a curated knowledge base for the deployment
  site. Refuse, visibly, when the question falls outside it rather than improvising.
- **FR-A2 Sign-first phrasing.** Produce short, concrete sentences suitable for
  rendering as sign, avoiding idiom and subordinate clauses.
- **FR-A3 Vocabulary awareness.** Prefer wording whose words exist in the sign clip
  library, so the signed answer is complete; mark any word rendered by fingerspelling.
- **FR-A4 Bilingual output.** Every answer in Arabic and English.
- **FR-A5 Escalation.** Offer a route to a human interpreter or staff member when the
  question is out of scope, safety-related, or repeatedly not understood.

### 3.3 Learning loop

- **FR-L1 Capture.** With opt-in consent, record each interaction's landmarks,
  prediction, confidence, and any user correction.
- **FR-L2 Weak labels.** Treat an on-screen candidate selection or correction as a
  weak label — evidence, never ground truth.
- **FR-L3 Review queue.** All captured samples land in a queue for Deaf-reviewer
  validation. Nothing enters training unreviewed.
- **FR-L4 Prioritised review.** Order the queue by expected value — low-confidence
  cases, corrected cases, and under-represented signs first.
- **FR-L5 Nightly candidate.** Train a candidate model on validated data on a schedule.
- **FR-L6 Promotion gate.** Promote only on passing every criterion in §5.3. Otherwise
  keep the champion and record why.
- **FR-L7 Shadow run.** Run a passing candidate against live traffic without affecting
  output, for a configured period, before promotion.
- **FR-L8 Rollback.** Revert to the previous model in one action, and automatically on
  configured live-metric degradation.

### 3.4 Operator console (web)

- **FR-W1 Fleet view.** Every kiosk with health, current model version, uptime,
  temperature, and last contact.
- **FR-W2 Live session view.** Watch an in-progress session — recognitions and
  confidences — with camera imagery gated behind explicit authorisation and audit.
- **FR-W3 Review workspace.** Play a clip, see prediction and confidence, confirm or
  correct the label, flag for a linguist. Optimised for volume: keyboard-driven, one
  decision per screen.
- **FR-W4 Dataset management.** Browse the dataset by language, sign, and signer; see
  coverage gaps and per-sign counts; upload recorded sessions; retire bad data.
- **FR-W5 Training visibility.** Current and past runs, curves, the promotion-gate
  verdict with each criterion's result, and the diff against the champion.
- **FR-W6 Model registry.** Every model version with its metrics, training data
  snapshot, and deployment history. Promote, roll back, or pin per kiosk.
- **FR-W7 Knowledge base editing.** Edit answering content, with preview of the signed
  rendering and a warning for words missing from the clip library.
- **FR-W8 Logs and alerts.** Searchable structured logs; alerts for kiosk offline,
  thermal throttling, accuracy drop, and review queue backlog.
- **FR-W9 Access control.** Roles — operator, reviewer, engineer, admin — with least
  privilege. Camera imagery requires elevated rights and is audited.

---

## 4. Non-functional requirements

### 4.1 Performance

| Metric | Target | Hard limit |
|---|---|---|
| **NFR-P1** End of sign → word on screen | ≤ 300 ms | 500 ms |
| **NFR-P2** Utterance complete → assembled text | ≤ 600 ms | 1 s |
| **NFR-P3** Question → answer text (local model) | ≤ 3 s | 6 s |
| **NFR-P4** Camera capture | 60 fps sustained | no visible stutter |
| **NFR-P5** Perception | 30 fps sustained, hot | ≥ 24 fps |
| **NFR-P6** Cold start to attract screen | ≤ 60 s | 120 s |

Measured on the device at thermal steady state, not from a cold boot (§4.3).

### 4.2 Memory

**NFR-M1.** Total resident memory ≤ 6.0 GB, leaving headroom on the 8 GB shared
budget. Allocation is a design constraint, not an outcome:

```
OS + display + application        ~1.5 GB
perception (landmarks)            ~0.6 GB
recognition models (×1 active)    ~0.2 GB
answering model (3B, quantised)   ~2.5 GB
speech synthesis                  ~0.2 GB
buffers, clip library, slack      ~1.0 GB
```

**NFR-M2.** No unbounded growth: resident memory after 14 days continuous operation
within 10% of the value after 1 hour.

### 4.3 Reliability

- **NFR-R1** 99% availability during opening hours, per kiosk, per month.
- **NFR-R2** Unattended operation ≥ 14 days without manual intervention.
- **NFR-R3** Automatic recovery from process crash within 30 s, returning to idle.
- **NFR-R4** No single failure — network, answering service, cloud — takes recognition
  and translation offline.
- **NFR-R5** Thermal: sustained operation at 35 °C ambient without dropping below the
  NFR-P5 floor.

### 4.4 Accuracy

Every figure is **signer-independent**, on signers absent from training (§5.2).

- **NFR-Q1** Top-1 word accuracy ≥ 85% on the frozen benchmark, per language.
- **NFR-Q2** Worst-signer accuracy ≥ 70%. The mean may not hide a failing signer.
- **NFR-Q3** Top-3 ≥ 95%, since the interface can offer candidates (FR-K8).
- **NFR-Q4** False-accept rate on non-signing movement ≤ 2%.
- **NFR-Q5** No promoted model regresses any individual sign by more than 5 points.

### 4.5 Security and privacy

- **NFR-S1** Video never leaves the device unless the visitor opted in, and then only
  encrypted in transit and at rest.
- **NFR-S2** Landmarks are the default retained representation; video retention is
  opt-in, time-boxed, and separately justified.
- **NFR-S3** Session data is erased on return to idle.
- **NFR-S4** Console access authenticated, role-scoped, and audited; camera imagery
  access is separately authorised and logged with reason.
- **NFR-S5** Device disk encrypted; no credentials in images or repositories.
- **NFR-S6** A visitor can request deletion of their contributed data; the pipeline
  must be able to locate and remove it, including from training snapshots.

### 4.6 Accessibility

- **NFR-X1** No information conveyed by sound alone, anywhere, ever.
- **NFR-X2** All controls reachable from a seated (wheelchair) position.
- **NFR-X3** Text contrast ≥ 7:1; minimum text size validated at 1 m viewing distance.
- **NFR-X4** All instructional content available as sign video, not only text.
- **NFR-X5** Validated with Deaf users before each release (§6.2).

### 4.7 Maintainability

- **NFR-N1** A recorded session can be replayed through the pipeline off-device,
  producing identical output — the primary debugging tool for field faults.
- **NFR-N2** Model artefacts versioned with the data snapshot and code commit that
  produced them; any deployed model reproducible from that triple.
- **NFR-N3** Structured logs with a session correlation ID across kiosk and services.
- **NFR-N4** Remote deployment of software and models to a fleet, staged, with
  automatic rollback.

---

## 5. Data and evaluation requirements

### 5.1 Dataset targets

There is no public Jordanian Sign Language dataset of usable size; Jordan accounts for
~2% of Arabic Sign Language dataset collection. Emirati has ESL127 (127 signs, 708
recordings) — a useful seed, not a training set. ASL has substantially more public
data. **The LIU and ESL datasets must be collected.**

Per language, for v1:

| | Minimum | Target |
|---|---|---|
| Signs | 50 | 150 |
| Signers | 10 | 25 |
| Repetitions per sign per signer | 3 | 5 |
| Left-handed signers | 1 | ≥ 10% of signers |

**DR-1.** Signer count is the binding constraint, not clip count: 20 signers × 5
repetitions is worth more than 3 × 50 (§5.2).

**DR-2.** Every sign validated by a Deaf reviewer as correctly produced and correctly
labelled. Automated checks cannot detect a clean recording of the wrong sign.

**DR-3.** Recording conditions deliberately varied across signers — lighting,
background, distance, clothing, device — and recorded in metadata.

### 5.2 Split protocol

**DR-4.** Splits are by signer. A person appears in exactly one of train, validation,
test — enforced in code, not by convention.

**DR-5.** The test signers are frozen at project start and never used for tuning.

Rationale, measured on this project's pipeline: a random split scored **99.2%** where
the signer-independent split scored **52.9%** — same model, same data. Reporting the
former would be reporting a number that predicts nothing.

### 5.3 Promotion gate

**DR-6.** A candidate model is promoted only if **all** hold against the champion, on
the frozen human-verified benchmark:

1. Top-1 accuracy improves, or is within 0.5 points while another criterion improves.
2. No signer regresses by more than 3 points.
3. No sign regresses by more than 5 points (NFR-Q5).
4. No class falls to zero recall.
5. False-accept rate does not increase (NFR-Q4).
6. Latency stays within NFR-P1 on device.
7. It has completed its shadow-run period without live-metric degradation (FR-L7).

**DR-7.** The benchmark set is frozen and human-verified, and the training process
cannot modify, extend, or select into it. This is what makes automatic retraining safe:
the *training* is automatic, the *judgement* is not.

---

## 6. Constraints and obligations

### 6.1 Consent and data protection

Face-and-body video of an identifiable person is biometric personal data and cannot be
anonymised. Obligations:

- Informed consent, presented in sign language, before capture.
- Separate opt-in for retention and training use.
- Full service for those who decline retention.
- Deletion on request, honoured through to training snapshots (NFR-S6).
- Jordan and UAE data-protection law reviewed per deployment before launch; a
  deployment may not proceed on assumption.

### 6.2 Community involvement

**C-1.** Deaf signers are partners, not data sources. Vocabulary, wording, and flow are
validated by Deaf reviewers, who are compensated for their time.

**C-2.** Regional variants are recorded, not normalised away by hearing judgment.

### 6.3 Safety

**C-3.** The kiosk must present itself as an assistive tool, not an interpreter. Any
context where a mistranslation carries real cost — medical, legal, financial,
emergency — routes to a human (FR-A5).

**C-4.** Wrong output must be visibly uncertain rather than confidently stated
(FR-K8).

### 6.4 Hardware

**C-5.** Jetson Orin Nano Super — 67 sparse TOPS, 8 GB shared LPDDR5 at 102 GB/s,
6-core Cortex-A78AE. The 8 GB shared budget, not compute, is the binding constraint
(§4.2).

**C-6.** Implementation language is Python for kiosk runtime, services, and training.
Performance-critical stages run through TensorRT rather than being rewritten.

---

## 7. Known limitations to state publicly

Stated here so they are decisions, not surprises.

**7.1 No facial grammar.** Hands and upper body only. Signs differing solely in facial
expression are not separable. Affects negation, question marking, and intensity.

**7.2 Vocabulary-bounded.** Recognition is limited to trained signs. Out-of-vocabulary
signing is rejected, not guessed.

**7.3 Utterances, not fluent grammar.** Output follows sign-language order and is
presented as recognised words, not idiomatic Arabic prose.

**7.4 Signed answers are assembled, not generated.** Playback of recorded clips
sequenced by word. Fluency is below that of a human interpreter, and words outside the
library are fingerspelled or flagged.

**7.5 Accuracy varies by person.** Signers unlike those in training will be recognised
less reliably. The worst-signer floor (NFR-Q2) exists to keep this bounded and visible.

---

## 8. Acceptance criteria for v1

The system is accepted when, for **one** language (LIU) at **one** site:

1. Every FR in §3 demonstrated, or formally deferred with a dated decision.
2. All NFR targets in §4 met on the device at thermal steady state.
3. NFR-Q1–Q5 met on the frozen benchmark, with test signers never used in tuning.
4. Ten Deaf users complete an unaided end-to-end session; ≥ 8 report the answer was
   understood, in structured post-session review.
5. Fourteen days unattended operation with no manual intervention (NFR-R2).
6. One full learning cycle observed: capture → review → candidate → gate → shadow →
   promotion, with the gate demonstrated rejecting a deliberately worse model.
7. Privacy review signed off; deletion request exercised end to end.

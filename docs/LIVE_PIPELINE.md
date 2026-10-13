# Live recognition pipeline

How one signed word travels from the camera to the captions, where time and accuracy are
lost, and what is planned to make it smoother. Numbers marked *measured* were taken on the
Jetson Orin Nano (CPU only); the rest are estimates and say so.

## 1. The path of one sign

| # | Step | Where | Cost |
|---|------|-------|------|
| 1 | Capture a frame every 70 ms, shrink to ≤480 px, encode JPEG. Not sent while the last frame is unanswered (gives up waiting after 200 ms) | browser (`assets/app.js`) | ~5–10 ms (estimate) |
| 2 | Send over the WebSocket `/ws/session` as base64 | browser → `kiosk/main.py` | base64 adds ~33% size; decode and hand-off ~3 ms on a PC (*measured*) |
| 3 | Find pose and both hands (MediaPipe Holistic, LIVE_STREAM, CPU) | `kiosk/perception.py` | ~100 ms per frame, **~10 updates/s on the Jetson, measured** |
| 4 | Send landmarks back the moment the tracker finishes (not when the next frame arrives); corner marks glide toward them | `kiosk/main.py` → `assets/vision.js` | one network hop |
| 5 | Cut the stream into signs: a word is ≥3 frames spanning ≥150 ms, ≤4.5 s; 500 ms of stillness or 800 ms with no hands ends one | Segmenter (`kiosk/segmentation.py`) | the wait you notice is that pause |
| 6 | Features, then one forward pass of that language's model → word probabilities | Recognizer, in a worker thread | median ~15 ms on a PC after the start-up warm-up, with stalls up to 200 ms when the machine is busy (*measured*); not timed on the Jetson |
| 7 | Compare with the bars → committed / candidates / rejected | `kiosk/main.py` | negligible |
| 8 | Join words, render Arabic and English boxes | Assembler | negligible |

Only the newest frame is kept at each hand-off (newest-wins). When tracking falls behind,
old frames are dropped rather than queued, so lag does not grow.

Two things happen before the first frame so that signing can start at once: each model
takes a throwaway pass when the kiosk starts, and the tracker starts when a language
card is pressed, while the visitor reads the consent screen (*measured on a PC*: tracker
start 3.3 s, consent to active 53 ms). The tracker is given no frame before consent.

## 2. What the visitor sees for a result

Per language, in `core/languages.json`:

- **Committed** — confidence at or above the commit bar (American: 0.80, or the word's own
  bar from the checkpoint's `word_bars`). The word joins the sentence.
- **Candidates** — between the reject bar and the commit bar. The top 3 words appear as
  touch targets. A tapped choice is stored as a weak label, never as ground truth.
- **Rejected** — below the reject bar (American: 0.25). The kiosk says it did not catch
  it. It does not guess.
- **Refused before the model runs** — tracking lost both hands for more than half of the
  sign's frames. The kiosk says "I couldn't see your hands" (reason `hands_lost`) and
  nothing is classified.

While this happens the camera badge shows the stage: "I can see you", "signing…" from
the moment the hands rise, "reading your sign…" while the model runs. A banner says when
the picture is too dark, too bright, or shows no body.

Every recognised sign writes one backend log line, `recognition`: language, status,
confidence, the bar used, the top-3 confidences, frames, frames without hands, span and
model time. The words themselves are written only when the visitor agreed to retention;
without that, nothing a visitor signed outlives the session (NFR-S3). No landmarks or
identity either way. To judge a session of your own by the words, tick the retention box.

## 3. Where roughness and errors come from

- **Tracker rate.** About 10 updates a second; markers would hop without easing. It was
  also silently costing words until 2026-10-13: see §15.
- **End-of-sign delay.** A word appears only after the signer pauses (0.5–0.8 s), which is
  itself over the 300 ms the requirements allow for "end of sign to word on screen".
- **Wasted frames.** Fixed 2026-10-13: the browser no longer sends a frame while the last
  one is unanswered.
- **Gaps in the sign.** A frame that loses a hand leaves a hole the model has to read
  through. The holes are now counted and logged, and a sign that is mostly holes is refused.
- **Live domain gap.** Training clips were extracted with MediaPipe 1.0.1; the Jetson runs
  0.10.18, on small compressed frames in a bright room. Not yet measured.
- **Optimistic bars.** American word bars were fitted on validation clips, the same ones
  used to report the gain; the live figure will be lower.

## 4. Plan to make it smoother

Ordered by effect. "Needs Jetson" means it cannot be built or measured without the device.

1. **GPU hand tracking** (TensorRT or MediaPipe's GPU build). Target 25–30 updates/s.
   *Needs Jetson.* Still the largest item: the design and NFR-P5 assume 30.
2. ~~Send a frame only when the last result has come back.~~ **Done 2026-10-13**
   (`assets/app.js`). No queue, no wasted work. It is only correct together with the
   backend sending each result as soon as it exists; on its own it halved the rate of
   hand positions (§15, "a second fault"). Change neither side without running
   `kiosk/live_check.py`.
3. **Binary WebSocket frames instead of base64.** About a third less data. Not done, and
   low priority: decode and hand-off measured at ~3 ms a frame against ~100 ms of tracking.
4. **Predict hand motion between tracker updates.** Marks continue along the hand's velocity
   until the next update. *Best after step 1, tuned on the Jetson.*
5. **Fill one- or two-frame gaps** in a sign by interpolation before the model sees it.
   `core/features.py` already fills gaps of up to 3 resampled frames (`MAX_FILLED_GAP`),
   as in training. What is new (2026-10-13) is that the holes are counted, and a sign that
   is mostly holes is refused.
6. ~~Show "listening" the moment hands rise.~~ **Done 2026-10-13**: "signing…" and
   "reading your sign…" come from the segmenter, not a timer.
7. **Try 320 px frames.** Faster tracking, possibly lower accuracy; measure both with
   `kiosk/replay.py`. *The speed half needs the Jetson.*
8. **Shorten the pause that ends a word** (500 ms still, 800 ms hands gone). This is the
   wait the visitor feels most. Not changed: the values were tuned on synthetic data and
   shortening them risks cutting real signs in two; it needs replayed real signing.

## 5. Known gaps (as of the end of 2026-10-13)

- **Nothing from 2026-10-13 is on the Jetson.** It did not answer ping all day. The
  likely reason is not the Jetson: the development PC was on a different network
  (Wi-Fi `192.168.1.x`, wired port unplugged) from the one the Jetson's address is on
  (`192.168.100.60`). Check `ipconfig` on the PC before suspecting the Jetson. Everything
  in §15 was measured on a PC that imitates the Jetson's tracker rate, not on the Jetson.
  `scripts/deploy-jetson.sh` will send it; only that script's dry run and its
  "cannot reach" path have been run.
- **Not seen in a browser:** the face corner marks, the new camera-badge states, the
  lighting banner and the "couldn't see your hands" message. The backend half of each is
  tested; the drawing and wording are not.
- **The Arabic of the new on-screen messages is mine and unreviewed.** A Deaf reviewer's
  wording replaces it.
- **Two thresholds are loose guesses, set so they rarely fire:** the hands-lost share (0.5)
  and the lighting limits (35 and 230 of 255). Tune them on site from the `recognition`
  and `frame_timing` logs.
- `kiosk/perception.py` patches a private MediaPipe 0.10.x function to survive hands-down
  frames. If a release renames it, the kiosk logs `mediapipe_empty_packet_guard_not_installed`
  and starts unguarded; a hands-down frame could then abort the backend.
- Jordanian (LIU) now reads "coming soon" and cannot be pressed: its only model is trained
  on synthetic data. No visitor can use Jordanian until real recordings exist.
- American: 50 words, about 71% top-1 on validation signers; the per-word bars were fitted
  and scored on the same clips. The 434-word model is research-only licensed.
- The measurements in §15 use two validation signers and ten words. They show the size of
  an effect, not an accuracy figure to quote.

## 6. Open-source projects that could help

Listed from memory on 2026-10-13. Versions and licences have **not** been checked; verify
each before adopting it. Nothing here is a reason to ship generative signing (see
`signoo-avatar`).

**Faster tracking on the Jetson (the ~10 updates/s ceiling)**

- **NVIDIA TensorRT, and ONNX Runtime's TensorRT backend.** The standard way to run models
  on the Jetson GPU: export a hand-pose model to ONNX, run it on the GPU.
- **RTMPose / RTMW (MMPose, via `rtmlib`).** Already tried: finds hands in far more
  motion-blurred frames than MediaPipe (48% vs 12% on the 8 worst clips) but ~6× slower on
  CPU. On the Jetson GPU through TensorRT it may be both faster and better. Best candidate
  to test first. *Needs Jetson.*
- **MediaPipe's own GPU build.** Same models; building it for the Jetson is difficult.

**Better recognition**

- **Google "Isolated Sign Language Recognition" (Kaggle ASL Signs) winning solutions.**
  Landmark-input transformers and augmentation recipes, public. Ideas could lift the ~71%
  top-1 without taking their data. Needs PC only.
- **OpenHands (AI4Bharat).** Pose-based sign recognition toolkit with several model types
  and training code; useful as a comparison.
- **SPOTER, Uni-Sign.** Pose-based recognition research code; Uni-Sign is newer and aimed at
  translation.

**Tools for our pipeline**

- **`pose-format` (sign-language-processing).** A standard format and helpers for sign
  landmarks; could replace some of our ad hoc handling.
- **sign.mt (Amit Moryossef).** Open translation app with a signing avatar. Worth studying
  for flow, but it uses generative signing, which our rules forbid.

**Avatar (B8)**

- **Blender, VRM models, Kalidokit-style retargeting** for playing back *real recorded*
  motion on a character. Read the `signoo-avatar` skill first.

**Data (the real bottleneck)**

- WLASL, MS-ASL, ASL Citizen, PopSign. We already use ASL Citizen and PopSign. MS-ASL and
  ASL Citizen are research-only, so they cannot ship in a commercial product.
- For Arabic and Jordanian: KArSL and ArASL exist but are Saudi and Arabic-alphabet data,
  not Jordanian. No open Jordanian set is known. Recording with consent from Jordanian Deaf
  signers is still the biggest need.

**Suggested order:** (1) test RTMPose/RTMW on the Jetson GPU through TensorRT, (2) borrow the
Kaggle winners' training tricks for accuracy. (1) needs the Jetson; (2) needs only the PC.

## 7. Ideas to make it feel exceptional (first list)

Ideas, not commitments. All stay inside the project rules: no generative signing, every
state visible, landmarks not video by default.

1. **Words appear as the sign is made.** A faint live guess that firms up at the end of the
   sign. It must never be committed before the bar is met.
2. **A real signed reply, played from recordings.** The "answered in sign" part of the
   mission. Needs clips recorded and reviewed by Deaf signers; the American clip library is
   empty today. A people task more than a code task.
3. **Tap to correct, and it learns.** A tapped candidate is already stored as a weak label;
   show the visitor that it helps ("thanks, I'll get this better") alongside the 3-day
   retraining.
4. **Setup in five seconds.** On-screen guidance ("step back", "raise your hands", "good").
   Standing too close is a common cause of bad results.
5. **Next-word suggestions.** After "I want", offer likely words as tap targets.
6. **Fingerspelling.** Letter by letter, for names and unknown words.
7. **Large, high-contrast Arabic and English captions** with text size adjustable by touch.
8. **Operator quality screen.** Staff-only view of confidence, tracker rate and the
   `recognition` log, for tuning on site.
9. **Smooth 30-updates-a-second tracking** (section 4, step 1). Everything above feels better
   on top of it.
10. **Short consent**, in sign video as well as text.

## 8. More ideas (second list)

**Accuracy and reliability**

1. **Handshape, location, movement and palm-direction checks.** When the model is unsure,
   narrow the candidates by these four parts so the top 3 improve.
2. **Adapt to each visitor.** Normalise to their shoulder width and signing speed within the
   first few signs. No new data needed.
3. **Test-time averaging.** Run the model on a few slightly shifted copies of the sign and
   average. Often a few points of accuracy for little cost. Verify on validation signers.
4. **Camera health check.** Detect a dirty lens, backlight, dark room or blocked camera and
   say so on screen. Bad input is a hidden cause of wrong words.
5. **Startup self-test.** Play a stored sample sign through the pipeline at boot and show a
   clear failure state if the result is wrong. Catches broken deployments before a visitor.

**Experience**

6. **Mirror-style preview** with the corner marks, so visitors adjust their own position.
7. **Light-strip cues** around the screen for listening, understood and unsure; visible from
   a distance, never audio-only.
8. **Whole phrases as single signs** ("where is the bathroom"): faster to recognise, more
   useful.
9. **Two-way conversation view.** Staff type or speak, the kiosk shows large text and later
   sign. Turns the kiosk from a translator into a conversation tool.
10. **Idle attract mode.** A short signed greeting so a Deaf passer-by knows it is for them.

**Operations**

11. **Fleet dashboard** in the console: which kiosks are online, tracker rate, errors, model
    version. The server already receives the reports.
12. **Automatic alerts** when a kiosk goes offline, overheats, or accuracy falls suddenly.
    The Jetson dropping off the network would have been caught this way.
13. **A visible "offline, still working" state.** The kiosk already works without internet
    and syncs later; the screen should say so.
14. **Privacy report for visitors.** One tap shows exactly what was kept (landmarks only, no
    video) and how to delete it.

**Suggested start:** camera health check (8.4) and startup self-test (8.5), which are cheap and
would have caught most of what went wrong in the first live runs; then 8.2 and 8.3 for
accuracy, since neither needs new data.

**Status, 2026-10-13.** 8.4 is partly built: too dark and too bright are detected from
the frame's mean brightness and shown; a dirty lens, backlight and a blocked camera are
not. 8.5 is partly built: every model takes a test pass at start-up and `/health` shows
whether it ran, but it is not a stored sample sign checked for the right answer, and a
failed pass does not take the language off the menu. 8.13 is not built. The rest of
sections 7 and 8 are untouched, and several are outside v1's scope as the requirements
draw it (fingerspelling, voice input, facial grammar; `REQUIREMENTS.md` §1.2).

## 9. Local AI that could help (runs on the Jetson, no cloud)

From memory, 2026-10-13; licences and Jetson support unchecked. The Orin Nano has 4 or 8 GB
shared between CPU and GPU, so every model here must be small and its memory use measured.

- **Speech to text** for the staff side of a two-way conversation: Whisper tiny or small
  (`faster-whisper` or `whisper.cpp`), Arabic and English, offline.
- **Text to speech** so staff can hear a visitor's sentence: Piper TTS, offline, Arabic and
  English voices. The kiosk already has a TTS call; check which engine it uses.
- **Arabic ⇄ English translation** for captions: NLLB-200 distilled or Opus-MT. For single
  words our own glossary is safer than a translation model.
- **Face and lighting checks** (YuNet, BlazeFace) for the camera health check (section 8.4).
- **TensorRT pose and hand models** (RTMPose / RTMW), the GPU tracking path (section 4.1).
- **Simple statistical alerts** on tracker rate, temperature and confidence.
- **Not recommended:** several large models at once. They would starve the tracker of memory
  and make the 10-updates-a-second problem worse.

## 10. Local AI that understands signs, trained on our data

What works:

- **Our current landmark model.** Small, fast, retrained every 3 days. Its limits are data
  and tracking quality, not the approach.
- **A stronger landmark model.** The Kaggle ASL Signs winners used transformers on landmarks
  and reached high accuracy on 250 signs. Their recipes can be trained on our data and still
  run on the Jetson. Best next accuracy step; PC only.
- **Fine-tuning a pretrained sign model** (OpenHands, SignBERT-style, Uni-Sign): a better
  starting point than training from scratch, but the weights' licences must be checked and
  some are research-only.
- **Video models** (VideoMAE, I3D): read raw video, so heavier and more privacy-sensitive.
  Too big for the Jetson's memory and speed today.

What does not work:

- **A general chatbot or vision-language model** (LLaVA, Qwen-VL, etc.) watching the camera.
  It was not trained on sign language and answers confidently and often wrongly. A wrong
  answer is worse than no answer.
- **Using one to label our data.** It would add wrong labels. Labels come from Deaf signers
  or reviewers.

Where an LLM does help: explaining *to us* why a sign was confused, reading the
`recognition` log, drafting Arabic glosses for review. A helper for people, not a recogniser.

How our data fits in: (1) collect signs with consent, as landmarks; (2) a Deaf reviewer checks
labels; (3) fine-tune a pretrained sign model, split by signer, test on unseen signers;
(4) the gate decides whether it ships. More varied signers help every option more than the
choice of model does.

## 11. A language model *after* recognition

Agreed as the safe place for one: it works on words we already recognised, never on the
camera. In `kiosk/`, after the Assembler and before the captions.

What it would do:

- Turn word strings into natural sentences in English and Arabic ("I want water please"),
  since sign grammar differs from spoken grammar.
- Use context to choose between close candidates ("bread" or "red" after "I want").
- Suggest next words as tap targets.
- Help staff draft a short written reply.

Guardrails (it can sound right while being wrong):

1. It may only rearrange and rephrase the words we recognised. It never adds a fact or word
   the visitor did not sign.
2. The raw recognised words always stay on screen next to the fluent sentence, so a Deaf
   visitor can see what the kiosk actually understood.
3. Context only re-ranks candidates. The final choice still goes to the visitor's tap or the
   confidence bars; the LLM never silently replaces a word.
4. If the model fails or is slow, the plain word string appears as it does today.
5. A fixed test set of validation-signer sentences shows how often it helps and how often it
   makes things worse.
6. Nothing it produces is stored as a training label.

Cost: a 1–3B quantised model fits in memory but shares RAM with the tracker and the sign
model, so it might slow tracking; measure first. It runs once per sentence, not per frame. A
plain word-frequency list gives next-word suggestions with no LLM at all and is the cheaper
first step.

Proposed build order: (1) next-word suggestions and context re-ranking, since they improve
accuracy; (2) the fluent sentence with the raw words shown beside it. Prototype on the PC
with a small local model and test on validation sentences before it goes near the Jetson.

## 12. Making the process correct and logical

1. **Measure on the real setup.** Record real sessions from the Jetson camera, with the correct
   words, and replay them as the true test. Today we judge by training clips that look
   nothing like the kiosk; this is the biggest gap. Report final numbers on unseen signers,
   once, on the final model. The word bars were tuned on validation, so they need a fresh check.
2. **Get the sign logic right.** Test the segmenter on real continuous signing (a pause inside
   a sign, or two signs run together, gives wrong words). Sign grammar differs from spoken
   word order and facial expression changes meaning (questions, negation); the face markers
   are a start, real expression recognition is later work. Check that a repeated sign counts
   once and a held sign is not cut in two.
3. **Make decisions explainable.** The `recognition` log says what was committed or refused.
   Add sign length, tracker quality (frames that lost a hand) and lighting so a wrong word can
   be traced to a cause. Keep every threshold in one place, with a note of how it was chosen.
4. **Protect against bad inputs.** If too many frames lost a hand, say "I couldn't see your
   hands" instead of guessing. With several people in view, follow the closest or active signer.
5. **Keep the process honest.** One test per `CLAUDE.md` rule (signer split, untrained language
   says "coming soon", gate cannot be bypassed); a check that fails while `liu` is "live" on
   synthetic data. Audit docs against code. Make every recognition traceable to the model,
   vocabulary, thresholds and MediaPipe version that produced it.
6. **Close the loop with people.** A Deaf reviewer for labels, Arabic glosses and
   understandability; a short consented pilot with a way to report a bad result.

Start with: the replay test set, the "couldn't see your hands" rejection, and the `liu` status check.

**Status, 2026-10-13.** Done: the "couldn't see your hands" rejection (12.4, first half);
the `liu` status check and the rule that every card starts disabled (12.5); sign length,
holes and model time in the `recognition` log (12.3, not lighting per sign). The replay
tool already existed (`kiosk/replay.py`, NFR-N1) and gained `--tracker-fps`; what is
still missing from 12.1 is *recordings from the Jetson's own camera*, which need the
Jetson and consenting signers. Not done: following one signer among several people
(12.4), a recorded version stamp on each recognition (12.5), everything in 12.2 and 12.6.

## 13. Making the project better, faster and easier

**Faster to build**

1. One-command Jetson deploy (`scripts/deploy-jetson.sh`): check reachability, sync changed
   files, restart the service, show the health check.
2. A quick test set (seconds) for every change, and the full heavy set before a release.
3. A camera simulator that feeds saved sign clips into the kiosk, so the whole pipeline can be
   tested on the PC without anyone signing and without the Jetson.
4. Continuous integration running the quick tests on every push.
5. Small, regular commits instead of a large uncommitted pile.

**Easier to run**

6. A self-installing kiosk setup: dependencies, services, model, camera check, first-run test.
7. A health page: tracker rate, temperature, model version, last error, online or not.
8. Automatic restart and recovery, with a visible out-of-service screen instead of a blank one.
9. Short runbooks ("if X happens, do Y") for the person at the kiosk.
10. Pin dependency versions so the Jetson and the PC behave the same (MediaPipe 0.10.18 vs
    1.0.1 is exactly this problem).

**Better results**

11. Record real data on purpose: short consented sessions with Deaf signers on the real kiosk camera.
12. Pick the 50–100 words visitors actually need first (greetings, directions, help, emergency,
    time, numbers) and make them excellent, rather than 434 words at about 60%.
13. A scoreboard of replay-set accuracy, tracker rate and latency across versions.

Start with: items 1, 3 and 7 for speed and ease; 11 and 12 for results.

**Status, 2026-10-13.** Item 1 is written (`scripts/deploy-jetson.sh`) but has only been
run as a dry run and against an unreachable Jetson. Item 3 exists as `kiosk/replay.py`,
which now also imitates a slow tracker. Item 7 is half there: `/health` carries tracker
rate, stage timings, memory, temperature, outcome counts and each model's start-up pass,
but there is no page that shows it and the heartbeat does not send it. Item 2 already
existed (`scripts/test-all.sh fast`). The others are not started.

## 14. Error handling and cleanup (done 2026-10-13)

Audit of `kiosk/` for swallowed errors and leaked resources found two real problems, both fixed:

- **Speech (`kiosk/main.py`, `_handle_speak`).** A failure inside the text-to-speech call was
  not caught, so the screen could stay in "speaking" with no end message. It is now logged
  (`tts_failed`) and reported as `tts_done` with `ok: false`. The background task was also
  not tracked, and the event loop holds only a weak reference to a task, so it could be
  collected before finishing. It is now kept in `_background_tasks` until done.
- **Landmarker (`kiosk/perception.py`, `HolisticPerception.close`).** Closing could raise
  inside the cleanup path and skip the cleanup after it. `close()` is now idempotent and never
  raises; a failure is logged (`landmarker_close_failed`).

Reviewed and left as they are: the sync loops (one language's failure must not stop the
others), the per-message handler in the WebSocket loop (a bad message costs the sender an
error, not the visitor the session), and the socket-close paths, which already log and move on.

Found later the same day, and the most serious of the three:

- **A tracker left unclosed could hang the whole backend.** A MediaPipe landmarker that
  is freed by Python's garbage collector, instead of being closed, deadlocks inside
  MediaPipe (reproduced in isolation: freed by reference count is fine, freed by the cycle
  collector never returns). It showed up as a test run that hung for eight minutes. The
  tracker class no longer forms a reference cycle, every path closes it, and one that is
  ever dropped anyway is closed on a thread of its own (`kiosk/perception.py`,
  `kiosk/tests/test_perception_lifecycle.py`).

## 15. Why words went missing, and what was measured (2026-10-13)

**The fault.** The segmenter accepted a word only if it had at least 6 tracked frames. A
frame count is a different length of time at every tracker speed: 0.2 s at 30 tracked
frames a second, 0.6 s at the Jetson's 10, a full second at 6. In the test clips a sign
keeps the hands in view for about 0.6 s. So on a slow tracker an ordinary sign was thrown
away, the badge went back to "I can see you", and nothing told the visitor. The model
never saw those signs, so no threshold could have fixed it.

**How it was found.** Not by a unit test: every one passed. It showed when the real
backend was driven over its WebSocket with recorded clips, paced the way the browser
paces them, on a PC busy enough to track at 6–7 frames a second: six clean clips, six
times no word. `kiosk/replay.py --tracker-fps N` now reproduces it on purpose.

**The fix.** A word is at least 3 frames spanning at least 150 ms, and is closed after
4.5 s. At 30 frames a second that is the same as before (6 frames span 167 ms); below
that it no longer depends on the tracker. The old ceiling of 45 frames had the mirror
fault (1.5 s at 30 a second) and became a time too.

**The measurement.** 60 clips: validation signers 7002 and 8012, ten words (apple, book,
cat, dad, dog, hello, milk, mom, red, yes), three clips each, replayed through the kiosk's
own tracker, segmenter and American model. "Right word reachable" means it was committed
correctly or was among the candidates offered. Each row is one run.

| Tracked frames/s | Rule | Committed right | Committed wrong | Right word offered | Offered, right word absent | Refused | **No word** | **Right word reachable** |
|---|---|---|---|---|---|---|---|---|
| 6 | old (≥6 frames) | 3 | 1 | 4 | 1 | 0 | **51** | 7 of 60 (12%) |
| 6 | new (≥3 frames, ≥150 ms) | 29 | 1 | 22 | 4 | 0 | **4** | 51 of 60 (85%) |
| 10 | old | 29 | 1 | 17 | 2 | 0 | **11** | 46 of 60 (77%) |
| 10 | new | 35 | 0 | 21 | 3 | 0 | **1** | 56 of 60 (93%) |
| 30 (every frame) | old | 38 | 1 | 17 | 3 | 1 | 0 | 55 of 60 (92%) |
| 30 (every frame) | new | 33 | 1 | 21 | 5 | 0 | 0 | 54 of 60 (90%) |
| 30 (every frame) | new, repeated | 34 | 1 | 20 | 5 | 0 | 0 | 54 of 60 (90%) |

What this does and does not show:

- **At 6 and 10 tracked frames a second the effect is far larger than any noise**: 51 and
  11 signs lost outright under the old rule, 4 and 1 under the new.
- **At full rate the two rules cannot be told apart.** The two runs with identical
  settings (last two rows) agree in total but gave a different outcome for 14 of the 60
  clips: the tracker does not see exactly the same frames twice, and a word whose
  confidence sits near a bar flips between "committed" and "offered". The old-against-new
  difference at full rate is inside that. It also says the model is fragile near its
  bars, which is the model's 71% top-1 showing, not the pipeline.
- **It is not an accuracy figure.** Two signers and ten words, chosen before looking at
  results but not at random, on validation signers that the per-word bars were fitted on.
  The test signers were not touched.
- **It is not the Jetson.** A PC imitated the Jetson's tracker rate. MediaPipe's version
  there (0.10.18 against 1.0.1 here), its camera, its lighting and its heat are all
  untested. The first thing to do when the Jetson answers is run this same command on it.
- One confident mistake recurs: "yes" committed as "apple" at 0.83–0.88 ("mom" as "blue"
  at 0.86 in one run). "yes" has a per-word bar of 1.0, but the bar that applies is the one
  on the word the model *picked*: "apple"'s is 0.61, below the language's 0.80, so 0.80
  applies and 0.83 clears it. A per-word bar guards the word it is on, not the word a
  sign is mistaken for.

**A second fault, found the same way and introduced the same day.** Tracking is
asynchronous: a frame goes in and its result comes out about 100 ms later. The backend
collected a result only when the *next* frame arrived. That had always cost a frame of
delay on every hand position. Then the browser was changed to stop sending while a frame
was unanswered (§4.2), and the two rules waited on each other: no result until another
frame, no other frame until a result, until the browser's 200 ms timeout. Frames then
arrived in pairs, two results finished between collections, and the older was overwritten
unread. The replay tool cannot see this (it drives the tracker directly); only the run
over the real WebSocket did. The backend now handles each result the moment the tracker
finishes, on its own task (`KioskConnection._ensure_result_consumer`).

The same six clips (one signer, validation) over the real WebSocket, on a PC busy with
another job, each row one run:

| | Frames sent/s | Results returned/s | Hand positions per sign | Clips with any outcome | Right word committed or offered |
|---|---|---|---|---|---|
| old word rule, results collected on next frame | 8.7–9.8 | 5.9–7.6 | not counted | 0 of 6 | 0 of 6 |
| new word rule, results collected on next frame | 8.3–9.4 | 5.5–7.2 | not counted | 2 of 6 | 2 of 6 |
| the same, run again | 9.1–13.1 | 6.6–12.4 | 2–3 (one clip 10) | 5 of 6 | 2 of 6 |
| new word rule, results pushed when ready | 8.9–9.8 | 8.7–9.6 | 4–5 | 6 of 6 | 4 of 6 |

Six clips is a demonstration, not a measurement of accuracy: the point is the second and
third columns coming together (results returned catching up with frames sent), and hand
positions per sign roughly doubling. "Any outcome" counts a refusal too; the last column
counts only a correct commit or the right word among the candidates.

That throwaway script became a tool, `kiosk/live_check.py`. Its first run with everything
above in place, 20 clips (the same two signers and ten words, one clip each) over the real
WebSocket on the same busy PC: 13 committed correctly, 1 committed wrongly ("cat" as "mom"
at 0.80), 5 with the right word offered, 1 offered without it, none left with nothing;
the right word committed or offered for **18 of 20**; and every frame sent got its result
back (7.3–9.6 a second each way). One run, and the same caveats as the table above it.

This path, not the replay tool, is the one to trust for "does the kiosk work". It is not
in the test suite as a real-tracker test, because it needs the video data and minutes to
run; the suite has a fake tracker that answers late
(`test_a_tracking_result_reaches_the_screen_without_waiting_for_the_next_frame`). Pointed
at the Jetson with `--url`, the same command puts known clips through the device itself:
its tracker, its MediaPipe, its CPU. **That is the first thing to run when the Jetson
answers again.**

Reproduce a replay row with:

```bash
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -m kiosk.replay data/popsign/raw --language asl \
    --signers 7002 8012 --signs apple book dog hello milk yes cat red mom dad --tracker-fps 10
# the old rule, for comparison:
SIGNOO_MIN_UTTERANCE_FRAMES=6 SIGNOO_MIN_UTTERANCE_MS=0 \
SIGNOO_MAX_UTTERANCE_MS=0 SIGNOO_MAX_UTTERANCE_FRAMES=45  <same command>
```

## 16. On "caching" and feeling fast

What makes the kiosk feel fast is not a cache of results (every sign is different) but
having the slow things done before they are needed. That is what was built on
2026-10-13: models warmed at start-up, the tracker started at the language screen, frames
paced to the tracker, status messages sent only when they change, and a visible state for
every stage so a wait never looks like a freeze. The frontend is still served with
`Cache-Control: no-cache` on purpose, so a stale `app.js` is never reused; the files are
small and local, and the browser revalidates them in milliseconds.

What would make it *actually* faster, in order: tracking on the Jetson's GPU (§4.1), then
a shorter end-of-word pause set from real signing (§4.8). Neither can be done without the
Jetson and recordings from it.

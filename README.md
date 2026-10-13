# SIGNOO — مترجم لغة الإشارة الأردنية

**SIGNOO** reads Jordanian Sign Language (لغة الإشارة الأردنية · LIU) from a camera
and writes what it sees in **Arabic and English** — built toward a walk-up public
kiosk, expanding to more sign languages over time.

No video ever leaves the device: only landmark coordinates are processed, and the
camera stream itself is never recorded, uploaded, or stored.

**This file is being rewritten in stages** as the project moves from a single-signer
browser demo to the full kiosk system — `CLAUDE.md` at the repo root is the current,
accurate map of what exists; treat anything below that conflicts with it as this
file lagging behind, not the other way around.

---

## ما هو هذا الآن · What this actually is today

Three things share this repository:

1. **The kiosk** (`kiosk/`, `electron/`, `index.html`/`assets/*.js`) — a FastAPI +
   WebSocket backend and a browser frontend, packaged as an installed app via
   Electron. This is the real, current system: presence auto-detect → greet → pick a
   sign language (one live today, eight more staged as "coming soon") → sign →
   see it written in both languages live → ask a question → get an answer, signed
   back. See `kiosk/README.md` to run it, `docs/SYSTEM_DESIGN.md` for how it works.
2. **The learning loop** (`server/`, `console/`) — the backend and admin UI that
   review corrected signs, retrain, and gate a new model before it ever reaches a
   kiosk. See `server/README.md`, `console/README.md`.
3. **The training pipeline** (`ml/`, `core/`) — turns recorded signer data into the
   model the kiosk actually runs. See "Training a real LIU model" below.

There is **no pre-trained model for Jordanian Sign Language** — public sign-language
models are almost all American Sign Language — so none of this ships knowing LIU; it
has to be taught from real recorded signers (`docs/DATA_COLLECTION.md`), which is the
project's actual bottleneck right now, not engineering throughput
(`docs/ROADMAP.md`).

## التشغيل · Running the kiosk

The frontend (`index.html`) is no longer a dependency-free static page — it talks to
`kiosk/main.py`'s backend over a WebSocket, so a bare `python3 -m http.server` will
not work (no `/ws/session`, no `/core/vocab.json`). Run the real backend instead:

```bash
# one-time: build the venv kiosk/ expects (see kiosk/README.md for full detail)
/path/to/python3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r kiosk/requirements.txt

.venv/Scripts/python.exe -m uvicorn kiosk.main:app --port 8000
```

Then open **http://localhost:8000** and allow camera access — or run it as an
installed app instead of a browser tab via `electron/` (`cd electron && npm install
&& npm start`; spawns the backend above itself, see `electron/README.md`).

Needs a recognition checkpoint at `models/signoo.pt`; `kiosk/README.md` has the
one-line synthetic-checkpoint command if you don't have a real one yet (no accuracy
claim is possible on synthetic data — it only proves the pipeline runs).

## الاستخدام · Using the kiosk

There is no "teach your own signs" tab anymore — that was the old single-signer
browser prototype, and the live kiosk flow replaced it:

1. **Hello** — the kiosk notices you (camera-based presence detection) or you tap
   the screen.
2. **Choose a language** — one sign-language card per card, not a caption toggle.
   Only a card with a real trained model is live; the rest say "recording in
   progress," honestly, rather than pretending to work.
3. **Sign** — a self-view camera box, with your words appearing live in your chosen
   language and in English underneath.
4. **Ask** — press Next, and the kiosk answers your question, signed back through an
   on-screen character and written out at the same time.
5. **Thank you** — then it resets, clean, for the next visitor.

Teaching the system a *new* sign is no longer a kiosk-visitor action — it's the
dataset-collection workflow below, done by (or with) someone who actually knows the
language, which is what makes the signs real instead of personal.

## تدريب نموذج خاص · Training a real LIU model

The kiosk above only recognises what a real trained model gives it — there's no
per-visitor teaching step anymore. Building and improving that model (for **any**
signer, not one) is this pipeline, in `ml/`:

```bash
pip install -r requirements.txt

python ml/extract_landmarks.py data/raw --out data/processed   # video → landmarks
python ml/clean.py data/processed/manifest.json                # quality checks
python ml/train.py data/processed/manifest.json --split data/split.json
python ml/evaluate.py models/signoo.pt data/processed/manifest.json --split data/split.json
```

### Collecting the data

`record.html` is a recording studio for collection sessions. Serve the folder and open
**http://localhost:8000/record.html**.

It walks a signer through the word list, records fixed-length clips on the space bar,
tracks progress, and exports the whole session as a ZIP in exactly the layout
`ml/extract_landmarks.py` expects:

```
S07/salam/S07_salam_r00.webm
S07/salam/S07_salam_r00.json     ← signer, sign, repetition, lighting, distance, device
S07/session.json
```

Unzip it into `data/raw/` and run the extractor — no manual filing. The studio will not
start a session until consent is confirmed, and rejects signer IDs that look like real
names, because these recordings are identifiable biometric data.

Two documents carry the real content:

- **[docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)** — how to collect a dataset
  that is worth training on: signer diversity, consent, recording protocol, splits.
- **[docs/TRAINING.md](docs/TRAINING.md)** — the model, the features, and how to
  evaluate honestly.

The one rule that matters most: **split by signer, never randomly.** On this pipeline
a random split scored 99.2% where the honest signer-independent split scored 52.9% —
same model, same data. `ml/compare_splits.py` measures that gap on your own data.

## بنية المشروع · Project layout

The kiosk, server, and console each have their own more detailed README
(`kiosk/README.md`, `server/README.md`, `console/README.md`, `electron/README.md`);
this table covers what lives at the repo root and in the shared/training pieces.

| File | Purpose |
|---|---|
| `index.html` | The kiosk's browser frontend (Arabic, RTL) — the six-screen visitor flow, served by `kiosk/main.py` |
| `assets/app.js` | WebSocket client driving the kiosk flow: camera, screens, presence auto-detect |
| `assets/vision.js` | Canvas drawing only: corner marks round the hands and face that glide between tracker updates (the older skeleton overlay is still in the file, switched off). Perception itself runs server-side in `kiosk/perception.py` |
| `assets/kiosk.css` | Kiosk visual design |
| `assets/vocab.js` | Kept only for `record.html`; the kiosk frontend fetches `core/vocab.json` directly |
| `record.html` · `assets/record.js` | Dataset recording studio (unchanged by the kiosk rebuild) |
| `assets/zip.js` | Minimal ZIP writer for session export |
| `assets/styles.css` | `record.html`'s styling only — the kiosk uses `assets/kiosk.css` |
| `core/features.py` | Landmark features: body-centred, scale- and speed-invariant. Shared by `ml/` and `kiosk/` so training and inference never diverge |
| `core/contracts.py` | Data types passed between kiosk blocks (B1–B8) |
| `core/vocab.json` | The one word list shared between the kiosk frontend, `kiosk/assembly.py`, and the console's KB editor |
| `ml/model.py` | Transformer encoder classifier (~1.3 M parameters) |
| `ml/data.py` | Dataset, augmentation, and signer-independent splitting |
| `ml/clean.py` | Dataset quality checks |
| `ml/train.py` · `ml/evaluate.py` | Training and honest evaluation |
| `ml/compare_splits.py` | Measures how much a random split inflates accuracy |
| `ml/make_synthetic.py` | Synthetic data for testing the pipeline — proves wiring, not accuracy |

### How recognition actually works now

B2 Perception (MediaPipe Holistic, server-side) turns each frame into pose + two-hand
landmarks. Each frame becomes a fixed-size feature vector: for each of two hand slots,
whether a hand is present, where it is, and its 21 landmarks measured relative to the
wrist and divided by hand size — so the *shape* is the same near or far from the
camera (`core/features.py`, shared bit-for-bit with training).

A segmented utterance is resampled to a fixed length, centred on the signer's own
hands and measured in hand-size units, so distance from the camera doesn't change the
result while movement *within* the clip is preserved.

Classification is a trained Transformer encoder (`ml/model.py`, `models/signoo.pt`) —
not nearest-neighbour against a personal recording set, which was the old single-signer
prototype's approach. Confidence below a rejection threshold reports nothing at all
rather than guessing (`docs/SYSTEM_DESIGN.md` B4); between the reject and confident
thresholds, the kiosk offers the top few candidates as touch targets instead of
silently picking one (`deaf-kiosk-ux` skill, FR-K8).

## الحدود · Limitations

Worth being honest about:

- **Hands and body, not face.** LIU, like every sign language, also carries meaning
  in facial expression, which this stack cannot see yet — a planned addition
  (`docs/SYSTEM_DESIGN.md` B2), not an oversight.
- **Word by word.** Output is a sequence of recognised words, not grammatical Arabic.
  LIU grammar differs from written Arabic, so the output reads as glosses — shown to
  the visitor as-is, not silently "corrected" into fluent Arabic.
- **No real data yet.** The model behind the live kiosk today is trained on
  synthetic data only — real signer recruitment is the project's actual bottleneck
  (`docs/ROADMAP.md`), not a finished, deployable accuracy number.
- **One language live.** Nine sign-language cards exist in the menu; eight say
  "recording in progress" honestly rather than running an untrained model
  (`docs/ROADMAP.md`, `signoo-language-scaling` skill).

## الخصوصية · Privacy

The camera stream is processed frame by frame and never recorded, uploaded, or sent
anywhere outside the device running the kiosk. What's captured for training (only
with opt-in consent) is landmark coordinates — no images, no video
(`kiosk/data_capture.py`). Declining retention still gets you the full translation
experience; a refusal that degraded service would not be a real choice
(`deaf-kiosk-ux` skill).

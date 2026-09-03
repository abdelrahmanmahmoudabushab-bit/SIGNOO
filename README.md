# SIGNOO — مترجم لغة الإشارة الأردنية

**SIGNOO** reads Jordanian Sign Language (لغة الإشارة الأردنية · LIU) from a webcam
and writes what it sees in **Arabic and English**.

Everything runs in the browser, on your own device. No server, no account, and no
video ever leaves your computer.

---

## كيف يعمل · How it works

There is **no pre-trained model for Jordanian Sign Language** — public sign-language
models are almost all American Sign Language. So SIGNOO does not ship knowing LIU.
It learns it from you:

1. **Hand tracking** — Google's MediaPipe finds 21 landmarks on each hand, up to two
   hands, ~30 times a second. This part is ready-made and needs no training.
2. **You teach the signs** — record each sign a few times in the *تعليم الإشارات* tab.
   Only the landmark numbers are stored; the video is discarded immediately.
3. **Recognition** — a live clip is matched against your recordings, and the matching
   word is written in Arabic and English, and optionally spoken aloud.

This means the app is only as good as the signs recorded into it — but it also means
it can be taught the real LIU signs by someone who actually knows them, which is
exactly what a generic model cannot do.

## التشغيل · Running it

No build step and no dependencies to install. Browsers block camera access on
`file://`, so serve the folder over `http://`:

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000** and allow camera access.

Needs a recent Chrome, Edge, Firefox or Safari. First load fetches the MediaPipe
model (~8 MB) and needs an internet connection; after that it is cached.

## الاستخدام · Using it

**Teach a sign** — *تعليم الإشارات* tab → pick a word → **تسجيل عيّنة** → sign it after
the countdown. Record **at least 3 times per word**, ideally 5–8. Vary your speed and
distance a little between takes; that variation is what makes recognition robust.
Add words that aren't in the starter list with *إضافة كلمة جديدة*.

**Translate** — *الترجمة* tab → **بدء الترجمة** → sign. Recognised words build up a
sentence in both languages. A word is only written once the match has held steady for
a few frames, so passing hand movement doesn't produce text.

At least **two words** must be trained before translating — recognition works by
comparison, so there has to be something to compare against.

**Tuning** (*إعدادات الحساسية*) — raise the confidence threshold if wrong words appear;
lower it if correct signs are being missed. Raise stability to reduce false triggers,
lower it for faster response.

**Back up your work** — *البيانات* tab → **تصدير إلى ملف** saves everything to a JSON
file. Recordings live in the browser's storage, so clearing site data erases them.
Export before you have much to lose, and to move a trained set to another device.

## تدريب نموذج خاص · Training a real LIU model

The browser app above learns from one signer. Building a model that works for **any**
signer is a separate, larger project, and the pipeline for it lives in `ml/`:

```bash
pip install -r requirements.txt

python ml/extract_landmarks.py data/raw --out data/processed   # video → landmarks
python ml/clean.py data/processed/manifest.json                # quality checks
python ml/train.py data/processed/manifest.json --split data/split.json
python ml/evaluate.py models/signoo.pt data/processed/manifest.json --split data/split.json
```

Two documents carry the real content:

- **[docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)** — how to collect a dataset
  that is worth training on: signer diversity, consent, recording protocol, splits.
- **[docs/TRAINING.md](docs/TRAINING.md)** — the model, the features, and how to
  evaluate honestly.

The one rule that matters most: **split by signer, never randomly.** On this pipeline
a random split scored 99.2% where the honest signer-independent split scored 52.9% —
same model, same data. `ml/compare_splits.py` measures that gap on your own data.

## بنية المشروع · Project layout

| File | Purpose |
|---|---|
| `index.html` | Interface (Arabic, RTL) |
| `assets/vision.js` | MediaPipe hand tracking and skeleton drawing |
| `assets/recognizer.js` | Feature extraction and the nearest-neighbour classifier |
| `assets/store.js` | IndexedDB persistence, import/export |
| `assets/vocab.js` | Starter word list (Arabic ↔ English labels) |
| `assets/app.js` | Camera loop, recording, live translation, UI |
| `assets/styles.css` | Styling |
| `ml/features.py` | Landmark features: body-centred, scale- and speed-invariant |
| `ml/model.py` | Transformer encoder classifier (~1.3 M parameters) |
| `ml/data.py` | Dataset, augmentation, and signer-independent splitting |
| `ml/clean.py` | Dataset quality checks |
| `ml/train.py` · `ml/evaluate.py` | Training and honest evaluation |
| `ml/compare_splits.py` | Measures how much a random split inflates accuracy |
| `ml/make_synthetic.py` | Synthetic data for testing the pipeline |

### How recognition works, in more detail

Each frame becomes 134 numbers: for each of two hand slots, whether a hand is present,
where it is, and its 21 landmarks measured relative to the wrist and divided by hand
size — so the *shape* is the same near or far from the camera.

A clip is resampled to a fixed 12 frames, which makes a sign look the same whether it
was made quickly or slowly. The clip is then centred on the signer's own hands and
measured in hand-size units, so sitting closer, further back or off to one side gives
the same result — while movement *within* the clip is preserved, because both hands
share one origin.

Matching is nearest-neighbour against your recordings. Confidence compares the best
match against the best match from any *other* word, so a sign that clearly resembles
one word scores high and something halfway between two words scores low. Anything
further away than a rejection threshold — derived from how tightly your own repeats
of a word cluster — is reported as nothing at all. That threshold is what keeps the
translator silent while you are just moving your hands.

## الحدود · Limitations

Worth being honest about:

- **Hands only.** LIU, like every sign language, also carries meaning in facial
  expression, head movement and body posture. SIGNOO currently sees none of that, so
  signs distinguished only by expression cannot be told apart.
- **Word by word.** Output is a sequence of recognised words, not grammatical Arabic.
  LIU grammar differs from written Arabic, so the output reads as glosses.
- **Personal, not universal.** It learns the signer who trained it. Another person's
  signing will be recognised less reliably, and regional variation is not handled.
- **Needs clear conditions.** Good lighting, hands in frame, plain background.

Natural next steps: adding face and body landmarks (MediaPipe Holistic) to capture
expression, and a sequence model instead of nearest-neighbour once a larger dataset
has been collected.

## الخصوصية · Privacy

The camera stream is processed frame by frame in the page and never recorded,
uploaded or sent anywhere. Recordings are landmark coordinates only — no images.
All storage is local to your browser.

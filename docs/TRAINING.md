# التدريب والتقييم · Training and Evaluation

How the model is trained, why it is built this way, and how to read the results
honestly.

---

## The pipeline

```
data/raw/          videos + metadata          (docs/DATA_COLLECTION.md)
      │
      ├── ml/extract_landmarks.py    →  data/processed/*.npy + manifest.json
      ├── ml/clean.py                →  quality report
      ├── ml/train.py                →  models/signoo.pt
      └── ml/evaluate.py             →  honest numbers on held-out signers
```

```bash
pip install -r requirements.txt

python ml/extract_landmarks.py data/raw --out data/processed
python ml/clean.py data/processed/manifest.json --report data/report.json
python ml/train.py data/processed/manifest.json --split data/split.json
python ml/evaluate.py models/signoo.pt data/processed/manifest.json --split data/split.json
```

To try the pipeline before any real data exists:

```bash
python ml/make_synthetic.py --out data/synthetic
python ml/train.py data/synthetic/manifest.json --split data/synthetic/split.json
```

---

## Why landmarks instead of video

The model never sees pixels. MediaPipe Holistic reduces each frame to body and hand
landmarks, and the model learns from those.

Video models need far more data than any LIU dataset will have for years, and with a
small dataset they learn the room, the lighting and the signer's clothes. Landmarks
throw all of that away before training starts. The cost is real — everything not in
the landmarks is invisible, most importantly **facial expression**, which is
grammatically meaningful in sign languages. That is a known limitation, not an
oversight; see *What this cannot do* below.

Landmarks also train in minutes on a laptop, which matters more than it sounds: it is
the difference between testing ten ideas a day and one.

## What the features are

`core/features.py`, per frame — shared with the kiosk runtime so training and inference
compute the same numbers:

| Part | Values |
|---|---|
| Upper-body pose (25 landmarks × xyz) | 75 |
| Left hand (21 × xyz) + presence flag | 64 |
| Right hand (21 × xyz) + presence flag | 64 |
| **Total per frame** | **203** |

Every point is expressed **relative to the midpoint between the shoulders, divided by
shoulder width**. So a tall signer and a short one, standing near or far, centred or
off to one side, produce the same numbers. Shoulder width is the unit because it is
stable throughout a clip — hand size is not, since hands move toward and away from
the camera constantly.

Clips are resampled to **64 frames**, which removes signing speed as a variable, and
**velocity is appended** (203 → 406 values per frame). Velocity matters because many
signs share a handshape and differ only in direction of movement; that distinction is
hard to see in positions alone and obvious in differences.

## The model

A 4-layer Transformer encoder, ~1.3 M parameters (`ml/model.py`).

Self-attention fits signs well: the informative moment may be at the start, the middle
or the end of a clip, and attention lets every frame consult every other one directly.

It is small on purpose. With a few thousand clips a bigger model memorises signers.
**If held-out-signer accuracy is much worse than held-out-repetition accuracy, the fix
is more signers, not more parameters.** That is the most common wrong turn in this
field.

Training details and why: label smoothing 0.1 (some labels will be wrong, and sign
boundaries are genuinely fuzzy — training for certainty trains for confident errors);
gradient clipping (small-dataset Transformers spike); cosine schedule; early stopping
on validation accuracy.

## Augmentation

Applied to training clips only (`ml/data.py`): speed change 0.8–1.25×, landmark
jitter, whole-body shift, occasional frame dropout imitating tracking failures.

Deliberately **not** applied:

- **Vertical flips** — change meaning.
- **Large rotations** — orientation is part of a sign.
- **Time reversal** — reverses meaning in directional signs (*give* vs *take*).
- **Mirroring, by default** — available via `--mirror-prob`, off by default. Mirroring
  usually preserves a sign's identity, since handedness is mostly a property of the
  signer. But some signs are genuinely asymmetric, and in a two-handed sign the
  dominant and non-dominant hands have different roles. Enable it only after checking
  it does not hurt validation accuracy, and never as a substitute for recording
  actual left-handed signers.

## Evaluation — the part that decides whether any of this is real

**Split by signer, never randomly.** `ml/data.py` enforces it and refuses overlapping
splits.

A random split puts the same person in train and test. The model recognises *them*,
not the sign, and accuracy typically lands 20–40 points too high. This is the single
most common reason a sign-recognition result cannot be reproduced on new people.

`ml/evaluate.py` reports:

- **top-1** and **top-5** accuracy
- **per-signer accuracy** — the important one. A model averaging 85% but scoring 40%
  on one signer is not ready, and the average hides it. The script warns when the
  spread exceeds 15 points.
- **per-sign accuracy** — finds signs that need more or better data.
- **confusion pairs** — worth watching by eye. Confused pairs often differ only in
  facial expression, which this model cannot see.

Run it on the test signers **once**, at the end. Tuning against the test set turns it
into a validation set and the number stops meaning anything.

### Measured on this pipeline

`ml/compare_splits.py` trains the same model twice on the same clips, changing only
how the test set is chosen. On a synthetic dataset built so that signer identity
competes with sign identity — the situation real data is always in:

```
signer-independent accuracy : 52.9%   ← the real number
random-split accuracy       : 99.2%
inflation                   : +46.2%
```

Same model, same data, same training. The extra 46 points are entirely the model
recognising people it had already seen. Run this once on your own dataset; the gap is
the amount of self-deception available to the project, measured in accuracy points.

### Expect lower numbers than published papers

Many reported ArSL results use random splits, single-signer datasets, or both. A
signer-independent 75% is a stronger result than a random-split 98%, and only the
first one predicts what happens when a stranger signs at your camera.

## Using existing Arabic datasets

LIU data will be scarce for a while. Larger Arabic datasets are useful for
**pre-training**, then fine-tuning on LIU:

- **[KArSL](https://hamzah-luqman.github.io/KArSL/)** — 502 signs, 3 signers, 50
  repetitions each, 75,300 samples, RGB + depth + skeleton. The largest word-level
  Arabic set. Also [on Hugging Face](https://huggingface.co/datasets/FatimahEmadEldin/karsl-502-arabic-sign-language-v2).
  Note the 3 signers: it is large in clips, small in people.
- **[ArASL / ArSL2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6661066/)** —
  54,049 images, 32 alphabet signs, 40 participants. Static letters only.

Both are **Saudi**, not Jordanian. Pre-train on them for general signing structure,
then fine-tune on LIU — and always report your final numbers on LIU test signers.
Check each dataset's licence before use or redistribution.

## What this cannot do

- **No face.** Facial expression carries grammar in sign languages — question marking,
  negation, intensity. Signs differing only in expression cannot be separated. Adding
  face landmarks is the highest-value next step.
- **Isolated signs only.** It classifies one pre-segmented sign at a time, not
  continuous signing. Continuous recognition needs segmentation and a CTC or
  sequence-to-sequence model.
- **Glosses, not Arabic.** Output is a sequence of sign labels. LIU grammar differs
  from Modern Standard Arabic; turning glosses into fluent Arabic is a separate
  translation problem.

## Roadmap

1. **20–30 signs, 15+ signers.** Prove the pipeline signer-independently.
2. **Add face landmarks.** Recover expression.
3. **Grow to 100–300 signs.** Keep adding signers, not just repetitions.
4. **Export to the browser** via ONNX, so the app runs the trained model.
5. **Continuous signing.** Segmentation, then sequence-to-sequence.
6. **Release the dataset** — with consent — as the first public LIU dataset.

## American pilot (`asl`) — a pipeline proof, not a product

The first model trained on real human signing is **American** Sign Language, built to
exercise the whole pipeline end to end and to put a real signer-independent number on
it. It is not Jordanian, and it must not be read as evidence about LIU.

**Data.** PopSign ASL v1.0 (CC BY 4.0, phone-camera clips, 250-sign vocabulary,
`signdata.cc.gatech.edu`, no login). 8 signs (dog, fish, car, airplane, cat, apple,
blue, red), up to 3 clips per signer per sign: 865 clips from 47 signers. Hands are
tracked in under 60% of frames in 80 clips, which is typical for fast signing.

**Rebuild.**

```bash
# 1. fetch each sign's archives:  .../popsign_v1_0/game/{train,val,test}/<sign>.tar
# 2. unpack ≤3 clips per signer per sign into data/popsign/raw/<signer>/<sign>/*.mp4
# 3. landmarks, downscaled and with idle lead-in trimmed, 3 parallel workers
python ml/extract_landmarks.py data/popsign/raw --out data/popsign/processed \
       --max-side 480 --trim-idle --workers 3
python ml/clean.py data/popsign/processed/manifest.json
python ml/make_split.py data/popsign/processed/manifest.json --val 5 --test 5
python ml/train.py data/popsign/processed/manifest.json \
       --split data/popsign/processed/split.json --out models/asl_pilot.pt --mirror-prob 0.5
python ml/evaluate.py models/asl_pilot.pt data/popsign/processed/manifest.json \
       --split data/popsign/processed/split.json
```

The signer IDs come from the clip file names (`4a.<id>-…`). `--trim-idle` matters:
the kiosk starts an utterance when a hand appears, so training on empty lead-in would
score a distribution the kiosk never sees.

**Result (one evaluation, on 5 held-out signers, 89 clips, 8 signs; chance is 12.5%).**

| | Validation (5 signers) | Test (5 other signers) |
|---|---|---|
| no mirroring | 81.9% | not run |
| `--mirror-prob 0.5` (chosen on validation only) | 87.6% | **85.4% top-1**, 95.5% top-5 |

Per signer on test: 68.8%, 81.0%, 90.5%, 93.3%, 93.8%. The weakest sign is `dog` (44%
on 9 clips; it had the fewest training clips), and the model's own spread warning fires.
`ml/compare_splits.py` (a separate, short 25-epoch run, not this checkpoint) scored
65.2% on the same held-out signers but 88.8% when the test set was drawn at random
from all clips, an inflation of +23.6 points: that gap is how much a clip-level split
would have flattered this dataset. With 5 test signers the figure is noisy; treat it as "works on people it has not seen,
unevenly", not as a benchmark. Mirroring helped because the signers include left-handed
people; see the caution under Augmentation before enabling it for LIU.

**Through the real kiosk path (2026-10-10).** `ml/evaluate.py` scores pre-extracted
features, so it cannot see live failures. `python -m kiosk.replay` runs the same
held-out videos through the kiosk's own live perception, segmenter and recognizer, in
real time, as browser-style JPEG frames. First run: **0 of 8 clips produced a word**,
because the segmenter threw a word away when the hands dropped out of frame and only
emitted after a held 500 ms pause. After the fix (a word ends when the hands leave),
on all 89 held-out clips: **71% correct**, 15% wrong, 15% no word, 18% split into
several words. A second segmenter pass (motion matched by position instead of by
MediaPipe's flipping left/right label, smoothed over 3 frames so one noisy frame cannot
restart the pause timer, and a resting hand no longer re-emitted as a stream of words)
measured on a quiet machine: **75% correct, 10% wrong, 15% no word, 6% split**. That is
the number a visitor would experience; 85.4% is the classifier alone. The 15% "no
word" are clips where hand tracking found a hand in only a few frames; segmentation
cannot recover those.

**Refusal is the bigger problem.** On 30 clips of two signs the model was never taught
(book, drink), it named one of its 8 signs about **80% of the time**. A closed vocabulary
must name something for anything it sees.

*Tried, did not work:* an explicit `__other__` class trained on four other signs
(`ml/make_open_set.py`; the kiosk reads that class as a refusal). On unknown signs it
had never seen it still accepted 77%, and known-sign accuracy did not improve, so it
was not promoted. Four negative signs teach "refuse these four", not "refuse anything
unfamiliar". Model confidence separates known from unknown only weakly (AUROC 0.755; the
`__other__` model scored 0.699).

*Mitigation in place:* a per-language commit bar (`confident_threshold` in
`core/languages.json`), 0.88 for the 8-sign pilot, so uncertain signs become candidates
the visitor taps instead of an automatic word. Measured through the live path on 119
held-out clips (`python -m kiosk.replay`):

| Known signs (89 clips) | |
|---|---|
| committed and correct | 67% |
| committed but wrong | 6% |
| shown as candidates, correct sign among them | 12% |
| no word (hand tracking found no hand) | 13% |

| Unknown signs (30 clips) | |
|---|---|
| committed as a known sign (false accept) | **47%** |
| shown as candidates | 37% |
| refused | 10% |

That 47% is still too high to call safe. It will not come down by threshold tuning on an
8-sign vocabulary; it needs a much larger vocabulary and many diverse negative signs.

**Limits.** 8 signs. Phone selfie video, not the kiosk camera, so accuracy on the
kiosk will differ and has not been measured. Not reviewed by Deaf ASL signers. The
kiosk shows the card as "Pilot" and it stays that way until Deaf ASL users have tried
it on the kiosk camera. `models/` is gitignored; rebuild from the steps above.

## American, 400+ words (2026-10-12) — a research prototype, not a product

Goal: take `asl` from 8 words to 400+. **The model trained here is not shippable**: it uses
**ASL Citizen** (Microsoft Research License: non-commercial research use only; ROADMAP risk R12).
Using it for a commercial kiosk needs Microsoft's permission (ASL_Citizen@microsoft.com). Keep that
in the checkpoint's notes. PopSign (CC BY 4.0) is the only licence-clean part.

**Data.** PopSign: 50 words (`core/vocab_asl.json`), ≤3 clips per signer per word. ASL Citizen
(2,731 glosses, 83k webcam videos, 52 signers, the dataset's own signer-disjoint split): 421 words
chosen in `ml/asl_citizen_words.json` (id → Arabic gloss; **the Arabic is unreviewed machine
translation, have a Deaf Arabic-speaking reviewer check it**). A word is matched to a gloss by exact
lower-case name; variants (`ABOUT1`/`ABOUT2`) are never matched. 434 words overall.

```bash
# the zip is 46 GB; `curl -C -` resumes, but check it afterwards: a resumed download corrupted
# entries in the middle of this one (CRC errors) although the size looked right
python ml/import_asl_citizen.py data/asl_citizen/ASL_Citizen.zip --out data/asl_citizen/raw \
       --per-sign 16 --train-per-sign 30        # --train-per-sign: top up with training signers
python ml/extract_landmarks.py data/asl_citizen/raw --out data/asl_citizen/processed \
       --max-side 480 --trim-idle --hand-fallback --retry-side 720 --face --stride 2 --workers 2
python ml/merge_datasets.py --out data/asl_all --source data/popsign/processed \
       --source data/asl_citizen/processed --max-train-per-sign 40
```

**Why `--train-per-sign`.** A uniform pick across signers sends a third of the clips (the dataset's
validation and test signers) to data the model never trains on: the first merge had a median of 4
training clips per word. Training clips per word is the lever accuracy depends on most.

**Better extraction** (`ml/extract_landmarks.py`; every option is off by default, so a plain run
behaves as before):

| Option | What it does | Measured |
|---|---|---|
| `--hand-fallback` | a dedicated MediaPipe hand detector on frames where Holistic lost a hand; each hand goes to the nearer *pose wrist*, never by MediaPipe's own left/right label | hand visibility on the 8 worst clips 0.19 → 0.28 |
| `--retry-side N` | re-run a clip whose hands were found in under 60% of frames at a larger size, keep the better pass | included above |
| `--rtmw-fallback` | last-resort hands from RTMW (OpenMMLab; `rtmlib`, ONNX Runtime, CPU). 2-D, so z = 0; ~6× slower | hand found in **48%** of frames on the same 8 clips vs **12%** for MediaPipe+fallback; drawn on a frame, the 21 points sit on a motion-blurred hand MediaPipe misses |
| `--face` | writes `<clip>.face`, (64, 12) face features: brow raise, eye openness, mouth opening/width/corner lift, head roll/yaw/pitch, brow furrow | extraction only; see feature versions |
| `--stride 2` | track every second frame (clips are resampled to 64 anyway) | halves the time |

**Feature versions** (`core/features.py: model_input`, recorded in the checkpoint). 1 = position +
velocity (every model before this). 2 = + hand shape: 15 finger-flexion cosines, thumb-tip distances,
tip reach, finger spread and palm normal per hand, all scale-free and computed from the landmarks
already stored, so no re-tracking. 3 = + the face features above. **The kiosk runs 1 and 2 only**:
it refuses a version-3 checkpoint, because B2 Perception does not carry face landmarks yet, and the
server's shadow scoring refuses it for the same reason.

**Model and training** (`ml/model.py`, `ml/train.py`; all new options default to the old behaviour):
`--arch signnet` is a conv stem + Transformer + attention pooling; `--balance` class-balanced
sampling; `--supcon` supervised contrastive loss (positives are other signers' clips of the same
word); `--strong-augment` hand dropout, small rotation and zoom; `--text-embeddings` a head that
scores a clip by cosine similarity to each word's name embedding (`ml/label_embeddings.py`:
multilingual MiniLM, English and Arabic names averaged), so a word with few clips borrows structure
from related words; `--pretrained` an encoder from `ml/pretrain.py` (masked-frame reconstruction on
training signers' clips only, no labels). `ml/ensemble.py` averages checkpoints;
`ml/word_bars.py` writes per-word commit bars (precision ≥ target on the *validation* signers) into a
checkpoint, which the kiosk applies as `max(language bar, word bar)`: a word can only get stricter.
Every consumer rebuilds a model with `ml/model.py: build_model`, so none can build a different
network from the one trained. NLP libraries (`ml/requirements-train.txt`) are training-only; the
checkpoint carries the word vectors.

**Not done, deliberately.** Language-model rescoring of word sequences: there is no sentence-level
sign data to show it helps. The kiosk cannot use face features. The Arabic glosses and every ASL
Citizen label are unreviewed by Deaf signers.

### The ASL data system (`ml/asldb.py`, `data/asl.db`)

One SQLite file (standard library, nothing to install) that knows every clip, signer, word, licence
and model, so the rules this project must not break are enforced by the data, not remembered:

- **A signer is in exactly one split**, and the split belongs to the signer, not the clip. A clip
  cannot be random-split away from its signer.
- **`freeze-test` locks the test signers**; triggers refuse any later move, unfreeze or delete (DR-7a:
  the benchmark cannot drift into training).
- **Every clip has a source and every source a licence.** `export --licence-clean-only` builds a
  training set from commercially usable data only; `model-add` records whether a model is clean, so a
  research-only model cannot be mistaken for a shippable one (R12).
- **Labels are `unreviewed` until a Deaf reviewer confirms them**; `export --confirmed-only` trains on
  reviewed clips alone. `review-sheet` writes a CSV (video path, Arabic gloss) for a reviewer to mark
  `confirmed`/`rejected`; `review-import` reads it back and refuses a decision with no reviewer name.

```bash
python ml/asldb.py ingest --source popsign --root data/popsign/processed
python ml/asldb.py ingest --source asl_citizen --root data/asl_citizen/processed
python ml/asldb.py gaps          # what stands between this data and a trustworthy model, with fixes
python ml/asldb.py freeze-test && python ml/asldb.py check
python ml/asldb.py export --out data/asl_all --max-train-per-sign 24     # manifest.json / split.json
python ml/asldb.py review-sheet --out data/asl_review/review_sheet.csv --per-word 2
python ml/asldb.py model-add models/asl.pt --name asl434 --sources popsign,asl_citizen
```

**Gaps found on 2026-10-12 and what was done** (`asldb gaps`):

| Gap | Finding | Action |
|---|---|---|
| Training clips per word | 383 of 434 words under 20; median 8, minimum 4 | `--train-per-sign` top-up (2,838 clips queued); 43 more licence-clean PopSign words downloading |
| Class imbalance | PopSign words have up to 119 clips, most others 4-10. On validation the commonest wrong answers were the best-supplied words (`happy→thankyou` 9, `please→thankyou` 8, `car→milk`, `cat→mom`) | cap training clips per word near the median (`export --max-train-per-sign`), train with `--balance` |
| No negative examples | the model must name a word for anything it sees; 47% of unknown signs were accepted earlier | `import_asl_citizen.py --neg-glosses 250 --neg-holdout 80`: 750 clips of 250 out-of-vocabulary signs as `__other__` (the kiosk reads it as a refusal), 240 clips of 80 held-out signs as `__unseen__`, never trained on, for measuring false accepts |
| Test signers | not frozen | frozen (16 signers) |
| Labels | none confirmed by a Deaf reviewer | review sheet of 868 clips (2 per word, different signers) written; **needs a reviewer** |
| Licence | 6,346 of 9,809 clips are research-only; 52 of 434 words have licence-clean data | PopSign has 106 more of the words (CC BY, ~400 GB); the smallest 43 are downloading; Kaggle `asl-signs` (CC BY) needs the competition rules accepted by a person |
| Words never right on validation | 32, nearly all with under 10 clips | follows the clip top-up; re-check after |
| Hands / face | 69 clips with hands barely tracked, face features on 3,526 of 9,809 clips | queued re-extraction with RTMW hand fallback and `--face` |

A near-synonym pair (`pen`/`pencil`, `home`/`house`) may be one sign under two English words; only a
Deaf reviewer can say. Until then they will confuse each other.

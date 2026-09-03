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

`ml/features.py`, per frame:

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

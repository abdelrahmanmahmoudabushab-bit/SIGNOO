# جمع البيانات · Data Collection Protocol

The dataset is the project. Model architecture is close to a solved problem for
isolated sign recognition; **a clean, well-designed dataset is what decides whether
this works.** Read this before recording anything — mistakes made during collection
usually cannot be fixed afterwards.

---

## 1. Why we must collect our own

There is no public Jordanian Sign Language (LIU) dataset of usable size. A 2025
systematic review of Arabic Sign Language covering 123 studies and 27 datasets puts
**Jordan at 2.08%** of all ArSL dataset collection — against Egypt at 43.75% and
Saudi Arabia at 37.50%. The large public Arabic datasets (KArSL, ArSL2018) are Saudi.

They are still useful — as *pre-training*, see `docs/TRAINING.md` — but a model
trained only on them will not read Jordanian signing correctly. Regional sign
languages differ in vocabulary and articulation the way spoken dialects do.

**So the dataset is the real contribution.** Nobody else has it.

---

## 2. The rule that matters most: signers, not repetitions

The single most common mistake in sign language recognition is collecting many
repetitions from few people, then reporting accuracy on a random split. The model
memorises *the people*, scores 99%, and fails completely on a new signer.

> **20 signers × 5 repetitions beats 3 signers × 50 repetitions — every time.**

Even though the second option has more total samples.

Aim for:

| | Minimum | Good | Strong |
|---|---|---|---|
| Signers | 5 | 15 | 30+ |
| Repetitions per sign per signer | 3 | 5 | 8 |
| Signs (vocabulary) | 20 | 100 | 300+ |

Start with **20–30 signs and as many signers as you can reach**. A small vocabulary
that works for many people is far more useful — and far more publishable — than a
large vocabulary that only works for you.

### Where to find signers in Jordan

- The Holy Land Institute for the Deaf (Salt) — الأكاديمية الدولية للصم
- Jordanian Deaf clubs and associations (نادي الصم)
- Sign language interpreter programmes and certified interpreters
- University Deaf student services (JUST, University of Jordan, PSUT)
- Deaf community groups on social media

**Involve Deaf signers as partners, not as data sources.** They know which signs
matter, which regional variant is standard, and whether a recording is correct.
Native signers should validate the vocabulary list before you record it.

---

## 3. Consent and ethics

Sign language data is **face and body video of identifiable people** — it is
biometric personal data, and cannot be anonymised the way text can.

Before recording, every participant must give **written, informed consent** covering:

- what is being recorded, and what it will be used for
- who will be able to access it (your team only / researchers / public release)
- whether their face may appear in publications or demos
- their right to withdraw and have their recordings deleted, at any time, without
  giving a reason
- how long the data is kept and how it is stored

Provide the consent form **in writing and in LIU** — a hearing person reading a form
aloud is not accessible consent. Use a Deaf interpreter.

Practical rules:

- Store the signed forms separately from the videos, linked only by signer ID.
- Never use real names in filenames or metadata. Use `S01`, `S02`, ….
- Get separate, explicit consent for public release. Do not assume it.
- If you publish the dataset, consider releasing **landmarks only** — these carry the
  linguistic content but not the person's face.
- Pay or otherwise compensate participants for their time.

If this becomes an academic project, get ethics board (IRB) approval **before**
collecting, not after.

---

## 4. Recording setup

Consistency matters less than *variety* — a model trained on one perfect setup breaks
in a real room. Deliberately vary conditions across signers.

**Keep constant:**
- Upper body visible: head, both shoulders, and room for hands above the head
- Camera roughly at chest height, level, not tilted
- Signer's hands never leave the frame
- ≥ 20 fps, ≥ 640×480

**Deliberately vary:**
- Lighting — window light, ceiling light, evening, slightly dim
- Background — plain wall, furniture, doorway, a busy room
- Distance — 1 m to 2.5 m
- Clothing — including long sleeves, short sleeves, and patterned shirts
- Device — different phones and laptops

Avoid: strong backlight (a bright window behind the signer), hands out of frame,
and sleeves that hide the wrists.

---

## 5. Session procedure

For each signer:

1. Explain the project and obtain consent (in LIU).
2. Assign an anonymous ID: `S07`.
3. Record a short calibration clip: signer stands still, hands down, 2 seconds.
4. For each sign in the list:
   - Show the sign's written word **and a reference video**, never a written
     description — descriptions cause signers to invent variants.
   - Record the agreed number of repetitions, starting and ending with hands at rest.
   - Watch the first repetition back. If the framing is wrong, fix it now.
5. Note anything unusual in the session log: left-handed signer, injury, regional
   variant used, interruptions.

**Left-handed signers matter.** Include them and label them — do not mirror their
recordings into right-handed ones, and do not exclude them. Record it in the metadata
and let the model see both.

Keep sessions under 45 minutes. Signing is physical work, and tired signing is sloppy
signing. Take breaks.

---

## 6. What to save

For every clip, one video file and one metadata record:

```
data/raw/S07/greeting_marhaba/S07_marhaba_r03.mp4
data/raw/S07/greeting_marhaba/S07_marhaba_r03.json
```

```json
{
  "signer_id": "S07",
  "sign_id": "marhaba",
  "label_ar": "مرحبا",
  "label_en": "hello",
  "repetition": 3,
  "handedness": "right",
  "recorded_at": "2026-09-14T10:22:31Z",
  "fps": 30,
  "resolution": [1280, 720],
  "lighting": "daylight_window",
  "background": "plain_wall",
  "distance_m": 1.5,
  "device": "pixel_7",
  "validated_by": "native_signer_A",
  "notes": "regional variant, Amman"
}
```

`validated_by` is not optional bookkeeping. **Every sign should be checked by a Deaf
signer** who confirms the recording shows the intended sign, correctly produced. A
mislabelled sign is worse than a missing one: it actively teaches the model wrong.

---

## 7. Splitting — decide before you train

Assign each signer to exactly one of train / validation / test, **and never move
them**. Write the split into a file and commit it.

```
train: S01 S02 S04 S05 S07 S08 S10 S11 S13 S14
val:   S03 S09
test:  S06 S12 S15
```

Every reported number must come from signers the model has never seen. This will make
your accuracy look worse than published papers that split randomly. Your number is
the honest one, and it is the only one that predicts real use.

Keep the test signers genuinely held out: do not look at their results while tuning.
Use the validation signers for that.

---

## 8. Checklist

Before recording:
- [ ] Vocabulary list reviewed by native LIU signers
- [ ] Reference video recorded for every sign
- [ ] Consent forms ready, in writing and in LIU
- [ ] Ethics approval, if academic
- [ ] Signer IDs assigned, names kept separately

During:
- [ ] Hands stay in frame, upper body visible
- [ ] Conditions varied across signers, not within a signer's session
- [ ] Session log kept

After:
- [ ] Every clip has metadata
- [ ] Signs validated by a Deaf signer
- [ ] Split file written and committed
- [ ] Backed up in two places

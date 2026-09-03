---
name: slr-methodology
description: Methodology for sign language recognition work — dataset splits, evaluation that does not lie, feature design, and the traps that produce impressive numbers on models that fail in front of real people. Use this whenever training, evaluating, comparing, or reporting on any sign recognition model, when designing datasets or splits, when someone quotes an accuracy figure, and when reviewing retraining or model-promotion logic.
---

# Sign language recognition: methodology

The field is full of published 98% accuracies attached to models that fail on the
first new person who signs at them. Almost always the cause is one of the traps below,
and almost always the authors did not realise. Getting this right is the difference
between a demo and a system.

## Split by signer. Always.

**The rule:** a person appears in exactly one of train, validation, or test. Never two.

**Why it matters more than anything else here:** signers differ far more from each
other than repetitions differ within one signer — body proportions, signing speed,
resting posture, regional variant, personal accent. Given a random split, a model
takes the cheapest available route: learn the *people*, then recognise which person is
signing and recall what they usually do. That scores brilliantly and generalises to
nobody.

Measured on this project's own pipeline, on data where signer identity competes with
sign identity: **52.9% signer-independent versus 99.2% random-split** — same model,
same data, same training. Forty-six points of pure leakage.

Enforce it in code, not in discipline. A function that returns splits should refuse to
return an overlapping one; a person will eventually forget, and the failure is silent.

## The evaluation traps

Check every one of these before believing a number.

- **Repetition leakage** — clips from one recording session, split randomly. Near-
  duplicate frames land on both sides.
- **Near-duplicate files** — a clip saved twice under different names. Detect and
  remove before splitting, not after.
- **Averaging over signers.** A model at 85% mean may be 95% on four signers and 40%
  on the fifth. The mean hides the person the system fails. **Always report the
  per-signer spread**, and treat the worst signer as the real score.
- **Single-signer classes.** A sign recorded by one person cannot be evaluated
  signer-independently at all — whichever split that person lands in, the class is
  absent from the others. These inflate reported coverage while testing nothing.
- **Tuning on test.** Every look at the test set spends a little of its value. Tune on
  validation; touch test once, at the end.
- **Comparing to papers.** Many published sign-language results use random splits,
  three signers, or both. A signer-independent 75% is a stronger result than a
  random-split 98%. Do not "improve" a number by adopting a weaker protocol.

## Collect people, not repetitions

**20 signers × 5 repetitions beats 3 signers × 50**, even though the second has more
clips. Signer diversity is the axis that buys generalisation; repetitions past a
handful mostly buy correlated data.

When someone proposes growing the dataset, the first question is always *how many new
people*, not how many new clips.

Include left-handed signers and label them. Do not mirror them into right-handed
clips to "fix" the imbalance — that discards a real population the system must serve.

## Feature design

Landmarks over pixels, when data is scarce. Video models with a few thousand clips
learn the room, the lighting and the clothing; landmarks discard all of that before
training. The cost is real and must be stated: **everything not in the landmarks is
invisible**, above all facial expression, which carries grammar in sign languages —
negation, question marking, intensity. Signs distinguished only by expression cannot
be separated by a hands-and-body model. Say so rather than letting a stakeholder
assume otherwise.

Normalise away what the camera happens to do, keep what the language does:

- **Remove**: absolute position in frame, distance from camera, body size, signing
  speed. These vary with where someone stands and how tall they are.
- **Keep**: movement trajectory, the spatial relationship between the hands, handshape,
  which hand does what.

Centre a clip on the signer's own body and measure in body-relative units (shoulder
width is stable; hand size is not — hands move toward and away from the camera
constantly). Centre both hands on one shared origin, or the distance between them —
linguistically meaningful — is destroyed.

Append velocity. Many signs share a handshape and differ only in direction of
movement; that is hard to see in positions and obvious in differences.

## Augmentation: what is safe and what changes meaning

Safe: speed variation, landmark jitter at tracker-noise scale, whole-body translation,
brief frame dropout.

**Not safe, because they change what the sign means:**
- **Time reversal** — reverses directional signs (*give* becomes *take*).
- **Vertical flip** — meaningless as a body pose.
- **Large rotation** — orientation is part of a sign.
- **Mirroring** — usually preserves identity, since handedness is mostly a property of
  the signer, but some signs are genuinely asymmetric and in two-handed signs the
  dominant and non-dominant hands have different roles. Off by default; enable only
  after checking it does not hurt validation, and never as a substitute for recording
  actual left-handed people.

## Model size

Small, deliberately. With a few thousand clips, extra capacity buys memorisation of
signers. **If held-out-signer accuracy trails held-out-repetition accuracy, the fix is
more signers, not more parameters** — this is the most common wrong turn in the field,
and it is expensive because it feels like progress.

## Automatic retraining

Continuous learning from live use is safe only under one condition: **the evaluation
must not be automatic.**

A model retrained on its own confident predictions drifts toward its own biases and
degrades in ways that no self-generated metric can detect — it grows more confident
exactly as it grows more wrong. The gate must be a **frozen, human-verified benchmark**
that the training process cannot modify, extend, or select into.

A sound promotion gate checks, at minimum: the candidate beats the champion on that
frozen set; no individual signer regresses beyond a set margin; no class collapses to
zero recall; latency stays within budget. Failing any one, the champion stays. Prefer
shadow-running a candidate against live traffic before promoting, and keep rollback
one action away.

Live confidence is a monitoring signal, never a label. A user's on-screen confirmation
is a weak label worth collecting, and still belongs in a review queue rather than
straight into training.

## Reporting

State together, or the number is not interpretable: **top-1, the per-signer spread,
the number of signers in test, and the split protocol.** Name what is not covered —
facial expression, continuous signing, regional variants. A number without its
protocol is not a result.

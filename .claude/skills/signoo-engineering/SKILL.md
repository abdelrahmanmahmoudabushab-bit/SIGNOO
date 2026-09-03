---
name: signoo-engineering
description: House engineering standards for the SIGNOO codebase — module boundaries, error handling, testing discipline, and what "done" means here. Use this whenever writing, reviewing, or restructuring code in this repository, whether Python services, the Jetson kiosk runtime, the training pipeline, or the web console — including small changes, since most damage in this system comes from small changes made without knowing the boundaries.
---

# SIGNOO engineering standards

This system runs unattended, in public, for people who cannot hear it fail. A Deaf
person standing at a kiosk cannot be told "sorry, try again" by a voice. That
constraint drives everything below.

## The rule that overrides the others

**A change is not done until you have watched it work.** Not "the code looks right",
not "the types check" — you ran it and observed the actual behaviour.

This is not ceremony. Every serious bug in this project so far was found by running
the real pipeline, not by reading code: a shared landmark tracker that bled state
between clips, an API that no longer existed in the installed library version. Both
looked correct on the page.

For each layer, "watched it work" means:

| Layer | Evidence |
|---|---|
| Pure functions (features, parsing) | Unit test with a case that fails if the logic inverts |
| Model code | One forward and backward pass, shapes printed |
| Pipeline stage | Real input file through the real stage, output inspected |
| Kiosk UI | Driven in a browser, screenshot taken |
| Service | Request issued, response body shown |

If you cannot run it in the current environment, say so plainly and name what remains
unverified. An honest gap is useful; a claim of "should work" is not.

## Module boundaries

The system is a pipeline of blocks with narrow contracts. Blocks may not reach past
their neighbours.

```
capture → perception → segmentation → recognition → assembly → output
                                          ↕
                                     model registry
```

Rules that keep this honest:

- **A block takes and returns plain data** — arrays, dataclasses, dicts. Never a live
  camera handle, never a model object, never a database session. This is what lets you
  replay a recorded session through the recognition stage without a camera attached,
  which is how you debug a kiosk that misbehaves in a mall three cities away.
- **Perception knows nothing about signs.** It emits landmarks. Recognition knows
  nothing about cameras or languages. Assembly knows nothing about neural networks.
  When you feel tempted to pass "just one flag" backwards through the pipeline, that
  is the design telling you the block boundary is in the wrong place — move it
  deliberately, don't tunnel through it.
- **Configuration enters at the edge**, resolved once at startup into a typed object
  and passed down. A function that reads an environment variable in the middle of the
  pipeline cannot be tested.

## Error handling

Distinguish three kinds of failure, because they need different responses:

1. **Expected and recoverable** — no hands in frame, network briefly down, a clip too
   short. Handle in place, return a typed result, keep serving. Never raise.
2. **Wrong input, caller's fault** — malformed manifest, unknown language code. Raise
   with a message that names the offending value and what was expected. The person
   reading it is tired and three layers away from the cause.
3. **Broken invariant, our fault** — model output shape does not match the label map.
   Fail loudly and immediately. A kiosk that silently mistranslates is worse than a
   kiosk showing an out-of-service screen.

Never write `except Exception: pass`. If a failure is genuinely ignorable, log it at
debug level with the reason it is ignorable — future you will not remember.

## Python style

Match the surrounding code first; these apply where there is no precedent.

- Type hints on anything crossing a module boundary. Inside a function, only where
  they clarify.
- `dataclass` for data that travels between blocks. Frozen where it should not change.
- `pathlib.Path`, never string paths.
- f-strings; no `%` or `.format`.
- Small functions, but not artificially so — a 40-line function doing one clear thing
  beats four 10-line functions that must be read together.
- Names say what a thing *is*, not what type it is: `clip`, `landmarks`, `signer_id`,
  not `data`, `arr`, `tmp`, `x2`.

## Comments

Comment the **why**, never the what. The code already says what it does.

```python
# Bad — restates the code
# resample to 64 frames
clip = resample(clip, 64)

# Good — explains a decision the reader cannot recover from the code
# Signers vary in speed by 2x or more, and the same person is inconsistent between
# takes. Resampling puts fast and slow productions on one time axis so the model only
# has to learn the shape of the movement.
clip = resample(clip, SEQ_LEN)
```

A constant with a tuned value deserves a note on what happens above and below it. That
is the comment that saves someone a week.

## Testing

Test what would actually break, and prefer one test that would catch a real regression
over five that restate the implementation.

Worth testing here: feature extraction invariances (does the same sign survive a
different camera distance?), split integrity (does a signer ever appear on both
sides?), promotion-gate logic (does a worse model get rejected?), state machines
(does the kiosk return to idle from every state?).

Not worth testing: that a dataclass stores its fields, that a wrapper calls the thing
it wraps.

**Fixtures must be honest.** A synthetic clip that is trivially separable proves
nothing — build fixtures where the confound you care about is actually present, or the
test passes for the wrong reason.

## What to do when uncertain

Prefer the boring choice. This system will be maintained by people who did not write
it, on hardware they cannot easily reach, serving users who cannot phone support. A
clever abstraction that saves fifty lines and costs an hour of understanding is a bad
trade here.

When two designs seem equal, pick the one that fails more visibly.

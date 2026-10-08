---
name: signoo-avatar
description: Design and build rules for SIGNOO's B8 signing character — the 3D avatar (face, body, hands) that signs an answer back to the visitor. Use whenever touching B8/sign playback, the avatar, retargeting, the rig, or anything that turns a recorded or recognised sign into on-screen motion.
---

# The signing character (B8)

B8 answers a question by *performing the sign back*, through a 3D character on
screen. This skill exists because the easy version of this feature is the dangerous
one — read it before writing any retargeting or rendering code.

## The rule that cannot be relaxed

**Every sign the character performs must be a real recording, retargeted onto the
rig — never generated.** `docs/SYSTEM_DESIGN.md` B8 states the reasoning in full; the
short version: generating grammatical sign language from text is an open research
problem, and a system that produces plausible-looking but linguistically wrong
signing is actively harmful to a Deaf visitor, who has no channel to detect the error
the way a hearing person might catch a garbled sentence. "It looked right to a hearing
engineer" is not a correctness check.

If a design under discussion involves a model producing *novel* pose sequences from
text — even a small one, even "just for words we don't have a recording of yet" — stop
and raise it explicitly rather than building it quietly. That line has been crossed
once already in this project's planning and was reversed after the safety
implication was made explicit; it is worth the same scrutiny every time it resurfaces.

## What "retargeted" means here, concretely

1. A real Deaf signer's landmark sequence is already captured by the existing pipeline
   (`core/features.py`'s representation, `kiosk/data_capture.py`'s recording path) —
   this is not new data collection, it's reusing data the project collects anyway.
2. That sequence is mapped onto a fixed 3D rig's skeleton (retargeting), so every word
   is performed by the *same* on-screen character regardless of which real signer
   contributed the recording.
3. Retargeting changes *who appears to be signing*, never *what sign is performed*. If
   a retargeting step starts smoothing, interpolating, or "cleaning up" the motion
   enough that it could plausibly drift the handshape or trajectory into a different
   sign, that is no longer retargeting — it is unreviewed generation wearing a
   retargeting label, and needs the same scrutiny as the rule above.

## Correctness check, before anything is trusted live

A retargeted clip is not verified by "it looks like a person moving." The check is: a
Deaf reviewer watches the retargeted output and confirms it still reads as the
intended sign, in the same way `slr-methodology`'s frozen benchmark is human-verified
before training trusts it. Treat the first retargeting implementation as a spike with
this check as its exit criterion, not as done once the pipeline runs end to end without
crashing.

## Technical shape (as of 2026-10, mostly undecided — see `docs/TECH_STACK.md` §3)

- The kiosk frontend is already browser-based (served statically by `kiosk/main.py`),
  so a WebGL renderer — most plausibly Three.js with a rigged GLTF/GLB humanoid — fits
  without adding a second runtime process. Treat this as the working default, not a
  locked decision; revisit if a concrete blocker shows up.
- Retargeting itself (MediaPipe Holistic landmarks → an arbitrary rig's bone
  transforms) has no chosen library or algorithm yet. Proving it on synthetic
  landmarks first (same caveat as `kiosk/tests/test_smoke_pipeline.py` — proves
  wiring, not correctness) is reasonable before any real recording is involved.
- Coverage is a content problem, not a modelling one: every word in every answer still
  needs a real recorded sequence, per language. This cost doesn't shrink by switching
  from video to landmarks — it gets cheaper to *capture* (no camera/lighting/wardrobe
  concerns, just the same landmark recording B10 already does) but the *count* of
  words needing a recording is unchanged. The console's unrenderable-word warning
  (FR-W7) still applies.

## When reviewing or extending this code

Ask, in order:

1. Does every path that ends in on-screen motion trace back to a real recorded
   sequence? If any path can produce motion with no such sequence behind it, that's
   the rule violation above, however it's described.
2. Has the specific sign been through the Deaf-reviewer correctness check, or is it
   still in the "pipeline proven, not verified" state?
3. Is the retargeting step doing anything beyond re-mapping joints — smoothing across
   different signs, blending two recordings, extrapolating a pose that wasn't
   recorded? Any of those needs the same scrutiny as generation.

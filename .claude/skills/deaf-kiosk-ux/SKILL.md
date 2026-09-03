---
name: deaf-kiosk-ux
description: Deaf-first design rules for the SIGNOO kiosk — what a public terminal must do when its user cannot hear it, how to show system state and uncertainty visually, consent and privacy for camera-based interaction, and language/RTL handling for Arabic. Use this whenever designing or reviewing any kiosk screen, flow, error state, prompt, or on-screen wording, and whenever deciding what feedback the system gives the user.
---

# Deaf-first kiosk design

The user standing at this kiosk cannot hear a beep, a voice prompt, or an error chime.
Anything the system needs to communicate has to be **visible**. That single constraint
invalidates most default interaction patterns, which quietly assume an audio channel.

Design for someone who may also be unfamiliar with the machine, in a public place,
possibly with people waiting behind them.

## Sign language is a language, not accessibility decoration

Jordanian (LIU), Emirati (ESL) and American (ASL) sign languages are distinct
languages with their own vocabulary and grammar — not signed versions of Arabic or
English. Two consequences that keep being got wrong:

- **Written Arabic is not automatically accessible.** Many Deaf people are far more
  fluent in their sign language than in written Arabic, which is effectively a second
  language learned through a channel they cannot hear. Text is necessary, not
  sufficient — pair it with sign video wherever the message matters.
- **Output glosses are not sentences.** A sequence of recognised signs follows the
  grammar of the sign language, not Arabic word order. Present it as what it is, and
  do not silently "correct" it into fluent Arabic without showing the user what was
  recognised — they need to see whether the machine understood them.

Say "Deaf person" or "signer", never "hearing-impaired", "deaf-mute", or "suffering
from". Wording in the product is read by the community it serves.

## Every state must be visible

The user cannot hear the system thinking. Show, continuously and without being asked:

- **Is the camera on and does it see me?** A live preview with the tracked skeleton
  overlaid answers this instantly and needs no language at all. This is the single
  highest-value element on the screen.
- **Am I in frame?** If hands leave the frame or the body is not detected, say so
  visually and immediately — an outline, an arrow, a dimmed edge. Do not wait for the
  interaction to fail.
- **Is it listening or thinking?** A distinct visual state for capturing versus
  processing. Ambiguity here makes people repeat themselves mid-sign, which corrupts
  the very clip being captured.
- **What did it understand?** Show the recognised words as they arrive, not only the
  final sentence. Partial output that the user can see going wrong is far better than
  a confident wrong answer at the end.

## Show uncertainty honestly

A translation system that hides its doubt is dangerous — the user acts on the output
believing it is right.

- Display confidence in a form that needs no reading: a solid word versus a faded one.
- Below the confidence threshold, **offer the top few candidates as large touch
  targets** rather than guessing. Choosing from three is faster than repeating a sign,
  and it produces a labelled example for training.
- Never silently substitute a guess. A visible "I did not catch that — please sign
  again" is a good outcome; a confident mistranslation is a harmful one.
- Make correction always available — a clear way to remove the last word and re-sign
  it, on every screen where text is produced.

## Consent, privacy, dignity

The camera is pointed at a person's face and body in public, and face-and-body video
of an identifiable person is biometric data that cannot be anonymised the way text can.

- **Ask before the camera starts**, in sign video and in text, and say plainly what is
  recorded, what is kept, for how long, and who sees it.
- **Recording for training must be a separate, explicit opt-in**, never bundled into
  "use this kiosk". The system must work fully for someone who declines — a refusal
  that degrades service is not a real choice.
- Default to keeping **landmarks rather than video**. They carry the linguistic
  content without the person's face, and they are what training actually needs.
- Show an unmistakable indicator whenever recording is happening, and a one-touch way
  to stop and delete the current session.
- **Delete on abandonment.** Someone who walks away has not consented to anything.
  Clear the session on timeout, and return to a clean idle screen so the next person
  never sees the previous person's conversation.

## Physical and interaction constraints

- **Reachable and readable from a standing position**, including from a wheelchair —
  controls in the lower two-thirds, nothing critical at the top edge.
- **Large touch targets**, generously spaced. People will be signing with both hands
  and then touching the screen; precision pointing is the wrong assumption.
- **One decision per screen.** The flow is: choose language → sign → see result → get
  answer. Keep each step doing one thing.
- **No time pressure.** Signing takes as long as it takes. If a timeout is unavoidable
  for privacy, warn visually with a countdown and let the user extend it with one
  touch.
- **Icons and sign video over text** wherever a message can be carried without words —
  it also sidesteps the Arabic/English choice entirely.

## Arabic and RTL

- Arabic UI is right-to-left: use logical CSS properties (`inset-inline-start`,
  `margin-inline`) so a language switch does not require a second stylesheet.
- Show Arabic and English **together**, not behind a toggle, when displaying a
  translation — a companion, family member, or staff member may read one and not the
  other, and the pairing is the point.
- Use Arabic-Indic digits where the surrounding Arabic text uses them, and keep
  identifiers (signer IDs, codes) left-to-right with `dir="ltr"` so they do not
  visually reorder.

## Test with Deaf users, early

Nothing in this document substitutes for that. Hearing designers reliably get three
things wrong: they underestimate how much is communicated by facial expression, they
overestimate written-Arabic fluency, and they build flows that assume the user will
wait patiently for a machine that gives no visible sign of life.

Recruit Deaf testers before the first build is considered finished, pay them, and
treat their judgment as authoritative on wording and flow.

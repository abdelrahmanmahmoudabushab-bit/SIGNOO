---
name: signoo-language-scaling
description: Rules for adding, prioritising, or gating a new sign-language "pack" in SIGNOO (Saudi, UAE, Omani, Qatari, French/LSF, American/ASL, British/BSL, International Sign, and whatever comes after). Use whenever deciding which language to recruit next, wiring a new language into the kiosk menu, or reviewing whether a language is ready to go live.
---

# Scaling to more than one sign language

SIGNOO's language menu names nine sign languages. Each one is a **separate
recognition pipeline** — its own signer recruitment, its own dataset, its own trained
head on the shared backbone, its own clip/motion library, its own Deaf reviewers — not
a caption-language toggle on top of one system. This skill is about not losing that
distinction under UI pressure to "just add a card."

## The principle this scaling has to respect

`docs/ROADMAP.md`'s phasing principle: **one language taken end to end before the
next is started.** Building N languages in parallel multiplies the data burden by N
while teaching the team nothing N times over — recruitment, consent, review capacity,
and evaluation discipline are the bottleneck, not engineering time. Nine cards in the
UI does not mean nine recruitment efforts running at once; it means nine *slots*, most
of them honestly empty.

## The live/not-live rule

A language card is **live** only when it has:

- A real (non-synthetic) dataset meeting Phase 1's bar: signer-independent split,
  enough signers that the per-signer floor is meaningful (`slr-methodology` —
  "20 signers × 5 repetitions beats 3 signers × 50")
- A trained head that passed the promotion gate (B13) against a frozen, human-verified
  benchmark for *that* language
- Deaf reviewers for that specific language/region — a Jordanian reviewer is not
  authoritative for Emirati Sign Language; regional sign languages are distinct
  languages, not dialects of each other, same reasoning `deaf-kiosk-ux` applies to LIU
  vs. written Arabic

Everything else shows **"recording in progress,"** plainly, and never runs an
untrained or synthetic-only model behind a card that looks live. A visitor who taps a
not-ready card and gets a confident wrong translation is the exact harm the rejection
threshold exists to prevent elsewhere in the system (`SYSTEM_DESIGN.md` B4) — don't
reintroduce it at the menu level.

## Prioritising which language is next

No fixed order is decided (`docs/ROADMAP.md`, "Next two weeks"). When the question
comes up, weigh:

- **Deployment site.** If a specific kiosk location/country is already committed,
  that country's sign language outranks every other consideration.
- **Public data availability.** ASL has by far the most public data and tooling of
  the nine — fastest path to a second working pipeline, but the resulting model will
  only be as good as that transfer actually helps the target language's head.
  Verify licence terms before using any public dataset beyond internal prototyping
  (`docs/ROADMAP.md` R12 — WLASL/ASL Citizen/How2Sign are non-commercial research
  licences; KArSL/ArabSign are gated behind author contact).
- **Shared recruitment channels.** The Gulf cluster (Saudi, UAE, Omani, Qatari) sits
  near Jordan's own institutional contacts (Holy Land Institute for the Deaf, Zayed
  Higher Organization) — recruiting one may open a path to the others faster than an
  unrelated language would.
- **Never** prioritise by "which card looks good in a demo." That is exactly the
  pressure this skill exists to resist.

## Wiring a new language in, once data exists

1. Language pack = {dataset, trained head, clip/motion library, knowledge-base content,
   Deaf review sign-off}, per `SYSTEM_DESIGN.md` B4's "Language packs" note.
2. Flip the card from "recording in progress" to live only after the promotion gate
   (B13) has passed *for that language's* frozen benchmark — not the LIU one.
3. Re-run the full evaluation discipline from scratch per language
   (`slr-methodology`): a language-independent accuracy claim does not transfer from
   LIU's numbers, even with a shared backbone.

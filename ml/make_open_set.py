"""Builds an open-set manifest: a closed vocabulary plus an explicit "not one of mine" class.

    python ml/make_open_set.py data/popsign/processed/manifest.json \
        --known dog fish car airplane cat apple blue red \
        --other bird hat shoe milk --other-max 220

A classifier trained only on its own signs must name one of them for anything it sees,
and does so confidently: on held-out recordings of signs it was never taught, such a
model called a known sign 80% of the time. Giving it a class of *other* signs to refuse
lets it say "none of these". The kiosk reads that class (`__other__`) as a rejection.

The output manifest is written next to the input so its relative feature paths stay
valid. Signs in neither list are dropped, which is how you keep some unknown signs
completely out of training to test refusal honestly on signs the model has never seen.
The "other" clips are sampled evenly across signers and capped, so that class does not
swamp the real ones.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OTHER = "__other__"


def build(records: list[dict], known: set[str], other: set[str], other_max: int, seed: int) -> list[dict]:
    kept = [dict(r) for r in records if r["sign_id"] in known]
    pool = [r for r in records if r["sign_id"] in other]
    rng = np.random.default_rng(seed)
    by_signer = defaultdict(list)
    for r in pool:
        by_signer[r["signer_id"]].append(r)
    for clips in by_signer.values():
        rng.shuffle(clips)

    chosen = []
    while len(chosen) < other_max and any(by_signer.values()):      # round-robin across signers
        for signer in sorted(by_signer):
            if by_signer[signer] and len(chosen) < other_max:
                chosen.append(by_signer[signer].pop())
    for r in chosen:
        kept.append({**r, "sign_id": OTHER, "label_ar": "", "label_en": ""})
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--known", nargs="+", required=True)
    ap.add_argument("--other", nargs="+", required=True)
    ap.add_argument("--other-max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: manifest_open.json next to the input")
    args = ap.parse_args()

    src = Path(args.manifest)
    overlap = set(args.known) & set(args.other)
    if overlap:
        raise SystemExit(f"signs listed as both known and other: {sorted(overlap)}")

    records = json.loads(src.read_text(encoding="utf-8"))
    out = build(records, set(args.known), set(args.other), args.other_max, args.seed)
    dest = Path(args.out) if args.out else src.with_name("manifest_open.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = defaultdict(int)
    for r in out:
        counts[r["sign_id"]] += 1
    print(f"{len(out)} clips → {dest}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()

"""Picks validation and test signers for a manifest, and writes split.json.

    python ml/make_split.py data/popsign/processed/manifest.json --val 5 --test 5

Splits by signer, never by clip: every signer lands in exactly one of train, val and
test, so a score measures recognition of people the model has never seen. It also
insists that val and test each contain every sign, because a held-out set missing a
sign cannot tell you whether that sign is learned. The seed is searched until that
holds, and the result is deterministic for a given manifest and --seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest


def choose(signs_by_signer: dict[str, set[str]], n_val: int, n_test: int, seed: int, tries: int = 500):
    signers = sorted(signs_by_signer)
    if n_val + n_test >= len(signers):
        raise SystemExit(f"{len(signers)} signers cannot give {n_val} val + {n_test} test and still train")
    every_sign = set().union(*signs_by_signer.values())
    for attempt in range(tries):
        order = list(signers)
        np.random.default_rng(seed + attempt).shuffle(order)
        test, val = order[:n_test], order[n_test:n_test + n_val]
        covers = lambda group: set().union(*(signs_by_signer[s] for s in group)) == every_sign
        if covers(test) and covers(val):
            return sorted(val), sorted(test)
    raise SystemExit("no val/test signer choice covers every sign; lower --val/--test or record more signers")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: split.json next to the manifest")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    signs_by_signer = defaultdict(set)
    for s in samples:
        signs_by_signer[s.signer_id].add(s.sign_id)

    val, test = choose(signs_by_signer, args.val, args.test, args.seed)
    out = Path(args.out) if args.out else Path(args.manifest).with_name("split.json")
    out.write_text(json.dumps({"val": val, "test": test}, indent=2), encoding="utf-8")

    held = set(val) | set(test)
    counts = {name: sum(1 for s in samples if (s.signer_id in group))
              for name, group in (("val", set(val)), ("test", set(test)))}
    counts["train"] = sum(1 for s in samples if s.signer_id not in held)
    print(f"{len(signs_by_signer)} signers · {len({s.sign_id for s in samples})} signs · "
          f"clips: train {counts['train']}, val {counts['val']}, test {counts['test']}")
    print(f"val signers:  {', '.join(val)}\ntest signers: {', '.join(test)}\n→ {out}")


if __name__ == "__main__":
    main()

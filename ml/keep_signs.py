"""Writes a manifest holding only the listed signs.

    python ml/keep_signs.py data/popsign/processed/manifest.json \
        --keep dog fish car airplane cat apple blue red

`ml/train.py` makes a class of every sign in the manifest it is given. Signs held out
to test refusal (clips of signs the model must never have seen) therefore have to be
absent from the training manifest, while staying in the full one so
`ml/calibrate.py --unknown` can score them. Train on this script's output; calibrate
on the full manifest.

The output is written next to the input, because the feature paths inside a manifest
are relative to the manifest's own directory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def keep_signs(records: list[dict], keep: set[str]) -> list[dict]:
    """The records whose sign is in `keep`, in their original order."""
    return [r for r in records if r["sign_id"] in keep]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--keep", nargs="+", required=True, help="sign ids to keep; all others are dropped")
    ap.add_argument("--out", help="default: manifest_known.json next to the input")
    args = ap.parse_args(argv)

    src = Path(args.manifest)
    dest = Path(args.out) if args.out else src.with_name("manifest_known.json")
    if dest.resolve().parent != src.resolve().parent:
        raise SystemExit(f"--out must be in {src.parent}: the manifest's feature paths are relative to it")

    records = json.loads(src.read_text(encoding="utf-8"))
    kept = keep_signs(records, set(args.keep))
    counts = Counter(r["sign_id"] for r in kept)
    # A misspelt sign would otherwise just vanish from the vocabulary without a word.
    empty = sorted(set(args.keep) - set(counts))
    if empty:
        raise SystemExit(f"no clips in {src} for: {empty}")

    dest.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    dropped = sorted({r["sign_id"] for r in records} - set(counts))
    print(f"{len(kept)} of {len(records)} clips, {len(counts)} signs → {dest}")
    print(f"dropped: {', '.join(dropped) if dropped else 'nothing'}")


if __name__ == "__main__":
    main()

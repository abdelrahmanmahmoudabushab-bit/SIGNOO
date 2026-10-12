"""Per-word confidence bars: the confidence each word must reach before the kiosk commits it unprompted.

    python ml/word_bars.py models/asl.pt data/asl_all/manifest.json --split data/asl_all/split.json \
        --target 0.9 --out models/asl.bars.pt

One bar per language treats every word alike, but a model is not equally reliable on every word: it
is confident and right on `apple`, confident and wrong on a word it mixes up with another. For each
word, this finds the lowest confidence at which the model's predictions of that word are right at
least --target of the time, measured on the *validation* signers (never the test signers: those
stay for the one final number). A word it cannot make that promise for gets bar 1.0: the kiosk then
always offers it as a candidate for the visitor to confirm and never commits it on its own.
Words with too few validation predictions get no entry and fall back to the language's bar.

The kiosk applies `max(language bar, word bar)` (Recognizer.commit_bar), so a word bar can only
make it stricter. The bars are written into a copy of the checkpoint under `word_bars`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_manifest, split_by_signer
from evaluate import predict
from model import build_model


def word_bars(predicted, confidence, truth, target: float, min_count: int) -> dict[str, float]:
    """{word: bar} from parallel arrays of predicted word, its confidence, and the true word."""
    bars = {}
    for word in sorted(set(predicted)):
        picks = [(c, p == t) for p, c, t in zip(predicted, confidence, truth) if p == word]
        if len(picks) < min_count:
            continue
        picks.sort(key=lambda x: -x[0])
        best = None
        right = 0
        for n, (conf, ok) in enumerate(picks, 1):
            right += ok
            if n >= min_count and right / n >= target:
                best = conf                    # lowest confidence so far that still keeps the promise
        bars[word] = float(min(max(best, 0.0), 1.0)) if best is not None else 1.0
    return bars


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=0.9, help="required precision of committed predictions")
    ap.add_argument("--min-count", type=int, default=3, help="validation predictions needed to set a word's bar")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]
    inverse = {v: k for k, v in labels.items()}
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    _, val, _ = split_by_signer(load_manifest(args.manifest), split["val"], split["test"])
    val = [s for s in val if s.sign_id in labels]

    model = build_model(ckpt)
    preds, probs = predict(model, val, labels, "cpu", feature_version=ckpt.get("feature_version", 1))
    predicted = [inverse[int(p)] for p in preds]
    confidence = probs[np.arange(len(preds)), preds]
    bars = word_bars(predicted, confidence, [s.sign_id for s in val], args.target, args.min_count)

    ckpt["word_bars"] = bars
    torch.save(ckpt, args.out)
    never = sorted(w for w, b in bars.items() if b >= 1.0)
    print(f"{len(val)} validation clips · {len(bars)} words got a bar · {len(never)} always shown as candidates")
    if never:
        print("never auto-committed:", ", ".join(never[:30]) + (" …" if len(never) > 30 else ""))


if __name__ == "__main__":
    main()

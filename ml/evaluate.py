"""Evaluates a trained model on the held-out test signers.

    python ml/evaluate.py models/signoo.pt data/processed/manifest.json --split data/split.json

Reports per-signer accuracy, per-sign accuracy and the worst confusions. The headline
number matters less than the spread: a model that averages 85% but collapses to 40%
on one signer is not ready, and the average hides that.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SignDataset, load_manifest, split_by_signer
from model import build_model


def predict(model, samples, labels, device, batch_size=32, feature_version=1, tta=1):
    """Predicted class and class probabilities per clip.

    With `tta` > 1 each clip is also scored through tta-1 gentle, deterministic re-timings of itself
    and the probabilities are averaged: a free few points when the model's answer wobbles with timing.
    """
    model.eval()
    totals = None
    for view in range(max(1, tta)):
        loader = DataLoader(SignDataset(samples, labels, augment=False, feature_version=feature_version,
                                        tta_view=view), batch_size=batch_size)
        views = []
        with torch.no_grad():
            for x, _ in loader:
                views.append(torch.softmax(model(x.to(device).float()), dim=1).cpu().numpy())
        views = np.concatenate(views)
        totals = views if totals is None else totals + views
    probs = totals / max(1, tta)
    return probs.argmax(1), probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--report", help="write full metrics to this JSON file")
    ap.add_argument("--tta", type=int, default=1,
                    help="score each clip through this many gentle re-timings and average (1 = off)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    labels = ckpt["labels"]
    inverse = {v: k for k, v in labels.items()}

    samples = load_manifest(args.manifest)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    _, _, test = split_by_signer(samples, split["val"], split["test"])

    model = build_model(ckpt).to(device)

    preds, probs = predict(model, test, labels, device, feature_version=ckpt.get("feature_version", 1),
                           tta=args.tta)
    truth = np.array([labels[s.sign_id] for s in test])
    correct = preds == truth

    print(f"\ntest signers: {', '.join(sorted({s.signer_id for s in test}))}")
    print(f"{len(test)} clips · {len(labels)} signs\n")
    print(f"top-1 accuracy: {correct.mean():.1%}")

    # Top-5 shows whether the right answer is at least in contention — useful when
    # the interface can offer the user a choice rather than committing to one word.
    top5 = np.argsort(-probs, axis=1)[:, :5]
    print(f"top-5 accuracy: {np.mean([t in row for t, row in zip(truth, top5)]):.1%}")

    # ── per signer ────────────────────────────────────────────────────────
    print("\nper signer:")
    by_signer = defaultdict(list)
    for s, ok in zip(test, correct):
        by_signer[s.signer_id].append(ok)
    per_signer = {k: float(np.mean(v)) for k, v in sorted(by_signer.items())}
    for signer, acc in per_signer.items():
        print(f"  {signer}: {acc:.1%}  ({len(by_signer[signer])} clips)")

    if per_signer and max(per_signer.values()) - min(per_signer.values()) > 0.15:
        print("\n  ⚠ large spread between signers — the model is signer-dependent.")
        print("    More signers in training will help more than a bigger model.")

    # ── per sign ──────────────────────────────────────────────────────────
    by_sign = defaultdict(list)
    for s, ok in zip(test, correct):
        by_sign[s.sign_id].append(ok)
    per_sign = {k: float(np.mean(v)) for k, v in by_sign.items()}

    worst = sorted(per_sign.items(), key=lambda kv: kv[1])[:10]
    print("\nweakest signs:")
    for sign, acc in worst:
        print(f"  {sign}: {acc:.0%}  ({len(by_sign[sign])} clips)")

    # ── confusions ────────────────────────────────────────────────────────
    confusions = defaultdict(int)
    for t, p in zip(truth, preds):
        if t != p:
            confusions[(inverse[t], inverse[p])] += 1

    print("\nmost common confusions:")
    for (true_sign, pred_sign), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {true_sign} → {pred_sign}  ×{n}")
    print("\nConfused pairs are worth watching by eye: often they genuinely differ")
    print("only in facial expression, which a hand-and-body model cannot see.")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "top1": float(correct.mean()),
            "top5": float(np.mean([t in row for t, row in zip(truth, top5)])),
            "per_signer": per_signer,
            "per_sign": per_sign,
            "confusions": {f"{a}->{b}": n for (a, b), n in confusions.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport → {args.report}")


if __name__ == "__main__":
    main()

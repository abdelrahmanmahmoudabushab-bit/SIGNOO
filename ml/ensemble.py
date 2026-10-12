"""Combines several trained checkpoints into one that averages their predictions.

    python ml/ensemble.py models/a.pt models/b.pt models/c.pt --out models/asl_ensemble.pt

Models trained from different seeds (or different architectures) make different mistakes, and
averaging their probabilities cancels some of them: typically a few points of top-1 for the price of
running N small networks. Members must share the vocabulary (same labels, same indices) and the
feature version, or the average would mix up words / feed one member features it was not trained on.
The result is a normal checkpoint with a `members` list; `ml/model.py:build_model` rebuilds it, so
the kiosk, evaluate.py and calibrate.py need no special case.
"""
from __future__ import annotations

import argparse

import torch


def combine(checkpoints: list[dict]) -> dict:
    if len(checkpoints) < 2:
        raise ValueError("an ensemble needs at least two checkpoints")
    first = checkpoints[0]
    for i, ckpt in enumerate(checkpoints[1:], 1):
        if ckpt["labels"] != first["labels"]:
            raise ValueError(f"checkpoint {i} has a different vocabulary or label order")
        if ckpt.get("feature_version", 1) != first.get("feature_version", 1):
            raise ValueError(f"checkpoint {i} was trained on a different feature version")
    for ckpt in checkpoints:
        if "members" in ckpt:
            raise ValueError("nesting ensembles is not supported")
    members = [{k: v for k, v in c.items() if k in ("model", "labels", "input_dim", "arch", "config")}
               for c in checkpoints]
    return {
        "members": members,
        "labels": first["labels"],
        "input_dim": first["input_dim"],
        "feature_version": first.get("feature_version", 1),
        "arch": "ensemble",
        "val_acc": max(float(c.get("val_acc", 0.0)) for c in checkpoints),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in args.checkpoints]
    torch.save(combine(ckpts), args.out)
    print(f"{len(ckpts)} models -> {args.out}")


if __name__ == "__main__":
    main()

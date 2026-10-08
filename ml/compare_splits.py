"""Measures how much a random split inflates accuracy.

    python ml/compare_splits.py data/processed/manifest.json --split data/split.json

Trains the same model twice on the same data, changing only how the test set is
chosen: once splitting by signer, once splitting clips at random. The random split
puts the same people in train and test, so the model can recognise the person instead
of the sign — and the score goes up without the model getting any better.

Run this once on your own dataset. The gap is the amount of self-deception available
to this project, measured in accuracy points.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SignDataset, label_index, load_manifest, split_by_signer
from core.features import FRAME_DIM
from model import SignTransformer


def train_and_score(train, test, labels, epochs, seed, device):
    torch.manual_seed(seed)
    model = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    loader = DataLoader(SignDataset(train, labels, augment=True, rng_seed=seed),
                        batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = criterion(model(x.to(device).float()), y.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(SignDataset(test, labels, augment=False), batch_size=64):
            pred = model(x.to(device).float()).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_manifest(args.manifest)
    labels = label_index(samples)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))

    # ── honest: test signers were never seen ───────────────────────────────
    train_s, _, test_s = split_by_signer(samples, split["val"], split["test"])
    print(f"signer-independent: {len(train_s)} train / {len(test_s)} test clips, "
          f"test signers {sorted({s.signer_id for s in test_s})}")
    honest = train_and_score(train_s, test_s, labels, args.epochs, args.seed, device)

    # ── inflated: same clip pool, shuffled ─────────────────────────────────
    pool = train_s + test_s
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    cut = len(pool) - len(test_s)
    train_r = [pool[i] for i in order[:cut]]
    test_r = [pool[i] for i in order[cut:]]
    print(f"random split:       {len(train_r)} train / {len(test_r)} test clips, "
          f"test signers {sorted({s.signer_id for s in test_r})}")
    inflated = train_and_score(train_r, test_r, labels, args.epochs, args.seed, device)

    print(f"\n  signer-independent accuracy : {honest:.1%}   ← the real number")
    print(f"  random-split accuracy       : {inflated:.1%}")
    print(f"  inflation                   : {inflated - honest:+.1%}")
    print("\nSame model, same data, same training. The only difference is whether the")
    print("test set contains people the model already saw. Report the first number.")


if __name__ == "__main__":
    main()

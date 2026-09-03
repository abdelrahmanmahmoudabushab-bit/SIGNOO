"""Trains the sign classifier.

    python ml/train.py data/processed/manifest.json --split data/split.json

Model selection uses the validation signers; the test signers are never touched here.
Reporting a number means running `evaluate.py` once, at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import SignDataset, describe, label_index, load_manifest, split_by_signer
from features import FRAME_DIM
from model import SignTransformer, count_parameters


def loaders(manifest, split_file, batch_size, mirror_prob, seed):
    samples = load_manifest(manifest)
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    train, val, test = split_by_signer(samples, split["val"], split["test"])

    # Labels come from the full dataset so indices stay stable across runs.
    labels = label_index(samples)

    print(describe(train, "train"))
    print(describe(val, "val"))
    print(f"test: {len(test)} clips held out, not used during training\n")

    train_ds = SignDataset(train, labels, augment=True, mirror_prob=mirror_prob, rng_seed=seed)
    val_ds = SignDataset(val, labels, augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size),
        labels,
    )


def run_epoch(model, loader, criterion, device, optimiser=None):
    training = optimiser is not None
    model.train(training)

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimiser.zero_grad()
                loss.backward()
                # Transformers on small datasets are prone to loss spikes; clipping
                # costs nothing and prevents a run from destroying itself.
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)

    return total_loss / count, correct / count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True, help="JSON with 'val' and 'test' signer lists")
    ap.add_argument("--out", default="models/signoo.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--mirror-prob", type=float, default=0.0,
                    help="probability of mirroring a training clip; see TRAINING.md "
                         "before enabling — it can erase real handedness distinctions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, labels = loaders(
        args.manifest, args.split, args.batch_size, args.mirror_prob, args.seed
    )

    model = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=len(labels)).to(device)
    print(f"{count_parameters(model):,} parameters · {len(labels)} signs · {device}\n")

    # Label smoothing: sign boundaries are genuinely fuzzy and some labels will be
    # wrong, so training the model to be absolutely certain is training it to be wrong.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_acc, best_epoch, history = 0.0, -1, []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimiser)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "labels": labels,
                        "input_dim": FRAME_DIM * 2, "epoch": epoch, "val_acc": val_acc},
                       out)
            marker = "  ← saved"

        if epoch % 5 == 0 or marker:
            print(f"epoch {epoch:3d}  train {train_loss:.3f}/{train_acc:.1%}   "
                  f"val {val_loss:.3f}/{val_acc:.1%}{marker}")

        if epoch - best_epoch >= args.patience:
            print(f"\nno improvement for {args.patience} epochs — stopping")
            break

    print(f"\nbest validation accuracy {best_acc:.1%} at epoch {best_epoch} "
          f"({time.time() - started:.0f}s)")
    print(f"model → {out}")
    Path(out.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))

    if best_acc > 0.98:
        print("\nValidation accuracy this high on a small dataset usually means the "
              "\nsplit is leaking — check that val signers appear nowhere in train.")
    print("\nNow run evaluate.py on the test signers. Once.")


if __name__ == "__main__":
    main()

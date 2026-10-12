"""Trains the sign classifier.

    python ml/train.py data/processed/manifest.json --split data/split.json

Model selection uses the validation signers; the test signers are never touched here.
Reporting a number means running `evaluate.py` once, at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SignDataset, describe, label_index, load_manifest, split_by_signer
from core.features import FEATURE_VERSIONS, FRAME_DIM, INPUT_DIMS
from losses import class_balanced_weights, supervised_contrastive
from model import ARCHITECTURES, count_parameters


def loaders(manifest, split_file, batch_size, mirror_prob, seed, feature_version=1,
            balance=0.0, strong=False):
    samples = load_manifest(manifest)
    if feature_version in (3, 4, 5):
        # Face features (3) and the clip's real duration (4, 5) exist only for clips extracted with
        # --face / the metadata sidecar; the rest cannot be used with these versions.
        usable = (lambda s: s.has_face()) if feature_version == 3 else (lambda s: s.has_meta())
        have = [s for s in samples if usable(s)]
        if len(have) < len(samples):
            print(f"feature version {feature_version}: using {len(have)} of {len(samples)} clips "
                  f"that have the data it needs\n")
        samples = have
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    train, val, test = split_by_signer(samples, split["val"], split["test"])

    # Labels come from the full dataset so indices stay stable across runs.
    labels = label_index(samples)

    print(describe(train, "train"))
    print(describe(val, "val"))
    print(f"test: {len(test)} clips held out, not used during training\n")

    train_ds = SignDataset(train, labels, augment=True, mirror_prob=mirror_prob, rng_seed=seed,
                           feature_version=feature_version, strong=strong)
    val_ds = SignDataset(val, labels, augment=False, feature_version=feature_version)
    if balance > 0:
        # Sample with replacement, rarer words more often (see class_balanced_weights).
        weights = class_balanced_weights([labels[s.sign_id] for s in train], balance)
        sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    return (
        train_loader,
        DataLoader(val_ds, batch_size=batch_size),
        labels,
    )


def run_epoch(model, loader, criterion, device, optimiser=None, supcon=0.0):
    training = optimiser is not None
    model.train(training)

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)

            if training and supcon > 0 and hasattr(model, "embed"):
                embedding = model.embed(x)
                logits = model.classify(embedding)
                loss = criterion(logits, y) + supcon * supervised_contrastive(embedding, y)
            else:
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
    ap.add_argument("--features", type=int, default=1, choices=FEATURE_VERSIONS,
                    help="feature version (core/features.py model_input): 1 = position + velocity, "
                         "2 = + hand shape, 3 = + face (needs clips extracted with --face; the "
                         "kiosk cannot run 3 yet), 4 = 2 + bridged hand dropouts + the sign's real duration, "
                         "5 = 4 with planar (xy-only) hand shape; 4 and 5 need clips with a .meta sidecar. "
                         "Recorded in the checkpoint.")
    ap.add_argument("--arch", default="transformer", choices=sorted(ARCHITECTURES),
                    help="transformer = the original model; signnet = conv stem + attention pooling")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="class-balanced sampling strength, 0 (off) to 1 (fully balanced); 0.5 is typical")
    ap.add_argument("--supcon", type=float, default=0.0,
                    help="weight of the supervised contrastive loss (signnet only); 0.1-0.3 is typical")
    ap.add_argument("--strong-augment", action="store_true",
                    help="also drop a hand for a stretch, rotate a few degrees, zoom slightly")
    ap.add_argument("--pretrained", help="encoder weights from ml/pretrain.py to start from (signnet only)")
    ap.add_argument("--text-embeddings",
                    help="npz from ml/label_embeddings.py: class prototypes from the words' names (signnet only)")
    args = ap.parse_args()
    if args.arch != "signnet" and (args.supcon or args.pretrained or args.text_embeddings):
        ap.error("--supcon, --pretrained and --text-embeddings need --arch signnet")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, labels = loaders(
        args.manifest, args.split, args.batch_size, args.mirror_prob, args.seed, args.features,
        balance=args.balance, strong=args.strong_augment,
    )

    config = {}
    class_text = None
    if args.text_embeddings:
        table = np.load(args.text_embeddings)
        by_id = dict(zip(table["ids"].tolist(), table["vectors"]))
        missing = sorted(set(labels) - set(by_id))
        if missing:
            raise SystemExit(f"--text-embeddings has no vector for: {missing[:10]}")
        class_text = torch.tensor(np.stack([by_id[sign] for sign, _ in sorted(labels.items(), key=lambda kv: kv[1])]))
        config["text_dim"] = class_text.shape[1]
    model = ARCHITECTURES[args.arch](input_dim=INPUT_DIMS[args.features], num_classes=len(labels), **config).to(device)
    if class_text is not None:
        model.class_text.copy_(class_text)
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device, weights_only=True)["encoder"]
        own = model.state_dict()
        usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(usable, strict=False)
        print(f"pretrained: loaded {len(usable)} of {len(own)} tensors from {args.pretrained}")
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
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimiser, args.supcon)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "labels": labels,
                        "input_dim": INPUT_DIMS[args.features], "feature_version": args.features,
                        "arch": args.arch, "config": config,
                        "epoch": epoch, "val_acc": val_acc},
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

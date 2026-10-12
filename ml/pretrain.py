"""Self-supervised pretraining: learn what signing looks like before learning which word is which.

    python ml/pretrain.py data/asl_all/manifest.json --split data/asl_all/split.json \
        --out models/pretrain.pt --features 2

Random stretches of each clip are blanked and the network has to reconstruct them from the rest, the
way a language model fills a gap. That needs no word labels, so it teaches the encoder hand motion,
handshape continuity and body geometry from *every* clip (including words with only a handful), which
is the standard remedy when labelled data per class is thin. Fine-tune with
`ml/train.py --arch signnet --pretrained models/pretrain.pt`.

Only training signers' clips are used: the validation and test signers' landmarks never influence the
encoder, labelled or not, so the held-out numbers stay honest.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import FEATURE_VERSIONS, INPUT_DIMS, SEQ_LEN, model_input
from data import load_manifest, split_by_signer
from model import SignNet


class ClipDataset(Dataset):
    def __init__(self, samples, feature_version):
        self.samples, self.version = samples, feature_version

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        face = s.load_face() if self.version == 3 else None
        duration = s.load_duration() if self.version in (4, 5) else None
        return torch.from_numpy(model_input(s.load(), self.version, face=face, duration=duration)).float()


def mask_spans(x, rng, ratio=0.3, span=4):
    """Blanks ~`ratio` of the frames in runs of `span`; returns (masked input, boolean mask)."""
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    for b in range(len(x)):
        while mask[b].float().mean() < ratio:
            start = int(rng.integers(0, SEQ_LEN - span + 1))
            mask[b, start:start + span] = True
    masked = x.clone()
    masked[mask] = 0.0
    return masked, mask


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", type=int, default=1, choices=FEATURE_VERSIONS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    samples = load_manifest(args.manifest)
    if args.features == 3:
        samples = [s for s in samples if s.has_face()]
    elif args.features in (4, 5):
        samples = [s for s in samples if s.has_meta()]
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train, _, _ = split_by_signer(samples, split["val"], split["test"])
    print(f"pretraining on {len(train)} training-signer clips (no labels used)")

    dim = INPUT_DIMS[args.features]
    model = SignNet(dim, num_classes=1)
    decoder = nn.Linear(model.d_model, dim)
    data = DataLoader(ClipDataset(train, args.features), batch_size=args.batch_size, shuffle=True)

    # Targets are standardised per feature so one loud dimension (a fast-moving wrist) does not
    # dominate the reconstruction loss.
    sample = torch.cat([b for _, b in zip(range(8), data)])
    mean, std = sample.mean((0, 1)), sample.std((0, 1)).clamp_min(1e-3)

    params = list(model.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for x in data:
            masked, mask = mask_spans(x, rng, args.mask_ratio)
            recon = decoder(model.encode(masked))
            loss = ((recon - (x - mean) / std)[mask] ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            total, n = total + loss.item() * len(x), n + len(x)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  reconstruction loss {total / n:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.state_dict(), "input_dim": dim, "feature_version": args.features,
                "loss": total / n}, args.out)
    print(f"encoder -> {args.out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

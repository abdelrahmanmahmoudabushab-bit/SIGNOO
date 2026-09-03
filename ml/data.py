"""Dataset loading and splitting.

Holds the rule the rest of the project depends on: **a signer appears in exactly one
split**. Everything here is built to make violating that rule difficult.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from features import SEQ_LEN, add_velocity, mirror


@dataclass
class Sample:
    """One recorded sign: its features plus who signed it."""
    signer_id: str
    sign_id: str
    path: Path
    handedness: str = "right"

    def load(self) -> np.ndarray:
        return np.load(self.path).astype(np.float32)


def load_manifest(manifest_path) -> list[Sample]:
    """Reads the manifest written by `extract_landmarks.py`."""
    manifest_path = Path(manifest_path)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    return [
        Sample(
            signer_id=r["signer_id"],
            sign_id=r["sign_id"],
            path=root / r["features"],
            handedness=r.get("handedness", "right"),
        )
        for r in records
    ]


def split_by_signer(samples, val_signers, test_signers):
    """Splits by signer, and refuses to return an overlapping split.

    A random split of sign clips puts the same person in train and test, and the
    resulting accuracy mostly measures whether the model recognises that person. It
    is typically 20-40 points higher than the honest number, which is why published
    results are often not reproducible on new signers.
    """
    val, test = set(val_signers), set(test_signers)

    overlap = val & test
    if overlap:
        raise ValueError(f"signers in both val and test: {sorted(overlap)}")

    buckets = {"train": [], "val": [], "test": []}
    for s in samples:
        name = "val" if s.signer_id in val else "test" if s.signer_id in test else "train"
        buckets[name].append(s)

    train_signers = {s.signer_id for s in buckets["train"]}
    leaked = train_signers & (val | test)
    if leaked:                                  # unreachable by construction; a guard
        raise ValueError(f"signer leaked into train: {sorted(leaked)}")

    for name, items in buckets.items():
        if not items:
            raise ValueError(
                f"the '{name}' split is empty — check the signer IDs in your split file"
            )
    return buckets["train"], buckets["val"], buckets["test"]


def label_index(samples) -> dict[str, int]:
    """Stable label→index mapping, sorted so it does not depend on file order."""
    return {sign: i for i, sign in enumerate(sorted({s.sign_id for s in samples}))}


def describe(samples, name="dataset") -> str:
    signers = sorted({s.signer_id for s in samples})
    per_sign = Counter(s.sign_id for s in samples)
    per_signer = Counter(s.signer_id for s in samples)
    rare = [f"{k} ({v})" for k, v in per_sign.most_common()[:-4:-1]]
    return (
        f"{name}: {len(samples)} clips · {len(per_sign)} signs · {len(signers)} signers "
        f"({', '.join(signers[:8])}{'…' if len(signers) > 8 else ''})\n"
        f"  clips per sign: min {min(per_sign.values())}, median "
        f"{int(np.median(list(per_sign.values())))}, max {max(per_sign.values())}\n"
        f"  clips per signer: min {min(per_signer.values())}, max {max(per_signer.values())}\n"
        f"  rarest signs: {', '.join(rare)}"
    )


class SignDataset(Dataset):
    """Feature tensors with light augmentation, applied to the training split only."""

    def __init__(self, samples, labels, augment=False, mirror_prob=0.0, rng_seed=0):
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.mirror_prob = mirror_prob
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        clip = sample.load()

        if self.augment:
            clip = self._augment(clip)

        return add_velocity(clip), self.labels[sample.sign_id]

    def _augment(self, clip):
        """Perturbations that leave the sign's identity intact.

        Deliberately excluded: vertical flips (change meaning), large rotations (a
        sign's orientation is part of it), and time reversal (reverses meaning in
        directional signs, e.g. 'give' vs 'take').
        """
        rng = self.rng

        if rng.random() < self.mirror_prob:
            clip = mirror(clip)

        # Speed: resample to a slightly shorter or longer clip, then back to SEQ_LEN.
        if rng.random() < 0.5:
            factor = rng.uniform(0.8, 1.25)
            n = max(4, int(SEQ_LEN * factor))
            idx = np.linspace(0, len(clip) - 1, n)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, len(clip) - 1)
            frac = (idx - lo)[:, None]
            stretched = clip[lo] * (1 - frac) + clip[hi] * frac
            idx2 = np.linspace(0, len(stretched) - 1, SEQ_LEN)
            lo2 = np.floor(idx2).astype(int)
            hi2 = np.minimum(lo2 + 1, len(stretched) - 1)
            frac2 = (idx2 - lo2)[:, None]
            clip = stretched[lo2] * (1 - frac2) + stretched[hi2] * frac2

        # Landmark jitter, at roughly the scale of real tracker noise.
        clip = clip + rng.normal(0, 0.01, clip.shape).astype(np.float32)

        # Whole-body shift: the signer stands somewhere slightly different.
        clip = clip + rng.normal(0, 0.03, (1, clip.shape[1])).astype(np.float32)

        # Frame dropout, imitating brief tracking failures.
        if rng.random() < 0.3:
            drop = rng.random(len(clip)) < 0.1
            clip[drop] = 0.0

        return clip.astype(np.float32)

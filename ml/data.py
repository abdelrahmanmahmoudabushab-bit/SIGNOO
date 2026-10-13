"""Dataset loading and splitting.

Holds the rule the rest of the project depends on: **a signer appears in exactly one
split**. Everything here is built to make violating that rule difficult.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import FACE_DIM, FRAME_DIM, HAND_DIM, POSE_DIM, SEQ_LEN, mirror, mirror_face, model_input


@dataclass
class Sample:
    """One recorded sign: its features plus who signed it."""
    signer_id: str
    sign_id: str
    path: Path
    handedness: str = "right"

    def load(self) -> np.ndarray:
        return np.load(self.path).astype(np.float32)

    def load_face(self) -> np.ndarray:
        """(SEQ_LEN, FACE_DIM) face features saved beside the clip as `<clip>.face` by
        `extract_landmarks.py --face`. Raises FileNotFoundError for a clip extracted without them."""
        with open(self.path.with_suffix(".face"), "rb") as handle:
            face = np.load(handle).astype(np.float32)
        if face.shape != (len(self.load()), FACE_DIM):
            raise ValueError(f"{self.path.with_suffix('.face')}: face features {face.shape} do not match the clip")
        return face

    def has_face(self) -> bool:
        return self.path.with_suffix(".face").exists()

    def load_meta(self) -> dict:
        """The `<clip>.meta` written by extract_landmarks.py: fps, stride, trimmed span, duration, lineage."""
        return json.loads(self.path.with_suffix(".meta").read_text(encoding="utf-8"))

    def load_duration(self) -> float:
        """The sign's real length in seconds. Raises FileNotFoundError for a clip extracted before metadata
        was recorded: feature versions 4 and 5 cannot use those clips."""
        return float(self.load_meta()["duration_s"])

    def has_meta(self) -> bool:
        return self.path.with_suffix(".meta").exists()


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


def _sample(arr, positions):
    """(T, D) sampled at fractional frame `positions` by linear interpolation; positions outside the
    clip hold its first or last frame (a loosely trimmed clip, not a made-up one)."""
    idx = np.clip(np.asarray(positions, dtype=np.float64), 0, len(arr) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(arr) - 1)
    frac = (idx - lo)[:, None]
    return (arr[lo] * (1 - frac) + arr[hi] * frac).astype(np.float32)


def time_positions(rng, frames=SEQ_LEN, warp=0.25, knots=4, crop=0.08, loosen=0.04):
    """Where to read the source clip so the result is a different *performance* of the same sign.

    The clip always spans the whole sign (it was resampled that way), so a plain stretch changes
    nothing: an earlier version of this augmentation resampled to a new length and straight back,
    which is the identity. What actually varies between performances is (1) pacing within the sign,
    a smooth non-uniform warp (fast preparation, slow stroke, ...), and (2) where the trim fell:
    a little onset cut off, or a little extra rest kept.

    Returns (positions, fraction): the (frames,) source positions, and the share of the original
    duration the result covers (so a duration feature can follow the crop).
    """
    start = rng.uniform(-loosen, crop) if rng.random() < 0.5 else 0.0
    end = 1.0 - (rng.uniform(-loosen, crop) if rng.random() < 0.5 else 0.0)
    speeds = np.exp(rng.normal(0.0, warp, knots))                  # pace of each stretch of the sign
    knot_source = np.concatenate([[0.0], np.cumsum(speeds) / speeds.sum()])
    knot_target = np.linspace(0.0, 1.0, knots + 1)
    warped = np.interp(np.linspace(0.0, 1.0, frames), knot_target, knot_source)   # monotone, 0 -> 1
    return (start + warped * (end - start)) * (SEQ_LEN - 1), float(end - start)


class SignDataset(Dataset):
    """Feature tensors with light augmentation, applied to the training split only."""

    def __init__(self, samples, labels, augment=False, mirror_prob=0.0, rng_seed=0, feature_version=1,
                 strong=False, tta_view=None):
        self.samples = samples
        self.labels = labels
        self.augment = augment
        self.mirror_prob = mirror_prob
        self.feature_version = feature_version
        self.strong = strong
        # Test-time augmentation: view 0/None is the clip as recorded; view k >= 1 is a gentle,
        # *deterministic* re-timing of it (same for a given clip and k), so averaging the model's
        # answers over views is repeatable and never trains on or leaks anything.
        self.tta_view = tta_view
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        clip = sample.load()
        face = sample.load_face() if self.feature_version == 3 else None
        duration = sample.load_duration() if self.feature_version in (4, 5) else None

        if self.augment:
            clip, face, duration = self._augment(clip, face, duration)
        elif self.tta_view:
            rng = np.random.default_rng([int(self.tta_view), i])
            positions, covered = time_positions(rng, warp=0.12, crop=0.03, loosen=0.02)
            clip = _sample(clip, positions)
            face = _sample(face, positions) if face is not None else None
            if duration is not None:
                duration = duration * covered

        return (model_input(clip, self.feature_version, face=face, duration=duration),
                self.labels[sample.sign_id])

    def _augment(self, clip, face=None, duration=None):
        """Perturbations that leave the sign's identity intact.

        Deliberately excluded: vertical flips (change meaning), large rotations (a
        sign's orientation is part of it), and time reversal (reverses meaning in
        directional signs, e.g. 'give' vs 'take').
        """
        rng = self.rng

        if rng.random() < self.mirror_prob:
            clip = mirror(clip)
            if face is not None:
                face = mirror_face(face)

        # Pacing and trim: a different performance of the same sign (see time_positions). The face
        # features are read at the same positions so they stay aligned with the landmarks, and the
        # duration follows the crop plus a signer's own natural speed.
        if rng.random() < 0.7:
            positions, covered = time_positions(rng)
            clip = _sample(clip, positions)
            face = _sample(face, positions) if face is not None else None
            if duration is not None:
                duration = duration * covered * rng.uniform(0.85, 1.15)

        if self.strong:
            clip = self._strong(clip)

        # Landmark jitter, at roughly the scale of real tracker noise.
        clip = clip + rng.normal(0, 0.01, clip.shape).astype(np.float32)

        # Whole-body shift: the signer stands somewhere slightly different.
        clip = clip + rng.normal(0, 0.03, (1, clip.shape[1])).astype(np.float32)

        # Frame dropout, imitating brief tracking failures.
        if rng.random() < 0.3:
            drop = rng.random(len(clip)) < 0.1
            clip[drop] = 0.0
            if face is not None:
                face = face.copy()
                face[drop] = 0.0

        return clip.astype(np.float32), face, duration

    def _strong(self, clip):
        """Heavier, still identity-preserving perturbations (SignNet recipe).

        A hand lost for a stretch of the clip (the commonest real tracking failure, and the reason
        the model must not hang on one hand's every frame); a small in-plane rotation (the camera is
        never exactly level); a small zoom (the signer's distance to the camera). Rotations stay
        within a few degrees: a sign's orientation is part of it.
        """
        rng = self.rng
        clip = np.array(clip, dtype=np.float32, copy=True)
        if rng.random() < 0.25:
            start = rng.integers(0, max(1, len(clip) // 2))
            span = rng.integers(len(clip) // 5, len(clip) * 2 // 5 + 1)
            block = slice(POSE_DIM, POSE_DIM + 1 + HAND_DIM) if rng.random() < 0.5 \
                else slice(POSE_DIM + 1 + HAND_DIM, FRAME_DIM)
            clip[start:start + span, block] = 0.0
        coords = np.ones(clip.shape[1], dtype=bool)
        coords[[POSE_DIM, POSE_DIM + 1 + HAND_DIM]] = False          # presence flags are not coordinates
        triples = clip[:, coords].reshape(len(clip), -1, 3)
        if rng.random() < 0.5:
            angle = float(np.clip(rng.normal(0, 0.06), -0.12, 0.12))
            c, s = np.cos(angle), np.sin(angle)
            x, y = triples[..., 0].copy(), triples[..., 1].copy()
            triples[..., 0], triples[..., 1] = c * x - s * y, s * x + c * y
        if rng.random() < 0.5:
            triples *= rng.uniform(0.92, 1.08)
        clip[:, coords] = triples.reshape(len(clip), -1)
        return clip

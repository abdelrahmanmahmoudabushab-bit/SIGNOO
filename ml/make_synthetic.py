"""Generates a synthetic dataset in the real format, for testing the pipeline.

    python ml/make_synthetic.py --out data/synthetic --signs 20 --signers 12

Each synthetic "sign" is a distinct motion pattern; each synthetic "signer" applies a
consistent personal style — different proportions, speed, and resting posture — on top
of it. That signer style is what makes the difference between a random split and a
signer-independent split visible, exactly as it is with real people.

This is for checking that the code runs and the evaluation is honest. It tells you
nothing about real signing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import FRAME_DIM, HAND_DIM, POSE_DIM, SEQ_LEN


def make_clip(sign, signer, rng, signal=0.6, style_strength=0.25, shared_axes=0):
    """One clip: a sign-specific trajectory rendered in a signer-specific style.

    `signal` and `style_strength` set how much of the clip is the sign versus the
    person. Lowering signal and raising style produces a dataset where signer identity
    competes with sign identity — which is the situation real data is always in, and
    the one that makes a random split flattering.
    """
    style = np.random.default_rng(1000 + signer)
    # Personal constants: body proportions, resting pose, tempo.
    body = style.normal(0, style_strength, FRAME_DIM).astype(np.float32)
    tempo = style.uniform(0.75, 1.35)
    sloppiness = style.uniform(0.02, 0.07)
    # A personal accent applied to the movement itself, not just the resting pose.
    accent = style.normal(0, style_strength, FRAME_DIM).astype(np.float32)

    shape = np.random.default_rng(sign)
    freqs = shape.uniform(0.5, 3.0, 4)
    phases = shape.uniform(0, 2 * np.pi, 4)
    if shared_axes:
        # Signs driving the same axes are harder to tell apart, like real minimal pairs.
        pool = np.random.default_rng(7).choice(FRAME_DIM, size=shared_axes, replace=False)
        axes = shape.choice(pool, size=min(24, shared_axes), replace=False)
    else:
        axes = shape.choice(FRAME_DIM, size=24, replace=False)

    t = np.linspace(0, 1, SEQ_LEN) ** tempo
    clip = np.tile(body, (SEQ_LEN, 1))
    for k, axis in enumerate(axes):
        f, p = freqs[k % 4], phases[k % 4]
        clip[:, axis] += signal * np.sin(2 * np.pi * f * t + p + 0.3 * k)
    clip += accent * np.sin(2 * np.pi * t)[:, None] * 0.5

    clip += rng.normal(0, sloppiness, clip.shape)
    clip[:, POSE_DIM] = 1.0                                   # left hand present
    clip[:, POSE_DIM + 1 + HAND_DIM] = 1.0                    # right hand present
    return clip.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--signs", type=int, default=20)
    ap.add_argument("--signers", type=int, default=12)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal", type=float, default=0.6,
                    help="how strongly the sign shows through (lower = harder)")
    ap.add_argument("--style", type=float, default=0.25,
                    help="how strongly signer identity shows through (higher = harder)")
    ap.add_argument("--shared-axes", type=int, default=0,
                    help="restrict all signs to this many axes, making them overlap")
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    manifest = []

    for signer in range(args.signers):
        signer_id = f"S{signer + 1:02d}"
        for sign in range(args.signs):
            sign_id = f"sign_{sign:03d}"
            for rep in range(args.reps):
                rel = Path(signer_id) / sign_id / f"{signer_id}_{sign_id}_r{rep:02d}.npy"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                np.save(out / rel, make_clip(sign, signer, rng, args.signal,
                                             args.style, args.shared_axes))
                manifest.append({
                    "signer_id": signer_id, "sign_id": sign_id,
                    "label_ar": f"إشارة {sign}", "label_en": f"sign {sign}",
                    "handedness": "right", "source": "synthetic",
                    "features": str(rel),
                })

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    signers = [f"S{i + 1:02d}" for i in range(args.signers)]
    split = {"val": signers[-4:-2], "test": signers[-2:]}
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    print(f"{len(manifest)} clips → {out}")
    print(f"split: val {split['val']}, test {split['test']}")


if __name__ == "__main__":
    main()

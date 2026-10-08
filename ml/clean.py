"""Dataset quality checks.

Run this before every training run. Most sign-recognition failures are dataset
problems wearing a modelling costume, and each check here corresponds to a mistake
that is cheap to fix now and expensive to discover after training.

    python ml/clean.py data/processed/manifest.json --report data/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_manifest
from core.features import HAND_DIM, POSE_DIM

MIN_CLIPS_PER_SIGN = 10
MIN_SIGNERS_PER_SIGN = 3
MIN_HAND_VISIBILITY = 0.6      # fraction of frames with at least one hand tracked
MIN_MOVEMENT = 0.05            # below this a "sign" is a still frame
NEAR_DUPLICATE_DIST = 1e-3


def hand_visibility(clip):
    left = clip[:, POSE_DIM]
    right = clip[:, POSE_DIM + 1 + HAND_DIM]
    return float(np.mean((left > 0.5) | (right > 0.5)))


def movement(clip):
    """Total motion across the clip, in shoulder-width units."""
    return float(np.abs(np.diff(clip, axis=0)).mean() * len(clip))


def check(manifest_path):
    samples = load_manifest(manifest_path)
    issues = defaultdict(list)

    by_sign = defaultdict(list)
    by_signer_sign = Counter()
    for s in samples:
        by_sign[s.sign_id].append(s)
        by_signer_sign[(s.signer_id, s.sign_id)] += 1

    # ── per-clip checks ────────────────────────────────────────────────────
    clips = {}
    for s in samples:
        clip = s.load()
        clips[s.path] = clip
        name = f"{s.signer_id}/{s.sign_id}/{s.path.name}"

        if not np.isfinite(clip).all():
            issues["corrupt"].append(name)
            continue
        if not clip.any():
            issues["all_zero"].append(name)
            continue

        vis = hand_visibility(clip)
        if vis < MIN_HAND_VISIBILITY:
            issues["hands_rarely_tracked"].append(f"{name} ({vis:.0%} of frames)")

        move = movement(clip)
        if move < MIN_MOVEMENT:
            issues["barely_moves"].append(f"{name} (motion {move:.3f})")

    # ── coverage checks ────────────────────────────────────────────────────
    for sign, items in sorted(by_sign.items()):
        signers = {s.signer_id for s in items}
        if len(items) < MIN_CLIPS_PER_SIGN:
            issues["too_few_clips"].append(f"{sign}: {len(items)} clips")
        if len(signers) < MIN_SIGNERS_PER_SIGN:
            issues["too_few_signers"].append(f"{sign}: {len(signers)} signer(s)")

    # Signs recorded by only one person cannot be evaluated signer-independently:
    # whichever split that signer lands in, the sign is missing from the others.
    for sign, items in by_sign.items():
        if len({s.signer_id for s in items}) == 1:
            issues["single_signer_sign"].append(sign)

    # ── near-duplicates ────────────────────────────────────────────────────
    # Identical clips usually mean a file was copied or a recording was saved twice.
    # Left in, they inflate accuracy by putting the same clip in train and test.
    seen = {}
    for s in samples:
        clip = clips[s.path]
        if clip.shape[0] == 0:
            continue
        key = (s.sign_id, round(float(clip.mean()), 6), round(float(clip.std()), 6))
        if key in seen:
            other = clips[seen[key]]
            if other.shape == clip.shape and np.abs(other - clip).mean() < NEAR_DUPLICATE_DIST:
                issues["near_duplicate"].append(f"{s.path.name} ≈ {seen[key].name}")
        else:
            seen[key] = s.path

    # ── balance ────────────────────────────────────────────────────────────
    counts = [len(v) for v in by_sign.values()]
    if counts and max(counts) > 3 * min(counts):
        issues["imbalanced"].append(
            f"most common sign has {max(counts)} clips, rarest has {min(counts)}"
        )

    return samples, by_sign, issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--report", help="write findings to this JSON file")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any issue is found (for CI)")
    args = ap.parse_args()

    samples, by_sign, issues = check(args.manifest)

    signers = sorted({s.signer_id for s in samples})
    print(f"\n{len(samples)} clips · {len(by_sign)} signs · {len(signers)} signers\n")

    if not issues:
        print("✓ no issues found")
    for kind, items in sorted(issues.items()):
        print(f"⚠ {kind}: {len(items)}")
        for item in items[:8]:
            print(f"    {item}")
        if len(items) > 8:
            print(f"    … and {len(items) - 8} more")

    # The checks above are necessary, not sufficient. They cannot see a clip that is
    # cleanly recorded but shows the wrong sign — only a Deaf signer can catch that.
    print("\nThese checks find technical faults, not wrong labels.")
    print("A mislabelled sign passes every one of them. Have a Deaf signer review.")

    if args.report:
        Path(args.report).write_text(
            json.dumps({"clips": len(samples), "signs": len(by_sign),
                        "signers": signers, "issues": dict(issues)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport → {args.report}")

    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

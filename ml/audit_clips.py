"""Scores every extracted clip on tracking quality and says which ones to fix, drop or keep.

    python ml/audit_clips.py data/popsign/processed --csv data/popsign/audit.csv

Read-only: it never moves or deletes a clip. For each `<clip>.npy` (+ its `.meta` and `.face`) it measures

  * how long the sign really is and how many frames the tracker actually saw,
  * how much of the clip each hand is visible, and the longest stretch with no hand at all,
  * jumpiness (a wrist teleporting between frames is a tracking glitch, not signing),
  * whether the face features exist,

and gives each clip a verdict. `bad` clips should be re-extracted (see --list) or kept out of training;
`weak` ones are usable but worth a look. Per-word and per-signer tables show where the damage is
concentrated, which is usually one signer's camera or one word's hand shape, not noise everywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import HAND_DIM, POSE_DIM

LEFT_FLAG = POSE_DIM
RIGHT_FLAG = POSE_DIM + 1 + HAND_DIM
MIN_SECONDS, MAX_SECONDS = 0.4, 6.0         # a real sign is longer than a blink and shorter than a speech
MIN_TRACKED = 12                            # real (not interpolated) frames the tracker saw
JUMP = 1.5                                  # wrist moves > 1.5 shoulder widths in one resampled step: a glitch


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def audit_clip(npy: Path) -> dict:
    clip = np.load(npy).astype(np.float32)
    left, right = clip[:, LEFT_FLAG] > 0.5, clip[:, RIGHT_FLAG] > 0.5
    either = left | right
    # wrist = first landmark of each hand block (flag, then x y z of landmark 0)
    jumps = []
    for flag, start in ((left, LEFT_FLAG + 1), (right, RIGHT_FLAG + 1)):
        wrist = clip[:, start:start + 3]
        both = flag[1:] & flag[:-1]
        if both.any():
            jumps.append(np.linalg.norm(np.diff(wrist, axis=0), axis=1)[both].max())
    meta_path = npy.with_suffix(".meta")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    row = {
        "clip": str(npy), "signer": npy.parent.parent.name, "word": npy.parent.name,
        "seconds": meta.get("duration_s"), "tracked_frames": meta.get("tracked_frames"),
        "hand_share_source": meta.get("hand_share"),
        "left_visible": round(float(left.mean()), 3), "right_visible": round(float(right.mean()), 3),
        "any_hand": round(float(either.mean()), 3), "longest_hand_gap": int(longest_run(~either)),
        "max_wrist_jump": round(float(max(jumps)), 2) if jumps else None,
        "has_face": npy.with_suffix(".face").exists(), "has_meta": bool(meta),
    }
    reasons_bad, reasons_weak = [], []
    if not meta:
        reasons_weak.append("no metadata (cannot use feature v4/v5)")
    if meta:
        if row["seconds"] is not None and not MIN_SECONDS <= row["seconds"] <= MAX_SECONDS:
            reasons_bad.append(f"duration {row['seconds']}s")
        if row["tracked_frames"] is not None and row["tracked_frames"] < MIN_TRACKED:
            reasons_bad.append(f"only {row['tracked_frames']} tracked frames")
    if row["any_hand"] < 0.5:
        reasons_bad.append(f"hands visible in {row['any_hand']:.0%} of frames")
    elif row["any_hand"] < 0.8:
        reasons_weak.append(f"hands visible in {row['any_hand']:.0%} of frames")
    if row["longest_hand_gap"] >= 16:
        reasons_bad.append(f"{row['longest_hand_gap']} frames in a row with no hand")
    if row["max_wrist_jump"] is not None and row["max_wrist_jump"] > JUMP:
        reasons_weak.append(f"wrist jumps {row['max_wrist_jump']} shoulder widths")
    if not row["has_face"]:
        reasons_weak.append("no face features")
    row["verdict"] = "bad" if reasons_bad else "weak" if reasons_weak else "good"
    row["reasons"] = "; ".join(reasons_bad + reasons_weak)
    return row


def summarise(rows: list[dict], key: str, top: int = 12) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    table = sorted(groups.items(), key=lambda kv: -sum(r["verdict"] == "bad" for r in kv[1]) / len(kv[1]))
    lines = [f"{key:12s} {'clips':>6s} {'bad':>6s} {'weak':>6s} {'hands visible':>14s}"]
    for name, items in table[:top]:
        bad = sum(r["verdict"] == "bad" for r in items)
        weak = sum(r["verdict"] == "weak" for r in items)
        lines.append(f"{name:12s} {len(items):6d} {bad / len(items):6.0%} {weak / len(items):6.0%} "
                     f"{np.mean([r['any_hand'] for r in items]):14.0%}")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder of <signer>/<word>/<clip>.npy")
    ap.add_argument("--csv", help="write every clip's scores here")
    ap.add_argument("--list", help="write the paths of the bad clips here, one per line (to re-extract or skip)")
    args = ap.parse_args(argv)

    rows = [audit_clip(p) for p in sorted(Path(args.root).rglob("*.npy"))]
    if not rows:
        raise SystemExit(f"no .npy clips under {args.root}")
    count = {v: sum(r["verdict"] == v for r in rows) for v in ("good", "weak", "bad")}
    print(f"{len(rows)} clips: " + ", ".join(f"{v} {n} ({n / len(rows):.0%})" for v, n in count.items()))
    reasons = defaultdict(int)
    for r in rows:
        for reason in filter(None, r["reasons"].split("; ")):
            reasons[reason.split(" ")[0] if reason[0].isdigit() else reason] += 1
    print("\nmost common problems:")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:5d}  {reason}")
    print("\nworst signers:\n  " + "\n  ".join(summarise(rows, "signer", 8)))
    print("\nworst words:\n  " + "\n  ".join(summarise(rows, "word", 12)))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nscores -> {args.csv}")
    if args.list:
        Path(args.list).write_text("\n".join(r["clip"] for r in rows if r["verdict"] == "bad"), encoding="utf-8")
        print(f"bad clip paths -> {args.list}")


if __name__ == "__main__":
    main()

"""Merges separately extracted landmark datasets into one training manifest.

    python ml/merge_datasets.py --out data/asl_all \
        --source data/popsign/processed --source data/asl_citizen/processed \
        --max-train-per-sign 40

Each source is a tree of <signer>/<sign>/<clip>.npy plus a split.json ({"val": [...],
"test": [...]}) written beside it, and the split is kept per source, so a signer is in exactly
one split and signers from different datasets never share an ID (check_signers refuses it).

Only signs in `core/vocab_asl.json` or `ml/asl_citizen_words.json` are kept. Clips that would
teach noise are dropped: non-finite or all-zero features, and clips where a hand is tracked in
under --min-hands of the frames (a sign recognised from a clip with no hands is a guess).
Training clips are capped per sign, seeded and spread across signers, so a sign with 140 clips
does not drown one with 8. Validation and test clips are never capped or dropped for balance.

Writes manifest.json, split.json and report.json into --out; feature paths are relative to it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.features import HAND_DIM, POSE_DIM


def hand_visibility(clip) -> float:
    left = clip[:, POSE_DIM]
    right = clip[:, POSE_DIM + 1 + HAND_DIM]
    return float(np.mean((left > 0.5) | (right > 0.5)))


def vocabulary(vocab_path: Path, words_path: Path) -> dict[str, dict]:
    """sign id (lower-case) -> {"ar", "en"}."""
    out = {}
    for entry in json.loads(vocab_path.read_text(encoding="utf-8")):
        out[entry["id"].lower()] = {"ar": entry["ar"], "en": entry["en"]}
    for gid, ar in json.loads(words_path.read_text(encoding="utf-8")).items():
        out.setdefault(gid.lower(), {"ar": ar, "en": gid})
    return out


def check_signers(splits: dict[str, dict]) -> None:
    seen = {}
    for src, split in splits.items():
        for signer in split["signers"]:
            if signer in seen and seen[signer] != src:
                raise SystemExit(f"signer {signer} appears in both {seen[signer]} and {src}")
            seen[signer] = src


def load_source(src: Path, vocab: dict, min_hands: float):
    split_file = src / "split.json" if (src / "split.json").exists() else src / "splits.json"
    split = json.loads(split_file.read_text(encoding="utf-8"))
    records, dropped = [], Counter()
    signers = set()
    for path in sorted(src.glob("*/*/*.npy")):
        signer, sign = path.parent.parent.name, path.parent.name.lower()
        if sign not in vocab:
            continue
        clip = np.load(path)
        if not np.isfinite(clip).all() or not clip.any():
            dropped["corrupt_or_empty"] += 1
            continue
        if hand_visibility(clip) < min_hands:
            dropped["hands_rarely_tracked"] += 1
            continue
        signers.add(signer)
        records.append({"signer_id": signer, "sign_id": sign, "path": path})
    split["signers"] = signers
    return split, records, dropped


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=str(ROOT / "core/vocab_asl.json"))
    ap.add_argument("--words", default=str(ROOT / "ml/asl_citizen_words.json"))
    ap.add_argument("--max-train-per-sign", type=int, default=40)
    ap.add_argument("--min-hands", type=float, default=0.3)
    ap.add_argument("--min-clips", type=int, default=0,
                    help="drop signs left with fewer than this many training clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = vocabulary(Path(args.vocab), Path(args.words))

    splits, records, dropped = {}, [], Counter()
    for src in map(Path, args.source):
        split, recs, drop = load_source(src, vocab, args.min_hands)
        splits[str(src)] = split
        dropped.update(drop)
        for r in recs:
            r["root"] = src
        records.extend(recs)
    check_signers(splits)

    val = sorted({s for sp in splits.values() for s in sp["val"]} & {r["signer_id"] for r in records})
    test = sorted({s for sp in splits.values() for s in sp["test"]} & {r["signer_id"] for r in records})
    held = set(val) | set(test)

    rng = random.Random(args.seed)
    by_sign = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["signer_id"] not in held:
            by_sign[r["sign_id"]][r["signer_id"]].append(r)
    keep_train = []
    for sign, signers in by_sign.items():
        order = sorted(signers)
        rng.shuffle(order)
        for clips in signers.values():
            rng.shuffle(clips)
        picked = []
        while len(picked) < args.max_train_per_sign and any(signers.values()):
            for signer in order:
                if signers[signer] and len(picked) < args.max_train_per_sign:
                    picked.append(signers[signer].pop())
        keep_train.extend(picked)
    train_count = Counter(r["sign_id"] for r in keep_train)
    thin = {s for s in {r["sign_id"] for r in records} if train_count[s] < args.min_clips}

    manifest = []
    for r in keep_train + [r for r in records if r["signer_id"] in held]:
        if r["sign_id"] in thin:
            continue
        manifest.append({
            "signer_id": r["signer_id"], "sign_id": r["sign_id"],
            "label_ar": vocab[r["sign_id"]]["ar"], "label_en": vocab[r["sign_id"]]["en"],
            "handedness": "right",
            "source": str(r["path"].relative_to(r["root"])),
            "features": os.path.relpath(r["path"], out).replace("\\", "/"),
        })

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "split.json").write_text(json.dumps({"val": val, "test": test}, indent=1), encoding="utf-8")
    kept = Counter(m["sign_id"] for m in manifest)
    report = {
        "clips": len(manifest), "signs": len(kept), "dropped": dict(dropped),
        "signs_dropped_for_too_few_training_clips": sorted(thin),
        "val_signers": val, "test_signers": test,
        "per_sign_min_median_max": [min(kept.values()), int(np.median(list(kept.values()))), max(kept.values())],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("val_signers", "test_signers")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

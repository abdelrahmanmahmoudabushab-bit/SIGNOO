"""Converts Google's Kaggle `asl-signs` landmark data into this project's training format.

    python ml/import_kaggle_asl.py data/asl --out data/asl_processed --signs 25 --per-sign 150

Reads `train.csv` + `train_landmark_files/*.parquet` (MediaPipe Holistic landmarks,
543 per frame) and writes one (SEQ_LEN, FRAME_DIM) `.npy` per clip through
`core.features`, the same code the kiosk uses at inference, plus a manifest in the
`ml/make_synthetic.py` format and a signer-independent split (participants never
straddle train/val/test).

This is American Sign Language, not Jordanian Sign Language. Its purpose is to
exercise the whole pipeline on real human data and to give a first honest
signer-independent number; it is not a Jordanian model.

Needs pandas + pyarrow, which the kiosk venv does not carry — run from a data-prep venv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import normalise_frame, resample

N_POSE, N_HAND = 33, 21


def _part(frame_rows, kind: str, n: int):
    """(n, 3) array for one body part in one frame, or None if it was not detected."""
    rows = frame_rows[frame_rows["type"] == kind]
    if len(rows) == 0:
        return None
    out = np.full((n, 3), np.nan, dtype=np.float32)
    idx = rows["landmark_index"].to_numpy()
    out[idx] = rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return None if np.isnan(out).all() else np.nan_to_num(out)


def clip_features(parquet_path: Path, swap_hands: bool = False):
    """One Kaggle sequence -> (SEQ_LEN, FRAME_DIM), or None if no hand is ever seen.

    Leading/trailing frames without a hand are trimmed: the kiosk's segmenter only
    starts an utterance when a hand enters the frame, so training on idle lead-in
    would teach the model a distribution it never sees live.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = []
    for _, rows in df.groupby("frame", sort=True):
        pose = _part(rows, "pose", N_POSE)
        left = _part(rows, "left_hand", N_HAND)
        right = _part(rows, "right_hand", N_HAND)
        if swap_hands:
            left, right = right, left
        frames.append((pose, left, right))

    with_hands = [i for i, (_, l, r) in enumerate(frames) if l is not None or r is not None]
    if not with_hands:
        return None
    frames = frames[with_hands[0]: with_hands[-1] + 1]
    return resample([normalise_frame(p, l, r) for p, l, r in frames])


def pick_split(signers: list[str], n_val: int, n_test: int, seed: int) -> dict:
    order = sorted(signers)
    np.random.default_rng(seed).shuffle(order)
    return {"test": sorted(order[:n_test]), "val": sorted(order[n_test:n_test + n_val])}


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="folder holding train.csv and train_landmark_files/")
    ap.add_argument("--out", default="data/asl_processed")
    ap.add_argument("--signs", type=int, default=25, help="use the N most frequent signs")
    ap.add_argument("--per-sign", type=int, default=150, help="max clips per sign (0 = all)")
    ap.add_argument("--val-signers", type=int, default=3)
    ap.add_argument("--test-signers", type=int, default=3)
    ap.add_argument("--swap-hands", action="store_true",
                    help="swap left/right hand blocks if Kaggle's labelling is mirrored vs ours")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    meta = pd.read_csv(src / "train.csv")
    keep = meta["sign"].value_counts().index[: args.signs]
    meta = meta[meta["sign"].isin(keep)]
    if args.per_sign:
        meta = meta.groupby("sign", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.per_sign), random_state=args.seed)
        )

    out.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], 0
    for n, row in enumerate(meta.itertuples(index=False), 1):
        feats = clip_features(src / row.path, args.swap_hands)
        if feats is None:
            skipped += 1
            continue
        rel = Path(str(row.participant_id)) / row.sign / f"{row.sequence_id}.npy"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / rel, feats.astype(np.float32))
        manifest.append({
            "signer_id": str(row.participant_id), "sign_id": row.sign,
            "label_ar": row.sign, "label_en": row.sign, "handedness": "right",
            "source": "kaggle-asl-signs", "features": str(rel).replace("\\", "/"),
        })
        if n % 500 == 0:
            print(f"  {n}/{len(meta)} clips")

    signers = sorted({m["signer_id"] for m in manifest})
    split = pick_split(signers, args.val_signers, args.test_signers, args.seed)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"{len(manifest)} clips · {len(keep)} signs · {len(signers)} signers "
          f"({skipped} skipped, no hand ever visible)\nsplit: {split}")


if __name__ == "__main__":
    main()

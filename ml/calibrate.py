"""Shows what the kiosk would do with held-out clips at each commit threshold.

    python ml/calibrate.py models/asl_pilot.pt data/popsign/processed/manifest.json \
        --split data/popsign/processed/split.json --unknown book drink

Top-1 accuracy does not describe the kiosk. The kiosk commits a word only at or above
a per-language `confident_threshold`, offers candidates to tap between
`reject_threshold` and that, and refuses below it or when the top class is
`__other__` (kiosk/recognition.py, kiosk/replay.py). This prints the share of clips in
each of those outcomes across a range of thresholds, for signs the model knows and,
with --unknown, for signs it was never taught, where a commit is always a false word.

It scores pre-extracted features, so it is the classifier alone: losses in live
tracking and segmentation are not in these numbers (`python -m kiosk.replay` has those).

Choose a threshold with `--on val`. The default scores the test signers, which is for
reporting the chosen bar once; picking the bar there spends the test set. The val
numbers are themselves slightly optimistic: `ml/train.py` kept the checkpoint that
scored best on those same signers, so expect the test table to come out a little worse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OTHER = "__other__"
COMMIT, CANDIDATES, REFUSE = "commit", "candidates", "refuse"
DEFAULT_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95]


def decide(confidence, predicted, confident: float, reject: float, other: int = -1) -> np.ndarray:
    """The kiosk's action per clip, from its top class and that class's probability.

    Refusal is checked first, as in the kiosk: an `__other__` top class is refused
    however confident it is.
    """
    confidence, predicted = np.asarray(confidence, dtype=float), np.asarray(predicted)
    action = np.where(confidence >= confident, COMMIT, CANDIDATES).astype(object)
    action[(confidence < reject) | (predicted == other)] = REFUSE
    return action


def known_shares(confidence, predicted, truth, confident: float, reject: float, other: int = -1) -> dict:
    """Shares of known-sign clips by outcome; the four values sum to 1."""
    action = decide(confidence, predicted, confident, reject, other)
    committed = action == COMMIT
    right = np.asarray(predicted) == np.asarray(truth)
    return {
        "committed_correct": float(np.mean(committed & right)),
        "committed_wrong": float(np.mean(committed & ~right)),
        "candidates": float(np.mean(action == CANDIDATES)),
        "refused": float(np.mean(action == REFUSE)),
    }


def unknown_shares(confidence, predicted, confident: float, reject: float, other: int = -1) -> dict:
    """Shares of unknown-sign clips by outcome. Any commit here is a wrong word shown
    as fact, which the visitor cannot detect; refusal is the only right answer."""
    action = decide(confidence, predicted, confident, reject, other)
    return {
        "false_commit": float(np.mean(action == COMMIT)),
        "candidates": float(np.mean(action == CANDIDATES)),
        "refused": float(np.mean(action == REFUSE)),
    }


def acceptance_score(confidence, predicted, other: int = -1) -> np.ndarray:
    """Confidence as the kiosk uses it: zero when the top class is `__other__`."""
    return np.where(np.asarray(predicted) == other, 0.0, np.asarray(confidence, dtype=float))


def auroc(known_scores, unknown_scores) -> float:
    """Chance that a known-sign clip scores above an unknown-sign clip (ties count half).

    1.0 means some threshold separates them perfectly, 0.5 means confidence carries no
    information about whether the sign is in the vocabulary.
    """
    known = np.asarray(known_scores, dtype=float)[:, None]
    unknown = np.asarray(unknown_scores, dtype=float)[None, :]
    return float(np.mean(known > unknown) + 0.5 * np.mean(known == unknown))


def predict(model, samples, batch_size: int = 64, feature_version: int = 1):
    """Top class and its probability per clip, on the features training used."""
    import torch
    from core.features import model_input

    confidence, predicted = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = np.stack([
                model_input(s.load(), feature_version,
                            duration=s.load_duration() if feature_version in (4, 5) else None)
                for s in samples[start:start + batch_size]])
            probs = torch.softmax(model(torch.from_numpy(batch).float()), dim=1)
            top = probs.max(dim=1)
            confidence.append(top.values.numpy())
            predicted.append(top.indices.numpy())
    return np.concatenate(confidence), np.concatenate(predicted)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--unknown", nargs="*", default=[],
                    help="sign ids in the manifest that the model was not trained on")
    ap.add_argument("--on", choices=("test", "val"), default="test", help="which held-out signers to score")
    ap.add_argument("--reject", type=float, default=0.55, help="the kiosk's reject_threshold")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                    help="confident_threshold values to tabulate")
    args = ap.parse_args(argv)

    import torch
    from data import load_manifest, split_by_signer
    from model import build_model

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]
    other = labels.get(OTHER, -1)

    taught = sorted(set(args.unknown) & set(labels))
    if taught:
        raise SystemExit(f"--unknown lists signs the checkpoint was trained on: {taught}")

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    _, val, test = split_by_signer(load_manifest(args.manifest), split["val"], split["test"])
    held_out = test if args.on == "test" else val

    known = [s for s in held_out if s.sign_id in labels and s.sign_id != OTHER]
    unknown = [s for s in held_out if s.sign_id in set(args.unknown)]
    ignored = sorted({s.sign_id for s in held_out} - {s.sign_id for s in known + unknown})
    if not known:
        raise SystemExit(f"no {args.on} clips of signs the checkpoint knows; is this its manifest and split?")
    absent = sorted(set(args.unknown) - {s.sign_id for s in unknown})
    if absent:
        raise SystemExit(f"--unknown signs with no {args.on} clips in this manifest: {absent}")

    model = build_model(ckpt)

    confidence, predicted = predict(model, known, feature_version=ckpt.get("feature_version", 1))
    truth = np.array([labels[s.sign_id] for s in known])

    print(f"{args.on} signers: {', '.join(sorted({s.signer_id for s in held_out}))}")
    print(f"known: {len(known)} clips of {len({s.sign_id for s in known})} signs · "
          f"top-1 {np.mean(predicted == truth):.1%} · reject_threshold {args.reject}")
    if unknown:
        u_confidence, u_predicted = predict(model, unknown, feature_version=ckpt.get("feature_version", 1))
        print(f"unknown: {len(unknown)} clips of {len({s.sign_id for s in unknown})} signs "
              f"({', '.join(sorted({s.sign_id for s in unknown}))})")
    if ignored:
        print(f"ignored, neither known nor listed in --unknown: {', '.join(ignored)}")

    header = f"\n{'confident':>9} | {'correct':>8} {'WRONG':>7} {'candid.':>8} {'refused':>8}"
    if unknown:
        header += f" | {'FALSE COMMIT':>12} {'candid.':>8} {'refused':>8}"
    print(header)
    print(f"{'':>9} | {'known signs':^34}" + (f" | {'unknown signs':^30}" if unknown else ""))
    for threshold in args.thresholds:
        k = known_shares(confidence, predicted, truth, threshold, args.reject, other)
        row = (f"{threshold:>9.2f} | {k['committed_correct']:>8.1%} {k['committed_wrong']:>7.1%} "
               f"{k['candidates']:>8.1%} {k['refused']:>8.1%}")
        if unknown:
            u = unknown_shares(u_confidence, u_predicted, threshold, args.reject, other)
            row += f" | {u['false_commit']:>12.1%} {u['candidates']:>8.1%} {u['refused']:>8.1%}"
        print(row)

    if unknown:
        area = auroc(acceptance_score(confidence, predicted, other),
                     acceptance_score(u_confidence, u_predicted, other))
        print(f"\nAUROC, known vs unknown by confidence: {area:.3f}  (0.5 = no separation, 1.0 = perfect)")
    if args.on == "test":
        print("\nScored on the test signers: report this, do not pick the threshold from it (use --on val).")


if __name__ == "__main__":
    main()

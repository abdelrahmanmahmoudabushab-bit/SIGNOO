"""Train several configurations on the same data with the same budget and compare them side by side.

    python ml/bench.py data/asl_clean/manifest.json --split data/asl_clean/split.json \
        --text-embeddings data/asl_all/label_text.npz --top-words 50 --epochs 25 \
        --only base,recipe,recipe+ema

Every configuration runs `ml/train.py` in its own process (so one cannot leak memory into the next, and
a crash in one is reported, not fatal), on the *validation* signers only: the test signers are never
touched here. A short run on the best-supplied words answers "does this change help?" in minutes, where
a full run answers it in hours; only the winners need a full run. Differences under about a point on a
small validation set are noise: repeat with --seed before believing one.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ML = Path(__file__).resolve().parent

RECIPE = ["--arch", "signnet", "--features", "2", "--balance", "0.5", "--supcon", "0.2", "--strong-augment",
          "--mirror-prob", "0.5"]

# name -> extra train.py arguments. `{text}` is replaced by --text-embeddings when it is given.
CONFIGS: dict[str, list[str]] = {
    "base": ["--arch", "transformer", "--features", "1", "--mirror-prob", "0.5"],
    "recipe": RECIPE + ["--text-embeddings", "{text}"],
    "recipe-no-text": RECIPE,
    "recipe+ema": RECIPE + ["--text-embeddings", "{text}", "--ema", "0.998"],
    "recipe+ema+warmup": RECIPE + ["--text-embeddings", "{text}", "--ema", "0.998", "--warmup", "3"],
    "v4": ["--arch", "signnet", "--features", "4", "--balance", "0.5", "--supcon", "0.2", "--strong-augment",
           "--mirror-prob", "0.5", "--text-embeddings", "{text}", "--ema", "0.998", "--warmup", "3"],
    "v5": ["--arch", "signnet", "--features", "5", "--balance", "0.5", "--supcon", "0.2", "--strong-augment",
           "--mirror-prob", "0.5", "--text-embeddings", "{text}", "--ema", "0.998", "--warmup", "3"],
    "balance-1": RECIPE[:5] + ["1.0"] + RECIPE[6:] + ["--text-embeddings", "{text}"],
}


def subset_manifest(manifest: Path, top_words: int) -> Path:
    """A manifest of the `top_words` words with the most clips, written next to the original
    (feature paths in a manifest are relative to the manifest's own directory)."""
    records = json.loads(manifest.read_text(encoding="utf-8"))
    counts = Counter(r["sign_id"] for r in records)
    keep = {w for w, _ in counts.most_common(top_words)}
    out = manifest.with_name(f"bench_manifest_{top_words}.json")
    out.write_text(json.dumps([r for r in records if r["sign_id"] in keep], ensure_ascii=False), encoding="utf-8")
    return out


def run_one(name: str, extra: list[str], manifest: Path, split: Path, epochs: int, seed: int,
            text: str | None, workdir: Path) -> dict:
    if any("{text}" in a for a in extra) and not text:
        return {"name": name, "status": "skipped: needs --text-embeddings"}
    args = [a.replace("{text}", text or "") for a in extra]
    out = workdir / f"{name.replace('+', '_')}.pt"
    command = [sys.executable, str(ML / "train.py"), str(manifest), "--split", str(split), "--out", str(out),
               "--epochs", str(epochs), "--patience", str(epochs), "--seed", str(seed), *args]
    started = time.time()
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    seconds = round(time.time() - started)
    match = re.search(r"best validation accuracy ([\d.]+)% at epoch (\d+)", done.stdout)
    if done.returncode != 0 or not match:
        reason = (done.stderr.strip().splitlines() or done.stdout.strip().splitlines() or ["no output"])[-1][:140]
        return {"name": name, "status": f"failed: {reason}", "seconds": seconds}
    return {"name": name, "status": "ok", "val_top1": float(match.group(1)), "best_epoch": int(match.group(2)),
            "seconds": seconds}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--split", required=True)
    ap.add_argument("--text-embeddings")
    ap.add_argument("--top-words", type=int, default=0, help="only the N words with the most clips (0 = all)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="comma-separated configuration names; default: all. Known: " + ", ".join(CONFIGS))
    ap.add_argument("--workdir", default=str(ML.parent / "models" / "bench"))
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.only.split(",")] if args.only else list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; known: {sorted(CONFIGS)}")
    manifest = Path(args.manifest)
    if args.top_words:
        manifest = subset_manifest(manifest, args.top_words)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in names:
        print(f"-- {name} ...", flush=True)
        result = run_one(name, CONFIGS[name], manifest, Path(args.split), args.epochs, args.seed,
                         args.text_embeddings, workdir)
        results.append(result)
        print("   ", json.dumps(result), flush=True)
    (workdir / "bench_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n{'configuration':22s} {'val top-1':>10s} {'best epoch':>11s} {'seconds':>8s}")
    for r in sorted(results, key=lambda r: -r.get("val_top1", -1)):
        if r["status"] == "ok":
            print(f"{r['name']:22s} {r['val_top1']:9.1f}% {r['best_epoch']:11d} {r['seconds']:8d}")
        else:
            print(f"{r['name']:22s} {r['status']}")


if __name__ == "__main__":
    main()

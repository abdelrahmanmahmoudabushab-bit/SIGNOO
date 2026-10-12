"""Unpacks a signer-balanced slice of ASL Citizen into the raw-video layout extract_landmarks.py reads.

    python ml/import_asl_citizen.py data/asl_citizen/ASL_Citizen.zip --out data/asl_citizen/raw --per-sign 8

ASL Citizen (Microsoft Research; 2,731 signs, 83k webcam videos, 52 signers) is licensed for
**non-commercial research use only** (its use.txt). A model trained on it is a prototype and must
not ship on a commercial kiosk without Microsoft's permission (ROADMAP risk R12). Keep that in the
model's notes.

Words come from `ml/asl_citizen_words.json` (id -> Arabic) plus every id already in
`core/vocab_asl.json`, matched to a gloss by exact lower-case name (`ABOUT1`/`ABOUT2` style
variants are never matched, so each id is one sign). Per word, clips are taken round-robin across
signers in a seeded order, so raising --per-sign only ever adds clips to an earlier selection and a
run stopped early still has an even spread of signers.

Output: <out>/<participant>/<id>/<video>.mp4, <out>/splits.json (the dataset's own signer-disjoint
val/test participants) and <out>/selected.json.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_rows(zf: zipfile.ZipFile) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        with zf.open(f"ASL_Citizen/splits/{split}.csv") as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")):
                row["split"] = split
                rows.append(row)
    return rows


def wanted_ids(words_path: Path, vocab_path: Path) -> dict[str, str]:
    """id -> Arabic, from the extension list and the existing vocabulary."""
    words = json.loads(words_path.read_text(encoding="utf-8"))
    for entry in json.loads(vocab_path.read_text(encoding="utf-8")):
        words.setdefault(entry["id"].lower(), entry["ar"])
    return words


def select(rows: list[dict], ids: set[str], per_sign: int, seed: int, train_per_sign: int = 0) -> dict[str, list[dict]]:
    """Per id, up to per_sign clips taken round-robin over signers. Stable in per_sign.

    With `train_per_sign`, each word is then topped up with clips from the dataset's *training*
    signers until it has that many of them. The validation and test signers (a third of the
    signers) are held out of training by design, so a uniform pick spends a third of the clips on
    data the model may never learn from; training clips per word are what accuracy depends on.
    """
    by_gloss = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gid = row["Gloss"].lower()
        if gid in ids:
            by_gloss[gid][row["Participant ID"]].append(row)
    chosen = {}
    for gid, signers in sorted(by_gloss.items()):
        rng = random.Random(f"{seed}:{gid}")
        order = sorted(signers)
        rng.shuffle(order)
        queues = {p: sorted(signers[p], key=lambda r: r["Video file"]) for p in order}
        picked = []
        while len(picked) < per_sign and any(queues.values()):
            for participant in order:
                if queues[participant] and len(picked) < per_sign:
                    picked.append(queues[participant].pop(0))
        if train_per_sign:
            have = {r["Video file"] for r in picked}
            train_signers = [p for p in order if signers[p] and signers[p][0]["split"] == "train"]
            pool = {p: [r for r in sorted(signers[p], key=lambda r: r["Video file"]) if r["Video file"] not in have]
                    for p in train_signers}
            n_train = sum(1 for r in picked if r["split"] == "train")
            while n_train < train_per_sign and any(pool.values()):
                for participant in train_signers:
                    if pool[participant] and n_train < train_per_sign:
                        picked.append(pool[participant].pop(0))
                        n_train += 1
        chosen[gid] = picked
    return chosen


def repair_entry(zip_path: str, info: zipfile.ZipInfo, url: str) -> None:
    """Overwrites one damaged zip entry, in place, with the server's bytes for the same range.

    A resumed multi-hour download of the 46 GB archive left a few entries corrupt although its size
    was right; the offsets of the good bytes never moved, so patching a byte range is enough.
    """
    import struct
    import urllib.request

    def fetch(start, end):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    head = fetch(info.header_offset, info.header_offset + 30)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    end = info.header_offset + 30 + name_len + extra_len + info.compress_size
    with open(zip_path, "r+b") as handle:
        handle.seek(info.header_offset)
        handle.write(fetch(info.header_offset, end))


OTHER = "__other__"      # trained negatives: the model learns "none of my words" (the kiosk refuses it)
UNSEEN = "__unseen__"    # held-out negatives: never trained on, only used to measure false accepts


def select_negatives(rows: list[dict], known: set[str], n_glosses: int, per_gloss: int, holdout: int,
                     seed: int) -> dict[str, list[dict]]:
    """Clips of signs that are *not* in the vocabulary, as {OTHER|UNSEEN: [clip rows]}.

    A closed-vocabulary model must name something for anything it sees; trained on a handful of
    negative signs it only learns to refuse those. This draws hundreds of different out-of-vocabulary
    signs, `holdout` of which are set aside and never trained on, so false accepts can be measured on
    signs the model has genuinely never seen. Only plain alphabetic glosses are eligible (a digit
    marks a variant of a word that might be in the vocabulary).
    """
    by_gloss = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gloss = row["Gloss"]
        if gloss.isalpha() and gloss.lower() not in known:
            by_gloss[gloss][row["Participant ID"]].append(row)
    eligible = sorted(g for g, signers in by_gloss.items() if sum(map(len, signers.values())) >= per_gloss)
    random.Random(f"{seed}:neg").shuffle(eligible)
    chosen = eligible[: n_glosses + holdout]
    unseen_glosses, other_glosses = chosen[:holdout], chosen[holdout:]
    out = {OTHER: [], UNSEEN: []}
    for label, glosses in ((UNSEEN, unseen_glosses), (OTHER, other_glosses)):
        for gloss in glosses:
            rng = random.Random(f"{seed}:{gloss}")
            order = sorted(by_gloss[gloss])
            rng.shuffle(order)
            queues = {p: sorted(by_gloss[gloss][p], key=lambda r: r["Video file"]) for p in order}
            picked = []
            while len(picked) < per_gloss and any(queues.values()):
                for participant in order:
                    if queues[participant] and len(picked) < per_gloss:
                        picked.append(queues[participant].pop(0))
            out[label].extend(picked)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sign", type=int, default=8)
    ap.add_argument("--words", default=str(ROOT / "ml/asl_citizen_words.json"))
    ap.add_argument("--vocab", default=str(ROOT / "core/vocab_asl.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-per-sign", type=int, default=0,
                    help="then top each word up to this many clips from the dataset's training signers")
    ap.add_argument("--limit-words", type=int, default=0, help="only the first N words (0 = all matched)")
    ap.add_argument("--neg-glosses", type=int, default=0,
                    help="also import this many out-of-vocabulary signs as the trained `__other__` class")
    ap.add_argument("--neg-per-gloss", type=int, default=3)
    ap.add_argument("--neg-holdout", type=int, default=0,
                    help="and this many more as `__unseen__`: never trained on, used to measure false accepts")
    ap.add_argument("--repair-url", help="if a zip entry is damaged, re-fetch its bytes from this URL and retry")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    words = wanted_ids(Path(args.words), Path(args.vocab))

    with zipfile.ZipFile(args.zip) as zf:
        rows = read_rows(zf)
        chosen = select(rows, set(words), args.per_sign, args.seed, args.train_per_sign)
        if args.limit_words:
            chosen = dict(list(chosen.items())[: args.limit_words])
        jobs = [(row, gid) for gid, clips in chosen.items() for row in clips]
        negatives = {}
        if args.neg_glosses or args.neg_holdout:
            negatives = select_negatives(rows, set(words), args.neg_glosses, args.neg_per_gloss,
                                         args.neg_holdout, args.seed)
            jobs += [(row, label) for label, clips in negatives.items() for row in clips]
        copied = skipped = repaired = 0
        for row, folder in jobs:
            dest = out / row["Participant ID"] / folder / row["Video file"]
            if dest.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            name = f"ASL_Citizen/videos/{row['Video file']}"
            for attempt in (1, 2):
                # Written aside and renamed, so a failed read never leaves a truncated clip
                # that the next run would treat as done.
                part = dest.with_suffix(".part")
                try:
                    with zf.open(name) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    part.replace(dest)
                    copied += 1
                    break
                except (zlib.error, zipfile.BadZipFile) as exc:
                    part.unlink(missing_ok=True)
                    if attempt == 2 or not args.repair_url:
                        raise SystemExit(f"{name} is damaged in the zip ({exc}); pass --repair-url to re-fetch it")
                    repair_entry(args.zip, zf.getinfo(name), args.repair_url)
                    repaired += 1
        if negatives:
            (out / "negatives.json").write_text(json.dumps(
                {label: sorted({r["Gloss"] for r in clips}) for label, clips in negatives.items()}, indent=1),
                encoding="utf-8")

    split_of = {}
    for row in rows:
        split_of[row["Participant ID"]] = row["split"]
    (out / "splits.json").write_text(json.dumps({
        "val": sorted(p for p, s in split_of.items() if s == "val"),
        "test": sorted(p for p, s in split_of.items() if s == "test"),
    }, indent=2), encoding="utf-8")
    (out / "selected.json").write_text(json.dumps(
        {gid: {"ar": words[gid], "clips": len(clips)} for gid, clips in chosen.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    missing = sorted(set(words) - set(chosen))
    print(f"{len(chosen)} words matched ({len(missing)} of {len(words)} candidates have no gloss of that name)")
    print(f"{copied} clips copied, {skipped} already present, {repaired} damaged entries repaired -> {out}")
    if negatives:
        print({label: f"{len(clips)} clips, {len({r['Gloss'] for r in clips})} signs" for label, clips in negatives.items()})


if __name__ == "__main__":
    main()

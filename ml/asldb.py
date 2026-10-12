"""The ASL data system: one SQLite database that knows every clip, signer, word, licence and model.

    python ml/asldb.py init
    python ml/asldb.py ingest --source popsign --root data/popsign/processed
    python ml/asldb.py ingest --source asl_citizen --root data/asl_citizen/processed
    python ml/asldb.py stats
    python ml/asldb.py check
    python ml/asldb.py export --out data/asl_all --max-train-per-sign 40
    python ml/asldb.py export --out data/asl_clean --licence-clean-only   # shippable data only
    python ml/asldb.py freeze-test                  # test signers can never move to train/val
    python ml/asldb.py review --clip 123 --status confirmed --reviewer NAME
    python ml/asldb.py model-add models/asl.pt --name asl434 --sources popsign,asl_citizen

Why a database and not loose JSON manifests: the rules this project must never break are properties
of the *data*, so they are enforced here, not remembered.

* **A signer is in exactly one split.** The split is a column of the signer, not of the clip, so a
  clip cannot be "random-split" away from its signer (slr-methodology: split by signer, always).
* **Frozen test signers cannot move.** `freeze-test` locks them; a trigger refuses any later change,
  so the benchmark cannot drift into training (DR-7a).
* **Every clip carries its source and every source its licence.** `export --licence-clean-only`
  builds a training set from commercially usable data only; `model-add` records whether a model is
  clean, so a research-only model cannot be mistaken for a shippable one (ROADMAP risk R12).
* **Labels are `unreviewed` until a Deaf reviewer confirms them,** and `export --confirmed-only`
  trains on reviewed clips alone.

The training scripts keep reading the same `manifest.json` / `split.json`; `export` writes them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.features import HAND_DIM, POSE_DIM

DEFAULT_DB = ROOT / "data" / "asl.db"
OTHER, UNSEEN = "__other__", "__unseen__"

# source id -> (name, licence, commercial_ok, url). Add a row here before ingesting a new dataset;
# an unknown source is refused rather than guessed, because the licence decides what may ship.
KNOWN_SOURCES = {
    "popsign": ("PopSign ASL v1.0", "CC BY 4.0", 1, "https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/"),
    "asl_citizen": ("ASL Citizen", "Microsoft Research License (non-commercial research only)", 0,
                    "https://www.microsoft.com/en-us/research/project/asl-citizen/"),
    "kaggle_asl_signs": ("Google ISLR asl-signs (PopSign landmarks)", "CC BY 4.0", 1,
                         "https://www.kaggle.com/competitions/asl-signs"),
    "kiosk": ("SIGNOO kiosk sessions", "own data, consent-based", 1, ""),
}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, licence TEXT NOT NULL,
    commercial_ok INTEGER NOT NULL CHECK (commercial_ok IN (0, 1)), url TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS signers (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL REFERENCES sources(id),
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    frozen INTEGER NOT NULL DEFAULT 0 CHECK (frozen IN (0, 1))
);
CREATE TABLE IF NOT EXISTS words (
    id TEXT PRIMARY KEY, ar TEXT NOT NULL DEFAULT '', en TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'word' CHECK (role IN ('word', 'negative', 'unseen')),
    review_status TEXT NOT NULL DEFAULT 'unreviewed' CHECK (review_status IN ('unreviewed', 'confirmed'))
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL REFERENCES words(id),
    signer TEXT NOT NULL REFERENCES signers(id),
    source TEXT NOT NULL REFERENCES sources(id),
    features_path TEXT NOT NULL UNIQUE,
    face_path TEXT,
    frames INTEGER NOT NULL,
    hand_visibility REAL NOT NULL,
    quality TEXT NOT NULL CHECK (quality IN ('ok', 'poor', 'corrupt')),
    label_status TEXT NOT NULL DEFAULT 'unreviewed' CHECK (label_status IN ('unreviewed', 'confirmed', 'rejected')),
    reviewer TEXT,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS clips_word ON clips(word);
CREATE INDEX IF NOT EXISTS clips_signer ON clips(signer);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
    arch TEXT NOT NULL, feature_version INTEGER NOT NULL, words INTEGER NOT NULL,
    val_acc REAL, test_acc REAL,
    sources TEXT NOT NULL, licence_clean INTEGER NOT NULL CHECK (licence_clean IN (0, 1)),
    created_at TEXT NOT NULL, notes TEXT NOT NULL DEFAULT ''
);
CREATE TRIGGER IF NOT EXISTS signer_frozen_cannot_move BEFORE UPDATE OF split ON signers
WHEN OLD.frozen = 1 AND NEW.split <> OLD.split
BEGIN SELECT RAISE(ABORT, 'signer is frozen in the test set and cannot change split'); END;
CREATE TRIGGER IF NOT EXISTS signer_frozen_cannot_unfreeze BEFORE UPDATE OF frozen ON signers
WHEN OLD.frozen = 1 AND NEW.frozen = 0
BEGIN SELECT RAISE(ABORT, 'a frozen test signer cannot be unfrozen'); END;
CREATE TRIGGER IF NOT EXISTS signer_frozen_cannot_delete BEFORE DELETE ON signers
WHEN OLD.frozen = 1
BEGIN SELECT RAISE(ABORT, 'a frozen test signer cannot be deleted'); END;
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    for sid, (name, licence, ok, url) in KNOWN_SOURCES.items():
        db.execute("INSERT OR IGNORE INTO sources VALUES (?, ?, ?, ?, ?)", (sid, name, licence, ok, url))
    db.commit()


def hand_visibility(clip: np.ndarray) -> float:
    return float(np.mean((clip[:, POSE_DIM] > 0.5) | (clip[:, POSE_DIM + 1 + HAND_DIM] > 0.5)))


def load_words(vocab: Path, extra: Path | None) -> dict[str, tuple[str, str]]:
    """sign id (lower-case) -> (arabic, english)."""
    words = {}
    for entry in json.loads(vocab.read_text(encoding="utf-8")):
        words[entry["id"].lower()] = (entry["ar"], entry["en"])
    if extra and extra.exists():
        for gid, ar in json.loads(extra.read_text(encoding="utf-8")).items():
            words.setdefault(gid.lower(), (ar, gid))
    return words


def ingest(db: sqlite3.Connection, source: str, root: Path, split_file: Path | None = None,
           vocab: Path = ROOT / "core/vocab_asl.json", extra: Path | None = ROOT / "ml/asl_citizen_words.json",
           min_hands: float = 0.3) -> dict:
    """Registers every `<signer>/<word>/<clip>.npy` under `root`. Idempotent: a clip already there is
    refreshed, never duplicated. Words outside the vocabulary are refused unless they are the
    negative classes (`__other__`, `__unseen__`)."""
    if source not in {r["id"] for r in db.execute("SELECT id FROM sources")}:
        raise SystemExit(f"unknown source {source!r}: add its licence to KNOWN_SOURCES first")
    split_file = split_file or (root / "split.json")
    if not split_file.exists():
        split_file = root / "splits.json"
    split = json.loads(split_file.read_text(encoding="utf-8")) if split_file.exists() else {"val": [], "test": []}
    held = {**{s: "val" for s in split.get("val", [])}, **{s: "test" for s in split.get("test", [])}}
    both = set(split.get("val", [])) & set(split.get("test", []))
    if both:
        raise SystemExit(f"signers in both val and test: {sorted(both)}")
    words = load_words(vocab, extra)
    stats = Counter()
    for path in sorted(root.glob("*/*/*.npy")):
        signer, raw_word = path.parent.parent.name, path.parent.name
        word = raw_word if raw_word in (OTHER, UNSEEN) else raw_word.lower()
        if word not in words and word not in (OTHER, UNSEEN):
            stats["skipped_unknown_word"] += 1
            continue
        existing = db.execute("SELECT source, split, frozen FROM signers WHERE id = ?", (signer,)).fetchone()
        if existing and existing["source"] != source:
            raise SystemExit(f"signer id {signer!r} already belongs to source {existing['source']!r}")
        wanted_split = held.get(signer, "train")
        if existing and existing["split"] != wanted_split:
            if existing["frozen"]:
                raise SystemExit(f"signer {signer!r} is frozen as {existing['split']!r}; the split file says {wanted_split!r}")
            db.execute("UPDATE signers SET split = ? WHERE id = ?", (wanted_split, signer))
            stats["signer_moved"] += 1
        if not existing:
            db.execute("INSERT INTO signers (id, source, split) VALUES (?, ?, ?)", (signer, source, wanted_split))
            stats["signers_added"] += 1
        role = "negative" if word == OTHER else "unseen" if word == UNSEEN else "word"
        ar, en = words.get(word, ("", ""))
        db.execute("INSERT OR IGNORE INTO words (id, ar, en, role) VALUES (?, ?, ?, ?)", (word, ar, en, role))
        try:
            clip = np.load(path)
            ok = bool(np.isfinite(clip).all() and clip.any())
            vis = hand_visibility(clip) if ok else 0.0
            frames = int(len(clip))
        except Exception:
            ok, vis, frames = False, 0.0, 0
        quality = "corrupt" if not ok else "poor" if vis < min_hands else "ok"
        face = path.with_suffix(".face")
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        db.execute(
            """INSERT INTO clips (word, signer, source, features_path, face_path, frames, hand_visibility, quality, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(features_path) DO UPDATE SET face_path = excluded.face_path, frames = excluded.frames,
                 hand_visibility = excluded.hand_visibility, quality = excluded.quality""",
            (word, signer, source, rel, os.path.relpath(face, ROOT).replace("\\", "/") if face.exists() else None,
             frames, vis, quality, now()))
        stats[f"clips_{quality}"] += 1
    db.commit()
    return dict(stats)


def freeze_test(db: sqlite3.Connection) -> int:
    cur = db.execute("UPDATE signers SET frozen = 1 WHERE split = 'test' AND frozen = 0")
    db.commit()
    return cur.rowcount


def review(db: sqlite3.Connection, clip_id: int, status: str, reviewer: str) -> None:
    if status not in ("confirmed", "rejected"):
        raise SystemExit("status must be confirmed or rejected")
    if not reviewer.strip():
        raise SystemExit("a review needs the reviewer's name")
    n = db.execute("UPDATE clips SET label_status = ?, reviewer = ? WHERE id = ?", (status, reviewer, clip_id)).rowcount
    if not n:
        raise SystemExit(f"no clip {clip_id}")
    db.commit()


RAW_ROOTS = ("data/asl_citizen/raw", "data/popsign/raw", "data/popsign/raw_cap")
SHEET_COLUMNS = ["clip_id", "word", "arabic", "signer", "source", "video", "decision", "reviewer", "note"]


def _video_for(features_path: str) -> str:
    """The source video of a clip, found beside its landmarks' sign/signer path ('' if it is gone)."""
    parts = Path(features_path).parts
    signer, word, stem = parts[-3], parts[-2], Path(features_path).stem
    for root in RAW_ROOTS:
        candidate = ROOT / root / signer / word / f"{stem}.mp4"
        if candidate.exists():
            return str(candidate)
    return ""


def review_sheet(db: sqlite3.Connection, out: Path, per_word: int = 2, seed: int = 0) -> int:
    """Writes a CSV for a Deaf reviewer: a few unreviewed clips per word, from different signers.

    The reviewer watches `video`, types `confirmed` (the clip shows this word) or `rejected` in
    `decision`, and their name in `reviewer`; `review-import` reads it back. A handful of confirmed
    clips per word is what the model's headline number most needs: until then every label is the
    dataset's say-so.
    """
    import csv
    rng = random.Random(seed)
    rows = []
    for word in db.execute("SELECT id, ar FROM words WHERE role = 'word' ORDER BY id").fetchall():
        clips = db.execute(
            "SELECT id, signer, source, features_path FROM clips WHERE word = ? AND quality = 'ok' "
            "AND label_status = 'unreviewed' ORDER BY id", (word["id"],)).fetchall()
        by_signer = defaultdict(list)
        for c in clips:
            by_signer[c["signer"]].append(c)
        order = sorted(by_signer)
        rng.shuffle(order)
        for signer in order[:per_word]:
            c = by_signer[signer][0]
            rows.append({"clip_id": c["id"], "word": word["id"], "arabic": word["ar"], "signer": signer,
                         "source": c["source"], "video": _video_for(c["features_path"]),
                         "decision": "", "reviewer": "", "note": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as handle:     # BOM so Excel reads Arabic
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def review_import(db: sqlite3.Connection, sheet: Path) -> dict:
    """Applies a filled-in review sheet. Rows with no decision are left alone; a decision without a
    reviewer's name, or one that is not confirmed/rejected, is refused (nothing is applied)."""
    import csv
    with open(sheet, newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["decision"].strip()]
    for r in rows:
        if r["decision"].strip().lower() not in ("confirmed", "rejected"):
            raise SystemExit(f"clip {r['clip_id']}: decision must be confirmed or rejected, not {r['decision']!r}")
        if not r["reviewer"].strip():
            raise SystemExit(f"clip {r['clip_id']}: a decision needs the reviewer's name")
    for r in rows:
        review(db, int(r["clip_id"]), r["decision"].strip().lower(), r["reviewer"].strip())
    return {"applied": len(rows), "confirmed": sum(r["decision"].strip().lower() == "confirmed" for r in rows)}


def stats(db: sqlite3.Connection) -> dict:
    q = lambda sql, *a: db.execute(sql, a).fetchall()  # noqa: E731
    per_word = {r["word"]: r["n"] for r in q(
        "SELECT word, COUNT(*) n FROM clips JOIN signers s ON s.id = clips.signer "
        "WHERE quality = 'ok' AND s.split = 'train' AND word IN (SELECT id FROM words WHERE role = 'word') GROUP BY word")}
    counts = sorted(per_word.values())
    return {
        "sources": [dict(r) for r in q(
            "SELECT s.id, s.licence, s.commercial_ok, COUNT(c.id) clips, COUNT(DISTINCT c.signer) signers "
            "FROM sources s LEFT JOIN clips c ON c.source = s.id GROUP BY s.id")],
        "splits": {r["split"]: {"signers": r["signers"], "clips": r["clips"]} for r in q(
            "SELECT s.split, COUNT(DISTINCT s.id) signers, COUNT(c.id) clips FROM signers s LEFT JOIN clips c ON c.signer = s.id GROUP BY s.split")},
        "quality": {r["quality"]: r["n"] for r in q("SELECT quality, COUNT(*) n FROM clips GROUP BY quality")},
        "words": {r["role"]: r["n"] for r in q("SELECT role, COUNT(*) n FROM words GROUP BY role")},
        "training_clips_per_word": {
            "min": counts[0] if counts else 0, "median": int(np.median(counts)) if counts else 0,
            "max": counts[-1] if counts else 0,
            "words_under_10": sorted(w for w, n in per_word.items() if n < 10)[:40],
            "words_with_none": sorted(r["id"] for r in q(
                "SELECT id FROM words WHERE role = 'word' AND id NOT IN (SELECT word FROM clips c JOIN signers s ON s.id = c.signer "
                "WHERE c.quality = 'ok' AND s.split = 'train')"))[:40]},
        "reviewed": {r["label_status"]: r["n"] for r in q("SELECT label_status, COUNT(*) n FROM clips GROUP BY label_status")},
        "models": [dict(r) for r in q("SELECT name, words, val_acc, test_acc, licence_clean FROM models ORDER BY id")],
    }


def gaps(db: sqlite3.Connection, min_train: int = 20, min_signers: int = 5,
         licence_clean_only: bool = False) -> list[dict]:
    """Everything standing between this data and a trustworthy model, each with its fix.

    Ordered most to least severe. `high` gaps change what the model can honestly claim; `medium` ones
    cost accuracy; `low` ones are hygiene. With `licence_clean_only`, only clips from commercially
    usable sources count, and only the words that have such clips: the audit of what could ship.
    """
    q = lambda sql, *a: db.execute(sql, a).fetchall()  # noqa: E731
    found = []
    if licence_clean_only:
        # A temporary view named `clips` shadows the real table for this connection's queries below.
        db.execute("DROP VIEW IF EXISTS temp.clips")
        db.execute("CREATE TEMP VIEW clips AS SELECT * FROM main.clips WHERE source IN "
                   "(SELECT id FROM sources WHERE commercial_ok = 1)")

    def add(severity, gap, detail, fix):
        found.append({"severity": severity, "gap": gap, "detail": detail, "fix": fix})

    per_word = {r["word"]: (r["clips"], r["signers"]) for r in q(
        "SELECT word, COUNT(*) clips, COUNT(DISTINCT signer) signers FROM clips c JOIN signers s ON s.id = c.signer "
        "WHERE c.quality = 'ok' AND s.split = 'train' AND word IN (SELECT id FROM words WHERE role = 'word') GROUP BY word")}
    scope = " AND id IN (SELECT word FROM clips)" if licence_clean_only else ""
    all_words = [r["id"] for r in q(f"SELECT id FROM words WHERE role = 'word'{scope}")]
    none = sorted(w for w in all_words if w not in per_word)
    if none:
        add("high", "words with no training clip", f"{len(none)}: {', '.join(none[:12])}",
            "extract clips for them, or drop the words from the vocabulary")
    thin = sorted((n, w) for w, (n, _) in per_word.items() if n < min_train)
    if thin:
        add("high", f"words with under {min_train} training clips",
            f"{len(thin)} of {len(all_words)} words; fewest: " + ", ".join(f"{w}={n}" for n, w in thin[:8]),
            "import more training-signer clips (ml/import_asl_citizen.py --train-per-sign), then extract them")
    few_signers = sorted(w for w, (_, s) in per_word.items() if s < min_signers)
    if few_signers:
        add("medium", f"words seen from fewer than {min_signers} training signers",
            f"{len(few_signers)}: {', '.join(few_signers[:10])}", "more signers, not more clips from the same signers")

    def held(split):
        return {r["word"] for r in q("SELECT DISTINCT word FROM clips c JOIN signers s ON s.id = c.signer "
                                     "WHERE c.quality = 'ok' AND s.split = ?", split)}
    for split in ("val", "test"):
        missing = sorted(set(all_words) - held(split))
        if missing:
            add("high" if split == "test" else "medium", f"words with no {split} clip",
                f"{len(missing)} of {len(all_words)} cannot be scored on held-out signers: {', '.join(missing[:10])}",
                f"include a {split} signer who signed them when importing")

    neg = {r["word"]: r["n"] for r in q("SELECT word, COUNT(*) n FROM clips WHERE word IN (?, ?) GROUP BY word", OTHER, UNSEEN)}
    if not neg.get(OTHER):
        add("high", "no negative examples (`__other__`)",
            "the model must name one of its words for anything it sees; earlier it accepted 47% of unknown signs",
            "import out-of-vocabulary signs as __other__ and hold some out as __unseen__")
    elif not neg.get(UNSEEN):
        add("medium", "no held-out unseen signs", "false accepts cannot be measured on signs the model never saw",
            "import __unseen__ signs from glosses not used for __other__")

    unfrozen = q("SELECT COUNT(*) n FROM signers WHERE split = 'test' AND frozen = 0")[0]["n"]
    if unfrozen:
        add("high", "test signers not frozen", f"{unfrozen} test signers could still be moved into training",
            "run: python ml/asldb.py freeze-test")

    reviewed = {r["label_status"]: r["n"] for r in q("SELECT label_status, COUNT(*) n FROM clips GROUP BY label_status")}
    total = sum(reviewed.values()) or 1
    if not reviewed.get("confirmed"):
        add("high", "no label confirmed by a Deaf reviewer",
            "every label is taken on trust from the dataset; a mislabelled clip passes every automatic check",
            "have a Deaf ASL signer review a sample per word (asldb review), then train with --confirmed-only")

    nc = q("SELECT COUNT(*) n FROM clips c JOIN sources s ON s.id = c.source WHERE s.commercial_ok = 0")[0]["n"]
    clean_words = {r["word"] for r in q("SELECT DISTINCT word FROM clips c JOIN sources s ON s.id = c.source "
                                        "JOIN signers sg ON sg.id = c.signer WHERE s.commercial_ok = 1 AND sg.split = 'train' "
                                        "AND c.quality = 'ok'")} & set(all_words)
    if nc:
        add("high", "most of the vocabulary is research-only data",
            f"{nc} of {total} clips are non-commercial; only {len(clean_words)} of {len(all_words)} words have licence-clean "
            "training data", "accept the Kaggle asl-signs rules (CC BY, 250 words) or download more PopSign signs (CC BY)")

    poor = q("SELECT COUNT(*) n FROM clips WHERE quality = 'poor'")[0]["n"]
    if poor:
        add("medium", "clips where hands were barely tracked", f"{poor} clips are excluded from training",
            "re-extract them: ml/extract_landmarks.py --rtmw-fallback")
    face = q("SELECT COUNT(face_path) f, COUNT(*) n FROM clips")[0]
    if face["f"] < face["n"]:
        add("medium", "face features missing", f"{face['f']} of {face['n']} clips have them; feature version 3 needs all",
            "re-run extraction with --face on the remaining clips")

    counts = sorted(n for n, _ in per_word.values())
    if counts and counts[-1] > 3 * int(np.median(counts)) and counts[-1] > 30:
        add("medium", "class imbalance pulls predictions toward the big words",
            f"{counts[0]} to {counts[-1]} training clips per word (median {int(np.median(counts))}); on validation the "
            "commonest wrong answers were the best-supplied words (happy->thankyou 9, please->thankyou 8)",
            "cap at about 1.5x the median with export --max-train-per-sign, and train with --balance 0.5 or more")
    if not q("SELECT 1 FROM models WHERE test_acc IS NOT NULL LIMIT 1"):
        add("low", "no model has a recorded test accuracy", "the held-out test number has not been spent yet",
            "evaluate the final model once with ml/evaluate.py and record it")
    order = {"high": 0, "medium": 1, "low": 2}
    if licence_clean_only:
        db.execute("DROP VIEW IF EXISTS temp.clips")
    return sorted(found, key=lambda g: order[g["severity"]])


def check(db: sqlite3.Connection) -> list[str]:
    """Integrity problems, empty when the data is sound."""
    problems = []
    for r in db.execute("SELECT id, features_path FROM clips"):
        if not (ROOT / r["features_path"]).exists():
            problems.append(f"clip {r['id']}: features file missing ({r['features_path']})")
    for r in db.execute("SELECT id, face_path FROM clips WHERE face_path IS NOT NULL"):
        if not (ROOT / r["face_path"]).exists():
            problems.append(f"clip {r['id']}: face file missing ({r['face_path']})")
    for r in db.execute("SELECT c.id, c.source, s.source AS ssource FROM clips c JOIN signers s ON s.id = c.signer WHERE c.source <> s.source"):
        problems.append(f"clip {r['id']}: clip source {r['source']} differs from its signer's source {r['ssource']}")
    for r in db.execute("SELECT id FROM signers WHERE split = 'test' AND frozen = 0"):
        problems.append(f"test signer {r['id']} is not frozen yet (run freeze-test)")
    leak = db.execute(
        "SELECT word FROM clips c JOIN signers s ON s.id = c.signer WHERE c.word = ? AND s.split = 'train' LIMIT 1", (UNSEEN,)).fetchone()
    if leak:
        problems.append("held-out negative signs (__unseen__) have clips from training signers")
    return problems


def export(db: sqlite3.Connection, out: Path, max_train_per_sign: int = 40, min_hands: float = 0.3,
           licence_clean_only: bool = False, confirmed_only: bool = False, other_cap: int = 200,
           seed: int = 0) -> dict:
    """Writes manifest.json / split.json / report.json in the format ml/train.py reads.

    `manifest.json` has the vocabulary (+ `__other__` if present) and never the held-out negatives;
    `manifest_with_unseen.json` adds `__unseen__`, for ml/calibrate.py to measure false accepts.
    Training clips are capped per word, spread across signers; val/test signers keep everything.
    """
    out.mkdir(parents=True, exist_ok=True)
    where = ["c.quality = 'ok'", "c.hand_visibility >= ?"]
    args: list = [min_hands]
    if licence_clean_only:
        where.append("src.commercial_ok = 1")
    if confirmed_only:
        where.append("c.label_status = 'confirmed'")
    else:
        where.append("c.label_status <> 'rejected'")
    rows = db.execute(
        f"""SELECT c.id, c.word, c.signer, c.features_path, c.source, s.split, w.ar, w.en, w.role
            FROM clips c JOIN signers s ON s.id = c.signer JOIN sources src ON src.id = c.source
            JOIN words w ON w.id = c.word WHERE {' AND '.join(where)} ORDER BY c.id""", args).fetchall()

    rng = random.Random(seed)
    by_word = defaultdict(lambda: defaultdict(list))
    held = []
    for r in rows:
        (held if r["split"] != "train" else by_word[r["word"]][r["signer"]]).append(r)
    kept = list(held)
    for word, signers in by_word.items():
        if word == UNSEEN:
            continue                                         # never trained on
        cap = other_cap if word == OTHER else max_train_per_sign
        order = sorted(signers)
        rng.shuffle(order)
        for clips in signers.values():
            rng.shuffle(clips)
        picked = []
        while len(picked) < cap and any(signers.values()):
            for signer in order:
                if signers[signer] and len(picked) < cap:
                    picked.append(signers[signer].pop())
        kept.extend(picked)

    def record(r):
        feats = os.path.relpath(ROOT / r["features_path"], out).replace("\\", "/")
        return {"signer_id": r["signer"], "sign_id": r["word"],
                "label_ar": r["ar"], "label_en": r["en"], "handedness": "right",
                "source": r["features_path"], "features": feats}

    full = [record(r) for r in sorted(kept, key=lambda r: r["id"])]
    train_manifest = [m for m in full if m["sign_id"] != UNSEEN]
    (out / "manifest.json").write_text(json.dumps(train_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "manifest_with_unseen.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    split = {"val": sorted({r["signer"] for r in kept if r["split"] == "val"}),
             "test": sorted({r["signer"] for r in kept if r["split"] == "test"})}
    (out / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    sources = sorted({r["source"] for r in kept})
    clean = all(db.execute("SELECT commercial_ok FROM sources WHERE id = ?", (s,)).fetchone()[0] for s in sources)
    counts = Counter(m["sign_id"] for m in train_manifest)
    report = {"clips": len(train_manifest), "words": len([w for w in counts if not w.startswith("__")]),
              "has_other_class": OTHER in counts, "unseen_clips": len(full) - len(train_manifest),
              "sources": sources, "licence_clean": bool(clean),
              "label_review": "confirmed only" if confirmed_only else "includes unreviewed labels",
              "val_signers": split["val"], "test_signers": split["test"]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def model_add(db: sqlite3.Connection, path: Path, name: str, sources: list[str], notes: str = "") -> dict:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    labels = [w for w in ckpt["labels"] if not w.startswith("__")]
    known = {r["id"] for r in db.execute("SELECT id FROM words")}
    unknown = sorted(set(labels) - known)
    if unknown:
        raise SystemExit(f"model has words the database does not know: {unknown[:10]}")
    rows = [db.execute("SELECT commercial_ok FROM sources WHERE id = ?", (s,)).fetchone() for s in sources]
    if any(r is None for r in rows):
        raise SystemExit(f"unknown source in {sources}")
    clean = int(all(r[0] for r in rows))
    db.execute(
        "INSERT INTO models (name, path, arch, feature_version, words, val_acc, sources, licence_clean, created_at, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, str(path), ckpt.get("arch", "transformer"), int(ckpt.get("feature_version", 1)), len(labels),
         float(ckpt["val_acc"]) if ckpt.get("val_acc") is not None else None, ",".join(sources), clean, now(), notes))
    db.commit()
    return {"name": name, "words": len(labels), "licence_clean": bool(clean)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("ingest")
    p.add_argument("--source", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split-file")
    p.add_argument("--min-hands", type=float, default=0.3)
    sub.add_parser("stats")
    p = sub.add_parser("gaps", help="audit the data for everything standing between it and a trustworthy model")
    p.add_argument("--min-train", type=int, default=20)
    p.add_argument("--min-signers", type=int, default=5)
    p.add_argument("--licence-clean-only", action="store_true",
                   help="audit only the data that could ship commercially")
    sub.add_parser("check")
    sub.add_parser("freeze-test")
    p = sub.add_parser("export")
    p.add_argument("--out", required=True)
    p.add_argument("--max-train-per-sign", type=int, default=40)
    p.add_argument("--min-hands", type=float, default=0.3)
    p.add_argument("--other-cap", type=int, default=200)
    p.add_argument("--licence-clean-only", action="store_true")
    p.add_argument("--confirmed-only", action="store_true")
    p = sub.add_parser("review-sheet", help="write a CSV of clips for a Deaf reviewer to confirm or reject")
    p.add_argument("--out", required=True)
    p.add_argument("--per-word", type=int, default=2)
    p = sub.add_parser("review-import", help="apply a reviewer's filled-in sheet")
    p.add_argument("sheet")
    p = sub.add_parser("review")
    p.add_argument("--clip", type=int, required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--reviewer", required=True)
    p = sub.add_parser("model-add")
    p.add_argument("path")
    p.add_argument("--name", required=True)
    p.add_argument("--sources", required=True, help="comma-separated sources the model was trained on")
    p.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    db = connect(args.db)
    init(db)
    if args.cmd == "init":
        print(f"database ready: {args.db}")
    elif args.cmd == "ingest":
        print(json.dumps(ingest(db, args.source, Path(args.root), Path(args.split_file) if args.split_file else None,
                                min_hands=args.min_hands)))
    elif args.cmd == "stats":
        print(json.dumps(stats(db), ensure_ascii=False, indent=1))
    elif args.cmd == "gaps":
        for g in gaps(db, args.min_train, args.min_signers, args.licence_clean_only):
            print(f"[{g['severity'].upper():6}] {g['gap']}\n         {g['detail']}\n         fix: {g['fix']}\n")
    elif args.cmd == "check":
        problems = check(db)
        print("\n".join(problems) if problems else "database is consistent")
        sys.exit(1 if problems else 0)
    elif args.cmd == "freeze-test":
        print(f"froze {freeze_test(db)} test signers")
    elif args.cmd == "export":
        print(json.dumps(export(db, Path(args.out), args.max_train_per_sign, args.min_hands, args.licence_clean_only,
                                args.confirmed_only, args.other_cap), ensure_ascii=False))
    elif args.cmd == "review-sheet":
        print(f"{review_sheet(db, Path(args.out), args.per_word)} clips written to {args.out}")
    elif args.cmd == "review-import":
        print(json.dumps(review_import(db, Path(args.sheet))))
    elif args.cmd == "review":
        review(db, args.clip, args.status, args.reviewer)
        print("recorded")
    elif args.cmd == "model-add":
        print(json.dumps(model_add(db, Path(args.path), args.name, args.sources.split(","), args.notes)))


if __name__ == "__main__":
    main()

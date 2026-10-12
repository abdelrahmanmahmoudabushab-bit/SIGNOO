"""The ASL database enforces the data rules the project must never break."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import asldb
from core.features import FRAME_DIM, HAND_DIM, POSE_DIM
from data import load_manifest, split_by_signer
from model import SignTransformer


def _clip(path: Path, hands: bool = True, nan: bool = False) -> None:
    clip = np.random.default_rng(abs(hash(path.name)) % 2**32).normal(size=(64, FRAME_DIM)).astype(np.float32)
    clip[:, POSE_DIM] = 1.0 if hands else 0.0
    clip[:, POSE_DIM + 1 + HAND_DIM] = 1.0 if hands else 0.0
    if nan:
        clip[0, 0] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, clip)


def _tree(root: Path, spec: dict[str, dict[str, int]]) -> Path:
    """{signer: {word: clip count}} -> <root>/<signer>/<word>/<n>.npy"""
    for signer, words in spec.items():
        for word, n in words.items():
            for i in range(n):
                _clip(root / signer / word / f"{signer}-{word}-{i}.npy")
    return root


@pytest.fixture
def db(tmp_path):
    database = asldb.connect(tmp_path / "asl.db")
    asldb.init(database)
    return database


@pytest.fixture
def two_sources(tmp_path, db):
    """PopSign-like (clean licence) and ASL-Citizen-like (research only) data, with a split each."""
    pop = _tree(tmp_path / "pop", {"7001": {"dog": 3, "cat": 3}, "7002": {"dog": 2, "cat": 2}, "7003": {"dog": 1}})
    (pop / "split.json").write_text(json.dumps({"val": ["7002"], "test": ["7003"]}))
    cit = _tree(tmp_path / "cit", {"P1": {"dog": 2, "book": 4}, "P2": {"book": 3}, "P3": {"book": 2, "__unseen__": 2},
                                   "P4": {"__other__": 3}})
    (cit / "splits.json").write_text(json.dumps({"val": ["P2"], "test": ["P3"]}))
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps([{"id": "dog", "ar": "كلب", "en": "dog"}, {"id": "cat", "ar": "قطة", "en": "cat"},
                                 {"id": "book", "ar": "كتاب", "en": "book"}]), encoding="utf-8")
    asldb.ingest(db, "popsign", pop, vocab=vocab, extra=None)
    asldb.ingest(db, "asl_citizen", cit, vocab=vocab, extra=None)
    return pop, cit, vocab


def test_ingest_records_signers_splits_words_quality_and_licences(db, tmp_path, two_sources):
    split = {r["id"]: (r["source"], r["split"]) for r in db.execute("SELECT * FROM signers")}
    assert split["7001"] == ("popsign", "train") and split["7002"] == ("popsign", "val")
    assert split["P3"] == ("asl_citizen", "test") and split["P1"] == ("asl_citizen", "train")
    roles = {r["id"]: r["role"] for r in db.execute("SELECT id, role FROM words")}
    assert roles["dog"] == "word" and roles["__other__"] == "negative" and roles["__unseen__"] == "unseen"
    assert db.execute("SELECT ar FROM words WHERE id = 'dog'").fetchone()[0] == "كلب"
    by_source = {x["id"]: x for x in asldb.stats(db)["sources"]}
    assert by_source["popsign"]["commercial_ok"] == 1 and by_source["asl_citizen"]["commercial_ok"] == 0
    again = asldb.ingest(db, "popsign", tmp_path / "pop", vocab=two_sources[2], extra=None)   # idempotent
    assert again["clips_ok"] == 11
    assert db.execute("SELECT COUNT(*) FROM clips WHERE source = 'popsign'").fetchone()[0] == 11


def test_clips_without_hands_or_with_nan_are_flagged_not_trusted(db, tmp_path):
    root = tmp_path / "x"
    _clip(root / "7001" / "dog" / "a.npy")
    _clip(root / "7001" / "dog" / "b.npy", hands=False)
    _clip(root / "7001" / "dog" / "c.npy", nan=True)
    vocab = tmp_path / "v.json"
    vocab.write_text(json.dumps([{"id": "dog", "ar": "", "en": "dog"}]))
    out = asldb.ingest(db, "popsign", root, vocab=vocab, extra=None)
    assert out == {"signers_added": 1, "clips_ok": 1, "clips_poor": 1, "clips_corrupt": 1}


def test_refuses_a_signer_in_two_splits_and_a_signer_id_reused_across_sources(db, tmp_path):
    root = _tree(tmp_path / "a", {"7001": {"dog": 1}})
    (root / "split.json").write_text(json.dumps({"val": ["7001"], "test": ["7001"]}))
    vocab = tmp_path / "v.json"
    vocab.write_text(json.dumps([{"id": "dog", "ar": "", "en": "dog"}]))
    with pytest.raises(SystemExit, match="both val and test"):
        asldb.ingest(db, "popsign", root, vocab=vocab, extra=None)
    (root / "split.json").write_text(json.dumps({"val": [], "test": []}))
    asldb.ingest(db, "popsign", root, vocab=vocab, extra=None)
    other = _tree(tmp_path / "b", {"7001": {"dog": 1}})
    with pytest.raises(SystemExit, match="already belongs"):
        asldb.ingest(db, "asl_citizen", other, vocab=vocab, extra=None)
    with pytest.raises(SystemExit, match="unknown source"):
        asldb.ingest(db, "mystery", other, vocab=vocab, extra=None)


def test_frozen_test_signers_can_never_move(db, tmp_path, two_sources):
    assert asldb.freeze_test(db) == 2                              # 7003 and P3
    with pytest.raises(sqlite3.IntegrityError, match="frozen"):
        db.execute("UPDATE signers SET split = 'train' WHERE id = '7003'")
    with pytest.raises(sqlite3.IntegrityError, match="unfrozen"):
        db.execute("UPDATE signers SET frozen = 0 WHERE id = '7003'")
    with pytest.raises(sqlite3.IntegrityError, match="deleted"):
        db.execute("DELETE FROM signers WHERE id = '7003'")
    pop, _, vocab = two_sources                                    # a later split file cannot quietly move them
    (pop / "split.json").write_text(json.dumps({"val": ["7002"], "test": []}))
    with pytest.raises(SystemExit, match="frozen"):
        asldb.ingest(db, "popsign", pop, vocab=vocab, extra=None)


def test_export_is_signer_independent_capped_and_readable_by_the_training_code(db, tmp_path, two_sources):
    out = tmp_path / "export"
    report = asldb.export(db, out, max_train_per_sign=2)
    samples = load_manifest(out / "manifest.json")
    split = json.loads((out / "split.json").read_text())
    train, val, test = split_by_signer(samples, split["val"], split["test"])   # raises on any overlap
    assert {s.signer_id for s in train}.isdisjoint({s.signer_id for s in val + test})
    per_word = Counter(s.sign_id for s in train)
    assert max(v for k, v in per_word.items() if not k.startswith("__")) <= 2   # capped
    assert all(s.path.exists() for s in samples)
    assert "__unseen__" not in {s.sign_id for s in samples}                     # never trained on ...
    assert report["unseen_clips"] > 0                                            # ... but kept for measuring false accepts
    with_unseen = {m["sign_id"] for m in json.loads((out / "manifest_with_unseen.json").read_text())}
    assert "__unseen__" in with_unseen and report["has_other_class"] is True


def test_licence_clean_export_drops_research_only_data_and_says_so(db, tmp_path, two_sources):
    everything = asldb.export(db, tmp_path / "all")
    clean = asldb.export(db, tmp_path / "clean", licence_clean_only=True)
    assert everything["licence_clean"] is False and set(everything["sources"]) == {"popsign", "asl_citizen"}
    assert clean["licence_clean"] is True and clean["sources"] == ["popsign"]
    words = {m["sign_id"] for m in json.loads((tmp_path / "clean/manifest.json").read_text())}
    assert words == {"dog", "cat"}                                               # `book` only exists in the research-only set


def test_confirmed_only_export_uses_only_clips_a_reviewer_confirmed(db, tmp_path, two_sources):
    ids = [r[0] for r in db.execute("SELECT id FROM clips WHERE source = 'popsign' ORDER BY id")]
    with pytest.raises(SystemExit):
        asldb.review(db, ids[0], "confirmed", "  ")                               # a review needs a name
    asldb.review(db, ids[0], "confirmed", "Deaf reviewer 1")
    report = asldb.export(db, tmp_path / "rev", confirmed_only=True)
    assert report["clips"] == 1 and report["label_review"] == "confirmed only"
    before = asldb.export(db, tmp_path / "before")["clips"]
    asldb.review(db, ids[1], "rejected", "Deaf reviewer 1")
    assert asldb.export(db, tmp_path / "after")["clips"] == before - 1           # a rejected clip never trains


def test_review_sheet_round_trip_and_refusals(db, tmp_path, two_sources):
    import csv
    sheet = tmp_path / "sheet.csv"
    n = asldb.review_sheet(db, sheet, per_word=2)
    with open(sheet, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert n == len(rows) and set(rows[0]) == set(asldb.SHEET_COLUMNS)
    by_word = {}
    for r in rows:
        by_word.setdefault(r["word"], []).append(r["signer"])
    assert all(len(s) == len(set(s)) <= 2 for s in by_word.values())             # different signers, at most 2
    assert next(r["arabic"] for r in rows if r["word"] == "dog") == "كلب"

    rows[0].update(decision="confirmed", reviewer="Reviewer A")
    rows[1].update(decision="rejected", reviewer="Reviewer A")
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=asldb.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert asldb.review_import(db, sheet) == {"applied": 2, "confirmed": 1}
    status = {r["id"]: r["label_status"] for r in db.execute("SELECT id, label_status FROM clips")}
    assert status[int(rows[0]["clip_id"])] == "confirmed" and status[int(rows[1]["clip_id"])] == "rejected"
    asldb.review_sheet(db, tmp_path / "again.csv", per_word=2)
    with open(tmp_path / "again.csv", newline="", encoding="utf-8-sig") as f:
        asked_again = {r["clip_id"] for r in csv.DictReader(f)}
    assert not asked_again & {rows[0]["clip_id"], rows[1]["clip_id"]}               # reviewed clips are not asked twice

    rows[2].update(decision="maybe", reviewer="Reviewer A")                         # not a valid decision
    rows[3].update(decision="confirmed", reviewer="")                               # no name
    for bad in (2, 3):
        with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=asldb.SHEET_COLUMNS)
            w.writeheader()
            w.writerows([rows[bad]])
        with pytest.raises(SystemExit):
            asldb.review_import(db, sheet)
    assert db.execute("SELECT label_status FROM clips WHERE id = ?", (int(rows[3]["clip_id"]),)).fetchone()[0] == "unreviewed"


def test_the_shippable_audit_ignores_research_only_data(db, tmp_path, two_sources):
    everything = {g["gap"] for g in asldb.gaps(db, min_train=3)}
    clean = asldb.gaps(db, min_train=3, licence_clean_only=True)
    assert "most of the vocabulary is research-only data" in everything
    assert "most of the vocabulary is research-only data" not in {g["gap"] for g in clean}   # none of it counts here
    detail = " ".join(g["detail"] for g in clean)
    assert "book" not in detail                       # `book` exists only in the research-only source
    # The shadowing view is gone afterwards, so a later full audit still sees every clip.
    assert db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == db.execute("SELECT COUNT(*) FROM main.clips").fetchone()[0]


def test_check_finds_missing_files_and_unfrozen_test_signers(db, tmp_path, two_sources):
    assert any("not frozen" in p for p in asldb.check(db))
    asldb.freeze_test(db)
    assert asldb.check(db) == []
    victim = db.execute("SELECT features_path FROM clips LIMIT 1").fetchone()[0]
    (asldb.ROOT / victim).unlink()
    assert any("features file missing" in p for p in asldb.check(db))


def test_a_model_trained_on_research_only_data_is_recorded_as_not_shippable(db, tmp_path, two_sources):
    net = SignTransformer(input_dim=FRAME_DIM * 2, num_classes=3)
    path = tmp_path / "m.pt"
    torch.save({"model": net.state_dict(), "labels": {"book": 0, "cat": 1, "dog": 2}, "input_dim": FRAME_DIM * 2,
                "val_acc": 0.5}, path)
    assert asldb.model_add(db, path, "research", ["popsign", "asl_citizen"])["licence_clean"] is False
    assert asldb.model_add(db, path, "clean", ["popsign"])["licence_clean"] is True
    torch.save({"model": net.state_dict(), "labels": {"book": 0, "cat": 1, "zzz": 2}, "input_dim": FRAME_DIM * 2}, path)
    with pytest.raises(SystemExit, match="does not know"):
        asldb.model_add(db, path, "bad", ["popsign"])
    assert [m["name"] for m in asldb.stats(db)["models"]] == ["research", "clean"]

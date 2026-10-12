"""Which clips the ASL Citizen importer picks: positives, training-signer top-ups, and negatives."""
from __future__ import annotations

from collections import Counter

import import_asl_citizen as imp


def _rows(glosses, signers=("P1", "P2", "P3", "P4", "P5", "P6"), split_of=None, per_signer=1):
    split_of = split_of or {"P1": "train", "P2": "train", "P3": "train", "P4": "train", "P5": "val", "P6": "test"}
    rows = []
    for gloss in glosses:
        for signer in signers:
            for i in range(per_signer):
                rows.append({"Participant ID": signer, "Video file": f"{signer}-{gloss}-{i}.mp4",
                             "Gloss": gloss, "split": split_of[signer]})
    return rows


def test_select_spreads_each_word_across_signers_and_is_stable_as_per_sign_grows():
    rows = _rows(["DOG", "CAT"])
    small = imp.select(rows, {"dog", "cat"}, per_sign=3, seed=0)
    large = imp.select(rows, {"dog", "cat"}, per_sign=5, seed=0)
    assert {len(v) for v in small.values()} == {3}
    assert all(len({r["Participant ID"] for r in clips}) == len(clips) for clips in small.values())   # one per signer
    for word in small:                                       # raising per_sign only ever adds clips
        assert {r["Video file"] for r in small[word]} <= {r["Video file"] for r in large[word]}


def test_train_per_sign_tops_up_with_training_signers_only():
    rows = _rows(["DOG"], per_signer=3)
    base = imp.select(rows, {"dog"}, per_sign=4, seed=0)["dog"]
    topped = imp.select(rows, {"dog"}, per_sign=4, seed=0, train_per_sign=10)["dog"]
    assert {r["Video file"] for r in base} <= {r["Video file"] for r in topped}
    extra = [r for r in topped if r["Video file"] not in {b["Video file"] for b in base}]
    assert extra and all(r["split"] == "train" for r in extra)                 # held-out signers gain nothing
    assert sum(r["split"] == "train" for r in topped) == 10
    assert len({r["Video file"] for r in topped}) == len(topped)                # no clip twice


def test_negatives_are_out_of_vocabulary_plain_glosses_and_the_holdout_is_disjoint():
    plain = ["SIGN" + c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    rows = _rows(["DOG", "ABOUT1", "ABOUT2"] + plain)
    out = imp.select_negatives(rows, known={"dog"}, n_glosses=8, per_gloss=3, holdout=4, seed=0)
    trained = {r["Gloss"] for r in out[imp.OTHER]}
    unseen = {r["Gloss"] for r in out[imp.UNSEEN]}
    assert len(trained) == 8 and len(unseen) == 4
    assert trained.isdisjoint(unseen)                                          # unseen signs are really unseen
    assert "DOG" not in trained | unseen                                       # never a word we teach
    assert not any(any(c.isdigit() for c in g) for g in trained | unseen)      # variants of vocabulary words excluded
    assert set(Counter(r["Gloss"] for r in out[imp.OTHER]).values()) == {3}
    again = imp.select_negatives(rows, {"dog"}, 8, 3, 4, seed=0)
    assert {r["Video file"] for r in again[imp.OTHER]} == {r["Video file"] for r in out[imp.OTHER]}   # deterministic


def test_a_gloss_with_too_few_clips_is_not_a_negative():
    rows = _rows(["RARE"], signers=("P1", "P2"))                                # only 2 clips
    out = imp.select_negatives(rows, known=set(), n_glosses=1, per_gloss=3, holdout=0, seed=0)
    assert out[imp.OTHER] == [] and out[imp.UNSEEN] == []

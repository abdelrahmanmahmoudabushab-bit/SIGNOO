"""The split chooser, the open-set builder and idle trimming: the pure parts of the
dataset pipeline, where a silent mistake would corrupt every number downstream."""
import numpy as np
import pytest

from extract_landmarks import LEFT_FLAG, RIGHT_FLAG, trim_idle
from make_open_set import OTHER, build
from make_split import choose

from core.features import FRAME_DIM, HAND_DIM, POSE_DIM


# ── make_split.choose ─────────────────────────────────────────────────────────

def covered(group, signs_by_signer):
    return set().union(*(signs_by_signer[s] for s in group))


def test_no_signer_lands_in_two_splits_and_some_are_left_to_train_on():
    signs_by_signer = {f"s{i}": {"dog", "cat"} for i in range(8)}

    val, test = choose(signs_by_signer, n_val=2, n_test=3, seed=0)

    assert len(val) == 2 and len(test) == 3
    assert not set(val) & set(test)
    assert len(set(signs_by_signer) - set(val) - set(test)) == 3


def test_val_and_test_each_cover_every_sign_even_when_most_signers_do_not():
    # Six signers did only one sign each; a careless draw of two leaves a sign out.
    signs_by_signer = {f"dog{i}": {"dog"} for i in range(3)}
    signs_by_signer |= {f"cat{i}": {"cat"} for i in range(3)}
    signs_by_signer |= {"both0": {"dog", "cat"}, "both1": {"dog", "cat"}}

    for seed in range(5):
        val, test = choose(signs_by_signer, n_val=2, n_test=2, seed=seed)
        assert covered(val, signs_by_signer) == {"dog", "cat"}
        assert covered(test, signs_by_signer) == {"dog", "cat"}
        assert not set(val) & set(test)


def test_the_choice_is_repeatable_for_a_seed():
    signs_by_signer = {f"s{i}": {"dog", "cat"} for i in range(8)}
    assert choose(signs_by_signer, 2, 2, seed=3) == choose(signs_by_signer, 2, 2, seed=3)


def test_too_few_signers_to_hold_out_and_still_train_is_an_error():
    signs_by_signer = {f"s{i}": {"dog"} for i in range(4)}
    with pytest.raises(SystemExit):
        choose(signs_by_signer, n_val=2, n_test=2, seed=0)


def test_a_sign_only_one_signer_made_cannot_be_in_both_val_and_test():
    signs_by_signer = {f"s{i}": {"dog"} for i in range(6)}
    signs_by_signer["only"] = {"dog", "cat"}
    with pytest.raises(SystemExit):
        choose(signs_by_signer, n_val=2, n_test=2, seed=0, tries=50)


# ── make_open_set.build ───────────────────────────────────────────────────────

def record(signer, sign, n=0):
    return {"signer_id": signer, "sign_id": sign, "label_ar": "ع", "label_en": sign,
            "features": f"{signer}/{sign}/{n}.npy"}


def open_set_records():
    records = [record("s1", "dog"), record("s2", "dog"), record("s2", "cat")]
    records += [record("s1", "bird", n) for n in range(6)]          # one signer has most of the "other" clips
    records += [record("s2", "hat"), record("s3", "hat")]
    records += [record("s1", "book"), record("s3", "book")]         # in neither list
    return records


def test_known_signs_are_kept_as_they_are_and_unlisted_signs_are_dropped():
    out = build(open_set_records(), {"dog", "cat"}, {"bird", "hat"}, other_max=3, seed=0)

    assert sorted(r["sign_id"] for r in out if r["sign_id"] != OTHER) == ["cat", "dog", "dog"]
    assert all(r["label_en"] == r["sign_id"] for r in out if r["sign_id"] != OTHER)
    assert not any("book" in r["features"] for r in out)


def test_other_signs_are_relabelled_capped_and_spread_across_signers():
    out = build(open_set_records(), {"dog", "cat"}, {"bird", "hat"}, other_max=3, seed=0)
    other = [r for r in out if r["sign_id"] == OTHER]

    assert len(other) == 3
    # s1 holds six of the eight candidates; an even draw still takes one from each signer.
    assert sorted(r["signer_id"] for r in other) == ["s1", "s2", "s3"]
    assert all(r["label_en"] == "" and r["label_ar"] == "" for r in other)
    assert all("bird" in r["features"] or "hat" in r["features"] for r in other)


def test_a_cap_above_the_pool_takes_the_whole_pool_and_the_input_is_untouched():
    records = open_set_records()
    out = build(records, {"dog", "cat"}, {"bird", "hat"}, other_max=100, seed=0)

    assert sum(r["sign_id"] == OTHER for r in out) == 8
    assert not any(r["sign_id"] == OTHER for r in records)


# ── extract_landmarks.trim_idle ───────────────────────────────────────────────

def test_the_flags_trim_idle_reads_are_the_hand_presence_slots():
    assert (LEFT_FLAG, RIGHT_FLAG) == (POSE_DIM, POSE_DIM + 1 + HAND_DIM)


def test_frames_before_the_first_hand_and_after_the_last_are_removed():
    frames = np.zeros((8, FRAME_DIM), dtype=np.float32)
    frames[:, 0] = np.arange(8)              # the body is tracked throughout; marks each frame
    frames[2, LEFT_FLAG] = 1.0
    frames[5, RIGHT_FLAG] = 1.0              # either hand counts; the gap between is kept

    trimmed = trim_idle(frames)

    assert trimmed[:, 0].tolist() == [2, 3, 4, 5]


def test_tracking_rate_follows_time_not_frame_count():
    from extract_landmarks import effective_stride, idle_bounds
    # `--stride 2` meant 60 tracked fps on a 120 fps phone clip and 15 on a 30 fps webcam clip.
    assert effective_stride(120, stride=2) == 2
    assert effective_stride(120, stride=2, target_fps=15) == 8
    assert effective_stride(30, stride=1, target_fps=15) == 2
    assert effective_stride(29.75, target_fps=15) == 2
    assert effective_stride(60, target_fps=15) == 4
    assert effective_stride(10, target_fps=15) == 1                  # never below one frame
    assert effective_stride(30, stride=3) == 3 and effective_stride(30) == 1
    frames = np.zeros((8, FRAME_DIM), dtype=np.float32)
    frames[2:5, RIGHT_FLAG] = 1.0
    assert idle_bounds(frames) == (2, 5) and idle_bounds(np.zeros((4, FRAME_DIM))) is None


def test_a_clip_where_no_hand_is_ever_seen_gives_none():
    frames = np.ones((8, FRAME_DIM), dtype=np.float32)
    frames[:, [LEFT_FLAG, RIGHT_FLAG]] = 0.0
    assert trim_idle(frames) is None
    assert trim_idle(list(frames)) is None   # the extractor passes a list of frames

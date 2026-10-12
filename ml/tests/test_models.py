"""SignNet, Ensemble, build_model, the contrastive loss, balanced sampling, and the new train.py flags."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from core.features import FACE_DIM, FRAME_DIM, INPUT_DIMS, SEQ_LEN, mirror_face
from losses import class_balanced_weights, supervised_contrastive
from model import ARCHITECTURES, Ensemble, SignNet, SignTransformer, build_model

ML = Path(__file__).resolve().parent.parent
DIM = INPUT_DIMS[1]


def _ckpt(arch="signnet", classes=5, **config):
    net = ARCHITECTURES[arch](input_dim=DIM, num_classes=classes, **config)
    return {"model": net.state_dict(), "labels": {f"w{i}": i for i in range(classes)},
            "input_dim": DIM, "arch": arch, "config": config}


def test_signnet_shapes_and_embedding():
    net = SignNet(DIM, 7).eval()
    x = torch.randn(3, SEQ_LEN, DIM)
    assert net(x).shape == (3, 7)
    assert net.embed(x).shape == (3, net.d_model)


def test_build_model_round_trips_both_architectures_and_old_checkpoints():
    x = torch.randn(2, SEQ_LEN, DIM)
    for arch in ("transformer", "signnet"):
        ckpt = _ckpt(arch)
        model = build_model(ckpt)
        assert not model.training and model(x).shape == (2, 5)
    old = {k: v for k, v in _ckpt("transformer").items() if k not in ("arch", "config")}
    assert isinstance(build_model(old), SignTransformer)       # a pre-2026-10-12 checkpoint
    with pytest.raises(ValueError):
        build_model({**_ckpt(), "arch": "nope"})


def test_text_head_scores_by_cosine_to_class_prototypes():
    config = {"text_dim": 16}
    net = SignNet(DIM, 4, **config).eval()
    prototypes = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    net.class_text.copy_(prototypes)
    logits = net(torch.randn(2, SEQ_LEN, DIM))
    assert logits.shape == (2, 4)
    bound = net.logit_scale.item() + 1e-4
    assert logits.abs().max() <= bound                          # cosine * scale cannot exceed the scale
    rebuilt = build_model({"model": net.state_dict(), "labels": {f"w{i}": i for i in range(4)},
                           "input_dim": DIM, "arch": "signnet", "config": config})
    torch.testing.assert_close(rebuilt.class_text, prototypes)  # the prototypes travel in the checkpoint


def test_ensemble_returns_the_mean_of_member_probabilities():
    torch.manual_seed(0)
    members = [build_model(_ckpt("signnet")) for _ in range(3)]
    x = torch.randn(4, SEQ_LEN, DIM)
    ensemble = Ensemble(members).eval()
    probs = torch.softmax(ensemble(x), dim=-1)
    expected = torch.stack([torch.softmax(m(x), dim=-1) for m in members]).mean(0)
    torch.testing.assert_close(probs, expected, atol=1e-5, rtol=1e-4)
    rebuilt = build_model({"members": [_ckpt("signnet"), _ckpt("transformer")],
                           "labels": {f"w{i}": i for i in range(5)}, "input_dim": DIM})
    assert isinstance(rebuilt, Ensemble) and len(rebuilt.members) == 2


def test_supervised_contrastive_prefers_clustered_embeddings():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.nn.functional.one_hot(labels, 3).float() + 0.01 * torch.randn(6, 3)
    scrambled = clustered[torch.tensor([0, 2, 4, 1, 3, 5])]
    assert supervised_contrastive(clustered, labels) < supervised_contrastive(scrambled, labels)
    lonely = torch.randn(4, 8, requires_grad=True)
    loss = supervised_contrastive(lonely, torch.arange(4))      # no clip has a same-word partner
    assert loss.item() == 0.0
    loss.backward()                                             # still differentiable, so train.py never breaks
    real = torch.randn(6, 8, requires_grad=True)
    supervised_contrastive(real, labels).backward()
    assert real.grad is not None and torch.isfinite(real.grad).all()


def test_class_balanced_weights_favour_rare_words():
    labels = [0] * 8 + [1] * 2
    natural, balanced = class_balanced_weights(labels, 0.0), class_balanced_weights(labels, 1.0)
    assert len(set(natural)) == 1
    assert balanced[-1] > balanced[0] and abs(sum(balanced[:8]) - sum(balanced[8:])) < 1e-9


def test_mirror_face_is_its_own_inverse_and_swaps_sides():
    face = np.random.default_rng(0).normal(size=(SEQ_LEN, FACE_DIM)).astype(np.float32)
    flipped = mirror_face(face)
    np.testing.assert_array_equal(flipped[:, 0], face[:, 1])    # brow L <-> R
    np.testing.assert_array_equal(flipped[:, 8], -face[:, 8])   # roll flips sign
    np.testing.assert_array_equal(mirror_face(flipped), face)


def test_strong_augmentation_keeps_shape_and_finite_values():
    from data import SignDataset
    clip = np.random.default_rng(1).normal(size=(SEQ_LEN, FRAME_DIM)).astype(np.float32)
    ds = SignDataset([], {}, augment=True, mirror_prob=0.5, rng_seed=3, strong=True)
    for _ in range(20):
        out, face, duration = ds._augment(clip, None, 1.5)
        assert out.shape == clip.shape and np.isfinite(out).all() and face is None
        assert 0.5 < duration < 2.5                              # follows the crop and the signer's own pace


def test_time_warp_changes_pacing_and_the_old_speed_augmentation_really_was_a_no_op():
    from data import _sample, time_positions
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)[:, None].repeat(3, 1)
    wave = np.sin(np.linspace(0, 6 * np.pi, SEQ_LEN)).astype(np.float32)[:, None].repeat(3, 1)

    # The retired augmentation resampled to a new length and straight back; on any smooth signal that
    # returns the signal. This test pins why it was replaced.
    def old_speed(arr, n):
        def interp(a, length):
            idx = np.linspace(0, len(a) - 1, length)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, len(a) - 1)
            f = (idx - lo)[:, None]
            return a[lo] * (1 - f) + a[hi] * f
        return interp(interp(arr, n), SEQ_LEN)
    assert np.allclose(old_speed(ramp, 51), ramp, atol=1e-4)

    # The new warp, with no cropping, keeps both ends of the sign but moves what happens in between.
    for _ in range(10):
        positions, covered = time_positions(rng, crop=0.0, loosen=0.0)
        assert covered == 1.0 and abs(positions[0]) < 1e-6 and abs(positions[-1] - (SEQ_LEN - 1)) < 1e-6
        assert (np.diff(positions) >= -1e-9).all()               # time never runs backwards
    positions, _ = time_positions(np.random.default_rng(1), crop=0.0, loosen=0.0)
    assert np.abs(_sample(wave, positions) - wave).max() > 0.05  # genuinely different timing
    assert np.abs(_sample(ramp, positions) - ramp).max() > 0.01


def test_a_looser_trim_holds_the_edge_frame_and_a_tighter_one_is_a_shorter_sign():
    from data import _sample, time_positions
    clip = np.arange(SEQ_LEN, dtype=np.float32)[:, None]
    held = _sample(clip, np.array([-3.0, -1.0, 0.0, SEQ_LEN + 5.0]))
    assert held[:, 0].tolist() == [0.0, 0.0, 0.0, SEQ_LEN - 1.0]   # beyond the ends: first/last frame, not invented
    rng = np.random.default_rng(3)
    fractions = [time_positions(rng)[1] for _ in range(200)]
    assert min(fractions) < 1.0 < max(fractions) + 0.2 and 0.8 < min(fractions) and max(fractions) < 1.1


def test_word_bars_demand_more_of_a_word_the_model_confuses():
    from word_bars import word_bars
    # "good" is right whenever it is confident; "shaky" is confident and wrong half the time.
    predicted = ["good"] * 6 + ["shaky"] * 6
    truth = ["good"] * 6 + ["shaky", "other"] * 3
    confidence = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7] + [0.95, 0.94, 0.93, 0.92, 0.91, 0.9]
    bars = word_bars(predicted, confidence, truth, target=0.9, min_count=3)
    assert bars["good"] <= 0.75                                  # trustworthy down to a low confidence
    assert bars["shaky"] == 1.0                                  # never promises 90%: always a candidate
    assert word_bars(["rare"] * 2, [0.9, 0.9], ["rare"] * 2, 0.9, min_count=3) == {}   # too few to judge


def test_ensemble_refuses_members_that_do_not_share_a_vocabulary_or_features():
    from ensemble import combine
    a, b = _ckpt("signnet"), _ckpt("signnet")
    assert combine([a, b])["arch"] == "ensemble"
    with pytest.raises(ValueError):
        combine([a])
    with pytest.raises(ValueError):
        combine([a, {**b, "labels": {"x": 0, "y": 1, "z": 2, "u": 3, "v": 4}}])
    with pytest.raises(ValueError):
        combine([a, {**b, "feature_version": 2}])


def test_kiosk_accepts_the_time_aware_feature_versions_but_never_the_face_one(tmp_path):
    from kiosk.recognition import ModelUnavailable, Recognizer
    for version in (4, 5):
        net = SignTransformer(input_dim=INPUT_DIMS[version], num_classes=3)
        path = tmp_path / f"v{version}.pt"
        torch.save({"model": net.state_dict(), "labels": {"a": 0, "b": 1, "c": 2},
                    "input_dim": INPUT_DIMS[version], "feature_version": version}, path)
        assert Recognizer(path, 0.5, 0.8, 3).feature_version == version
    net = SignTransformer(input_dim=INPUT_DIMS[3], num_classes=3)
    torch.save({"model": net.state_dict(), "labels": {"a": 0, "b": 1, "c": 2}, "input_dim": INPUT_DIMS[3],
                "feature_version": 3}, tmp_path / "v3.pt")
    with pytest.raises(ModelUnavailable, match="face"):
        Recognizer(tmp_path / "v3.pt", 0.5, 0.8, 3)


def test_kiosk_recognizer_applies_word_bars_only_to_be_stricter(tmp_path):
    from kiosk.recognition import ModelUnavailable, Recognizer
    ckpt = {**_ckpt("transformer"), "word_bars": {"w1": 0.97, "w2": 0.1}}
    path = tmp_path / "m.pt"
    torch.save(ckpt, path)
    rec = Recognizer(path, reject_threshold=0.5, confident_threshold=0.8, top_k=3)
    assert rec.commit_bar("w1") == 0.97                          # stricter than the language bar
    assert rec.commit_bar("w2") == 0.8                           # a lower word bar never loosens it
    assert rec.commit_bar("w3") == 0.8 and rec.commit_bar(None) == 0.8
    torch.save({**ckpt, "word_bars": {"w1": 2.0}}, path)
    with pytest.raises(ModelUnavailable):
        Recognizer(path, 0.5, 0.8, 3)
    torch.save({**ckpt, "feature_version": 3, "input_dim": INPUT_DIMS[3]}, path)
    with pytest.raises(ModelUnavailable):                        # the kiosk cannot supply face features yet
        Recognizer(path, 0.5, 0.8, 3)


def _run(*args, cwd):
    # The scripts print arrows and Arabic; the Windows console default (cp1252) cannot encode them.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True,
                          timeout=600, env=env, encoding="utf-8")


def test_train_cli_end_to_end_with_every_new_flag(tmp_path):
    root = ML.parent
    made = _run(ML / "make_synthetic.py", "--out", tmp_path / "syn", "--signs", 6, "--signers", 8, "--reps", 3, cwd=root)
    assert made.returncode == 0, made.stderr[-800:]
    manifest = json.loads((tmp_path / "syn/manifest.json").read_text(encoding="utf-8"))
    signs = sorted({m["sign_id"] for m in manifest})
    np.savez(tmp_path / "text.npz", ids=np.array(signs),
             vectors=np.random.default_rng(0).normal(size=(len(signs), 16)).astype(np.float32))

    out = tmp_path / "m.pt"
    trained = _run(ML / "train.py", tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json",
                   "--out", out, "--epochs", 2, "--arch", "signnet", "--features", 2, "--balance", 0.5,
                   "--supcon", 0.1, "--strong-augment", "--text-embeddings", tmp_path / "text.npz", cwd=root)
    assert trained.returncode == 0, trained.stderr[-1200:]
    ckpt = torch.load(out, weights_only=True)
    assert ckpt["arch"] == "signnet" and ckpt["feature_version"] == 2 and ckpt["input_dim"] == INPUT_DIMS[2]
    assert ckpt["config"] == {"text_dim": 16}
    assert build_model(ckpt)(torch.zeros(1, SEQ_LEN, INPUT_DIMS[2])).shape == (1, len(signs))

    evaluated = _run(ML / "evaluate.py", out, tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json",
                     cwd=root)
    assert evaluated.returncode == 0, evaluated.stderr[-1200:]
    assert "top-1 accuracy" in evaluated.stdout

    # Per-word bars and an ensemble, built from what was just trained, still evaluate.
    barred = tmp_path / "bars.pt"
    bars = _run(ML / "word_bars.py", out, tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json",
                "--out", barred, "--min-count", 1, cwd=root)
    assert bars.returncode == 0, bars.stderr[-1200:]
    assert isinstance(torch.load(barred, weights_only=True)["word_bars"], dict)
    both = tmp_path / "ens.pt"
    assert _run(ML / "ensemble.py", out, out, "--out", both, cwd=root).returncode == 0
    ens = _run(ML / "evaluate.py", both, tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json", cwd=root)
    assert ens.returncode == 0, ens.stderr[-1200:]


def test_pretraining_feeds_fine_tuning(tmp_path):
    root = ML.parent
    assert _run(ML / "make_synthetic.py", "--out", tmp_path / "syn", "--signs", 4, "--signers", 8, "--reps", 3,
                cwd=root).returncode == 0
    pre = tmp_path / "pre.pt"
    done = _run(ML / "pretrain.py", tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json",
                "--out", pre, "--epochs", 1, "--features", 2, cwd=root)
    assert done.returncode == 0, done.stderr[-1200:]
    tuned = _run(ML / "train.py", tmp_path / "syn/manifest.json", "--split", tmp_path / "syn/split.json",
                 "--out", tmp_path / "ft.pt", "--epochs", 1, "--arch", "signnet", "--features", 2,
                 "--pretrained", pre, cwd=root)
    assert tuned.returncode == 0, tuned.stderr[-1200:]
    assert "pretrained: loaded" in tuned.stdout

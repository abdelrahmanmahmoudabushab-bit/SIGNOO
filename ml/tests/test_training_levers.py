"""Weight averaging, the learning-rate schedule, test-time views, and the bench's configuration list."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import bench
from core.features import FRAME_DIM, INPUT_DIMS, SEQ_LEN
from data import Sample, SignDataset
from evaluate import predict
from model import SignTransformer
from train import ModelEMA, lr_factor

ML = Path(__file__).resolve().parent.parent


def test_ema_tracks_the_weights_and_starts_gently():
    net = torch.nn.Linear(3, 2)
    ema = ModelEMA(net, decay=0.9)
    start = {k: v.clone() for k, v in ema.module.state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)                                              # the live weights move by +1
    ema.update(net)
    moved = [(ema.module.state_dict()[k] - start[k]).mean().item() for k in start]
    assert all(0 < m < 1.0 for m in moved)                          # between old and new, never past
    first_decay = 2 / 11                                            # eased-up decay on the first update
    assert abs(moved[0] - (1 - first_decay)) < 1e-5
    for _ in range(500):
        ema.update(net)
    for k, v in net.state_dict().items():
        torch.testing.assert_close(ema.module.state_dict()[k], v, atol=1e-3, rtol=1e-3)   # converges to constant weights
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_ema_copies_non_float_state_exactly():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)
            self.register_buffer("count", torch.tensor(0))
    net = Net()
    ema = ModelEMA(net)
    net.count += 7
    ema.update(net)
    assert ema.module.count.item() == 7


def test_learning_rate_warms_up_then_decays_to_zero():
    epochs, warmup = 20, 4
    rates = [lr_factor(e, epochs, warmup) for e in range(epochs)]
    assert rates[0] < rates[1] < rates[2] < rates[3] and abs(rates[3] - 1.0) < 1e-9     # ramps up
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))                            # then only decays
    assert rates[-1] < 0.02
    plain = [lr_factor(e, epochs, 0) for e in range(epochs)]
    assert abs(plain[0] - 1.0) < 1e-9                                                   # no warm-up: starts at full rate


def _samples(tmp_path, n=4):
    out = []
    for i in range(n):
        path = tmp_path / f"s{i}" / "w" / f"{i}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        clip = np.random.default_rng(i).normal(size=(SEQ_LEN, FRAME_DIM)).astype(np.float32)
        clip = np.cumsum(clip, axis=0) * 0.05                                          # smooth, so re-timing visibly moves it
        np.save(path, clip)
        out.append(Sample(signer_id=f"s{i}", sign_id="w", path=path))
    return out


def test_test_time_views_are_deterministic_and_view_zero_is_the_clip_itself(tmp_path):
    samples = _samples(tmp_path)
    labels = {"w": 0}
    plain = SignDataset(samples, labels)[0][0]
    zero = SignDataset(samples, labels, tta_view=0)[0][0]
    one = SignDataset(samples, labels, tta_view=1)[0][0]
    two = SignDataset(samples, labels, tta_view=2)[0][0]
    np.testing.assert_array_equal(plain, zero)                                       # view 0 changes nothing
    assert np.abs(one - plain).max() > 1e-3 and np.abs(one - two).max() > 1e-3        # other views really differ
    np.testing.assert_array_equal(one, SignDataset(samples, labels, tta_view=1)[0][0])  # and are repeatable
    assert one.shape == plain.shape


def test_predict_with_views_averages_probabilities_and_keeps_the_default_behaviour(tmp_path):
    samples = _samples(tmp_path)
    labels = {"w": 0, "x": 1, "y": 2}
    torch.manual_seed(0)
    model = SignTransformer(input_dim=INPUT_DIMS[1], num_classes=3)
    preds1, probs1 = predict(model, samples, labels, "cpu")
    preds3, probs3 = predict(model, samples, labels, "cpu", tta=3)
    assert probs1.shape == probs3.shape == (4, 3)
    np.testing.assert_allclose(probs3.sum(axis=1), 1.0, atol=1e-5)
    assert not np.allclose(probs1, probs3)                                            # the views contribute
    plain = SignDataset(samples, labels)
    batch = torch.stack([torch.from_numpy(plain[i][0]) for i in range(4)])
    manual = torch.softmax(model.eval()(batch), dim=1).detach().numpy()
    np.testing.assert_allclose(probs1, manual, atol=1e-5)                              # tta=1 is exactly the old prediction
    assert (preds1 == probs1.argmax(1)).all()


def test_every_bench_configuration_uses_flags_train_py_actually_has():
    help_text = subprocess.run([sys.executable, str(ML / "train.py"), "--help"], capture_output=True, text=True,
                               encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    for name, args in bench.CONFIGS.items():
        for token in args:
            if token.startswith("--"):
                assert token in help_text, f"bench config {name!r} uses {token}, which train.py does not accept"


def test_bench_subset_keeps_the_best_supplied_words_beside_the_original_manifest(tmp_path):
    records = [{"signer_id": "a", "sign_id": w, "features": f"{w}{i}.npy"}
               for w, n in (("big", 5), ("mid", 3), ("tiny", 1)) for i in range(n)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    subset = bench.subset_manifest(manifest, top_words=2)
    assert subset.parent == manifest.parent                                            # paths in it stay relative to the same folder
    assert {r["sign_id"] for r in json.loads(subset.read_text())} == {"big", "mid"}
    result = bench.run_one("recipe", bench.CONFIGS["recipe"], manifest, tmp_path / "s.json", 1, 0, None, tmp_path)
    assert result["status"].startswith("skipped")                                      # needs text embeddings, says so

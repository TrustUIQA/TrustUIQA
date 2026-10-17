import numpy as np
import pytest

from uwiqa.detect.metrics import Dets, dataset_map
from uwiqa.detect.policy import MatchCache


@pytest.fixture
def cache(tmp_path):
    gts, store, preds_by = {}, {}, {}
    rng = np.random.default_rng(0)
    for i in range(6):
        b = np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]])
        gts[f"im{i}"] = Dets(b, np.array([0, 1]))
        good = np.c_[b, [0.9, 0.8], [0, 1]]
        bad = np.c_[b + 0.15, [0.9, 0.8], [0, 1]]           # mislocalised
        noisy = np.c_[b + rng.normal(0, 0.01, b.shape), [0.7, 0.6], [0, 1]]
        for m, a in (("raw", noisy if i % 2 else good), ("enh", good if i % 2 else bad)):
            store[f"im{i}|{m}"] = a.astype(np.float32)
            preds_by[(f"im{i}", m)] = Dets(a[:, :4], a[:, 5].astype(int), a[:, 4])
    np.savez(tmp_path / "preds_0000.npz", **store)
    return MatchCache(str(tmp_path / "preds_*.npz"), gts), gts, preds_by


def test_cache_matches_dataset_map(cache):
    c, gts, preds = cache
    imgs = sorted(gts)
    for choice in (["raw"] * 6, ["enh"] * 6, ["raw", "enh"] * 3):
        ref = dataset_map([preds[(i, m)] for i, m in zip(imgs, choice)], [gts[i] for i in imgs])
        got = c.map(imgs, choice)
        assert got["map"] == pytest.approx(ref["map"]) and got["map50"] == pytest.approx(ref["map50"])


def test_oracle_choice_beats_both(cache):
    c, gts, _ = cache
    imgs = sorted(gts)
    oracle = ["raw" if i % 2 == 0 else "enh" for i in range(6)]
    assert c.map(imgs, oracle)["map"] >= max(c.map(imgs, ["raw"] * 6)["map"], c.map(imgs, ["enh"] * 6)["map"])
    b = c.bootstrap_diff(imgs, oracle, ["enh"] * 6, n_boot=50)
    assert b["diff_mean"] > 0 and b["lo"] <= b["diff_mean"] <= b["hi"]

import numpy as np
import pytest

from uwiqa.eval import correlate, grouped_srcc, paired_bootstrap_srcc, plcc, srcc


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 300)
    mos = 1 + 4 / (1 + np.exp(-8 * (x - 0.5))) + rng.normal(0, 0.15, 300)
    return x, mos


def test_logistic_fit_helps_nonlinear(xy):
    x, mos = xy
    raw = np.exp(6 * x)  # monotone but very non-linear in mos
    assert plcc(raw, mos, fit=True) > plcc(raw, mos, fit=False) + 0.05
    assert srcc(raw, mos) == pytest.approx(srcc(x, mos))


def test_ci_brackets_point(xy):
    x, mos = xy
    r = correlate(x, mos, n_boot=200, bootstrap_plcc_fit=False)
    assert r.srcc_lo <= r.srcc <= r.srcc_hi
    assert r.plcc_lo <= r.plcc <= r.plcc_hi
    assert r.srcc > 0.9


def test_lower_better_flip(xy):
    x, mos = xy
    a = correlate(x, mos, n_boot=50, bootstrap_plcc_fit=False)
    b = correlate(-x, mos, higher_better=False, n_boot=50, bootstrap_plcc_fit=False)
    assert a.srcc == pytest.approx(b.srcc)


def test_nan_rows_dropped(xy):
    x, mos = xy
    x = x.copy()
    x[:10] = np.nan
    assert correlate(x, mos, n_boot=20, bootstrap_plcc_fit=False).n == 290


def test_paired_bootstrap_detects_better_metric(xy):
    x, mos = xy
    rng = np.random.default_rng(1)
    worse = x + rng.normal(0, 0.5, len(x))
    r = paired_bootstrap_srcc(x, worse, mos, n_boot=500)
    assert r["diff"] > 0 and r["p"] < 0.01


def test_grouped_srcc():
    # Global ordering is driven by scene offsets; within scene metric is perfect.
    scenes = np.repeat(np.arange(20), 5)
    within = np.tile(np.arange(5), 20).astype(float)
    mos = within + scenes * 10
    pred = within - scenes * 10  # anti-correlated globally, perfect intra-scene
    assert srcc(pred, mos) < 0
    g = grouped_srcc(pred, mos, scenes)
    assert g["n_groups"] == 20 and g["mean"] == pytest.approx(1.0)

"""Algorithm-level agreement: does a metric rank the enhancement METHODS the way MOS does?

This is how UIE benchmark tables are used (one mean score per method, best method in bold), and how the UIEB / SAUD /
UID2021 / SOTA dataset papers rank enhancement algorithms by MOS. To keep scales comparable across scenes (SAUD's
Bradley-Terry scores, NUIQ's per-scene scores) each method's score is its mean within-scene percentile rank, for the
metric and for MOS alike. Per metric:
  ktau            Kendall tau between the metric's and MOS's method ranking
  ktau_lo/hi      95% scene-bootstrap CI
  top_hit         1 if the metric's best method is the MOS-best method
Writes mos/results/method_rank.csv. Usage: python -I mos/method_rank.py uid2021 saud saud2 sota
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parents[1]
B = 1000  # bootstrap resamples


def method_table(d, col):
    """scene x method matrix of within-scene percentile ranks of `col` (NaN where a scene lacks a method)."""
    return d.assign(r=d.groupby("scene")[col].rank(pct=True)).pivot(index="scene", columns="method", values="r")


def evaluate(ds, rng):
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    y = method_table(man, "mos")
    rows = []
    for f in sorted((REPO / "mos/scores" / ds).glob("*.csv")):
        d = man.merge(pd.read_csv(f), on="id")
        d["s"] = np.where(d.lower_better, -d.score, d.score)
        if not np.isfinite(d.s).all():
            continue
        s = method_table(d, "s").loc[y.index, y.columns]
        ms, my = np.nanmean(s, 0), np.nanmean(y, 0)
        boot = []
        for _ in range(B):
            p = rng.integers(0, len(y), len(y))
            boot.append(kendalltau(np.nanmean(s.to_numpy()[p], 0), np.nanmean(y.to_numpy()[p], 0))[0])
        rows.append(dict(ds=ds, metric=f.stem, n_methods=len(y.columns), ktau=kendalltau(ms, my)[0],
                         ktau_lo=np.nanpercentile(boot, 2.5), ktau_hi=np.nanpercentile(boot, 97.5),
                         top_metric=y.columns[np.argmax(ms)], top_mos=y.columns[np.argmax(my)],
                         top_hit=int(np.argmax(ms) == np.argmax(my))))
    return pd.DataFrame(rows).sort_values("ktau", ascending=False)


if __name__ == "__main__":
    # self-check: MOS itself ranks methods perfectly, its negation perfectly backwards
    t = pd.DataFrame(dict(scene=[1, 1, 1, 2, 2, 2], method=list("abcabc"), mos=[1., 2, 3, 2, 3, 4]))
    m = np.nanmean(method_table(t, "mos"), 0)
    assert kendalltau(m, [1, 2, 3])[0] == 1 and kendalltau(-m, [1, 2, 3])[0] == -1
    rng = np.random.default_rng(0)
    out = pd.concat([evaluate(ds, rng) for ds in sys.argv[1:]])
    out.round(4).to_csv(REPO / "mos/results/method_rank.csv", index=False)
    for ds, g in out.groupby("ds", sort=False):
        print(f"== {ds} ({len(g)} metrics, {g.n_methods.iloc[0]} methods, MOS-best {g.top_mos.iloc[0]})")
        print(g.drop(columns=["ds", "n_methods", "top_mos"]).round(2).to_string(index=False))

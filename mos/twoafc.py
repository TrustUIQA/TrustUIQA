"""BAPPS-style 2AFC score on our study (Zhang et al., CVPR 2018, "The Unreasonable Effectiveness of Deep Features").

Per pair (condition x image) with n votes, k of them for the edit: a metric earns the share of votes that agree with
its pick (a metric tie earns 0.5). The human ceiling is the chance that two different raters of that pair agree,
[k(k-1) + (n-k)(n-k-1)] / [n(n-1)] (unbiased; BAPPS's p^2 + (1-p)^2 counts a rater agreeing with itself).
Pairs with fewer than 2 votes are dropped. Reads mos/results/study_pairs.csv (mos/study_kappa.py).
Writes mos/results/twoafc.csv: metric, score, 95% CI (image-cluster bootstrap), ceiling, score / ceiling, n pairs.
Usage: python -I mos/twoafc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
B = 1000


def credit(edit_better, tie, p):
    """Share of votes agreeing with the metric's pick; p = share of votes for the edit."""
    return np.where(tie, 0.5, np.where(edit_better, p, 1 - p))


def ceiling(k, n):
    return (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))


if __name__ == "__main__":
    assert credit(np.array([True]), np.array([False]), np.array([0.8]))[0] == 0.8
    assert ceiling(5, 5) == 1 and ceiling(1, 2) == 0 and abs(ceiling(2, 3) - 1 / 3) < 1e-12
    d = pd.read_csv(REPO / "mos/results/study_pairs.csv", dtype={"image": str})
    d = d[d.n_votes >= 2]
    k = (d.human_pref_edit * d.n_votes).round()
    d["ceil"] = ceiling(k, d.n_votes)
    tie = d.original == d.edited
    d["credit"] = credit(d.metric_prefers_edit.astype(bool), tie, d.human_pref_edit)
    rng = np.random.default_rng(0)
    images = d.image.unique()
    rows = []
    for m, g in d.groupby("metric"):
        per = g.groupby("image")[["credit", "ceil"]].agg(["sum", "count"])
        c, n, h = per["credit"]["sum"], per["credit"]["count"], per["ceil"]["sum"]
        boot = []
        for _ in range(B):
            p = rng.choice(per.index, len(per))
            boot.append(c[p].sum() / n[p].sum())
        rows.append(dict(metric=m, score=c.sum() / n.sum(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                         ceiling=h.sum() / n.sum(), n=int(n.sum())))
    out = pd.DataFrame(rows).assign(rel=lambda x: x.score / x.ceiling).sort_values("score", ascending=False)
    out.round(4).to_csv(REPO / "mos/results/twoafc.csv", index=False)
    print(out.round(3).to_string(index=False))

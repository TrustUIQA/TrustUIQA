"""gMAD on the MOS sets (Ma et al., CVPR 2016, "Group MAximum Differentiation competition"), with MOS as the judge.

gMAD pits an attacker metric A against a defender D: among pairs of images that D scores (nearly) the same, pick the
pairs A separates most. If people prefer A's pick, D failed to see a difference A found. The original asks new
raters; here the pool is already rated, so MOS decides. Pairs are enhancements of the same scene (MOS is not
comparable across SAUD scenes). Scores are put on one scale per metric as dataset-wide percentile ranks.
  D tie        |delta percentile of D| < TIE
  A's pairs    the K tied pairs with the largest |delta percentile of A|
  win(A, D)    share of those K pairs where MOS prefers A's pick (MOS-tied pairs dropped)
Per metric: aggressiveness = mean win as attacker, resistance = mean (1 - win) as defender, over all opponents.
Writes mos/results/gmad_<ds>.csv (matrix, rows = attacker) and mos/results/gmad.csv (summary).
Usage: python -I mos/gmad.py uid2021 saud saud2 sota
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TIE, K = 0.02, 20


def load(ds):
    """MOS sign per within-scene pair, {metric: percentile-rank difference per pair} (higher = better), and the
    manifest with the pair index arrays i, j (for montages)."""
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    i, j = [], []
    for idx in man.groupby("scene").indices.values():
        a, b = np.triu_indices(len(idx), 1)
        i.append(idx[a])
        j.append(idx[b])
    i, j = np.concatenate(i), np.concatenate(j)
    y = np.sign(man.mos.to_numpy()[i] - man.mos.to_numpy()[j])
    diff = {}
    for f in sorted((REPO / "mos/scores" / ds).glob("*.csv")):
        sc = man[["id"]].merge(pd.read_csv(f), on="id", how="left")
        s = np.where(sc.lower_better, -sc.score, sc.score)
        if not np.isfinite(s).all():
            continue
        r = pd.Series(s).rank(pct=True).to_numpy()
        diff[f.stem] = r[i] - r[j]
    return y, diff, man, i, j


def picks(da, dd, y):
    """A's K most separated pairs among D's ties (MOS-tied pairs dropped); None if fewer than K usable pairs."""
    ok = (np.abs(dd) < TIE) & (y != 0) & (da != 0)
    return np.flatnonzero(ok)[np.argsort(-np.abs(da[ok]))[:K]] if ok.sum() >= K else None


def win(da, dd, y):
    """Share of A's picks where MOS agrees with A; NaN if there are too few."""
    top = picks(da, dd, y)
    return np.nan if top is None else float((np.sign(da[top]) == y[top]).mean())


if __name__ == "__main__":
    # self-check: an attacker equal to MOS always wins, its negation always loses
    y0 = np.array([1., -1, 1, -1] * 10)
    assert win(y0 * 0.5, np.zeros(40), y0) == 1 and win(-y0 * 0.5, np.zeros(40), y0) == 0
    summary = []
    for ds in sys.argv[1:]:
        y, diff = load(ds)[:2]
        ms = sorted(diff)
        w = pd.DataFrame([[win(diff[a], diff[d], y) if a != d else np.nan for d in ms] for a in ms], ms, ms)
        w.round(3).to_csv(REPO / "mos/results" / f"gmad_{ds}.csv")
        s = pd.DataFrame(dict(ds=ds, metric=ms, aggressiveness=w.mean(1).to_numpy(),
                              resistance=(1 - w).mean(0).to_numpy(), n_defended=w.notna().sum(0).to_numpy()))
        summary.append(s)
        print(f"== {ds} ({len(ms)} metrics, {len(y)} pairs)")
        print(s.sort_values("resistance", ascending=False).round(3).to_string(index=False))
    pd.concat(summary).round(4).to_csv(REPO / "mos/results/gmad.csv", index=False)

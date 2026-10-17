"""Agreement of every scored metric with the published MOS of each dataset.

Per metric (score oriented so higher = better):
  srcc / krcc            whole-dataset rank correlation (the number UIQA papers report)
  plcc / rmse            after the VQEG 5-parameter logistic fit, rmse in MOS units
  scene_srcc             mean within-scene SRCC: can the metric rank enhancements of the SAME photo?
  pair_acc               within-scene pairwise agreement with MOS (ties in MOS skipped) - the analogue
                         of our study's two-alternative preference
  kappa                  Cohen's kappa between metric and MOS pairwise preference within a scene (pairs tied
                         in either dropped) - the same statistic as our study's metric-vs-human kappa
  top1                   share of scenes where the metric's best image is the people's best
  srcc_lo/hi, pair_lo/hi 95% scene-bootstrap CIs
Writes mos/results/<ds>.csv. Usage: python -I mos/eval_mos.py uid2021 [saud ...]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
B = 1000  # bootstrap resamples


def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1 / (1 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_plcc_rmse(s, y):
    """VQEG logistic map from metric score to MOS; falls back to linear if the fit does not converge."""
    s = (s - s.mean()) / (s.std() + 1e-12)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(logistic5, s, y, p0=[y.max() - y.min(), 1, 0, 0, y.mean()], maxfev=20000)
        yh = logistic5(s, *p)
    except RuntimeError:
        yh = np.polyval(np.polyfit(s, y, 1), s)
    return abs(pearsonr(yh, y)[0]), float(np.sqrt(np.mean((yh - y) ** 2)))


def scene_stats(g):
    """Per scene: SRCC, (agree, total) over MOS-untied pairs, top-1 hit, then the 2x2 counts for Cohen's kappa on
    pairs neither MOS nor metric ties: (n, metric says first better, MOS says first better, both say it)."""
    s, y = g.s.to_numpy(), g.mos.to_numpy()
    ds, dy = np.sign(s[:, None] - s[None]), np.sign(y[:, None] - y[None])
    iu = np.triu_indices(len(s), 1)
    ds, dy = ds[iu], dy[iu]
    ok = dy != 0
    agree = ((ds == dy) & ok).sum() + 0.5 * ((ds == 0) & ok).sum()  # metric tie = half credit
    r = spearmanr(s, y)[0] if np.ptp(s) > 0 else 0.0
    k = ok & (ds != 0)
    a, h = ds[k] > 0, dy[k] > 0
    return r, agree, ok.sum(), float(y[np.argmax(s)] == y.max()), k.sum(), a.sum(), h.sum(), (a & h).sum()


def pair_kappa(n, a, h, ah):
    """Cohen's kappa from pooled 2x2 counts (study/analyze_votes.py's kappa, as counts)."""
    po = (ah + (n - a - h + ah)) / n
    pe = (a / n) * (h / n) + (1 - a / n) * (1 - h / n)
    return (po - pe) / (1 - pe)


def evaluate(ds):
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted((REPO / "mos/scores" / ds).glob("*.csv")):
        sc = pd.read_csv(f)
        d = man.merge(sc, on="id")
        assert len(d) == len(man), (f.name, len(d))
        d["s"] = np.where(d.lower_better, -d.score, d.score)
        s, y = d.s.to_numpy(), d.mos.to_numpy()
        if not np.isfinite(s).all():
            print(f"skip {ds}/{f.stem}: {(~np.isfinite(s)).sum()} non-finite scores", flush=True)
            continue
        plcc, rmse = fit_plcc_rmse(s, y)
        per = {k: scene_stats(g) for k, g in d.groupby("scene")}
        st = np.array(list(per.values()))  # cols: srcc, agree, n, top1, then kappa counts n, a, h, ah
        scenes = list(per)
        idx = {k: np.flatnonzero(d.scene.to_numpy() == k) for k in scenes}
        boot_srcc, boot_pair, boot_k = [], [], []
        for _ in range(B):
            pick = rng.integers(0, len(scenes), len(scenes))
            ii = np.concatenate([idx[scenes[j]] for j in pick])
            boot_srcc.append(spearmanr(s[ii], y[ii])[0])
            boot_pair.append(st[pick, 1].sum() / st[pick, 2].sum())
            boot_k.append(pair_kappa(*st[pick, 4:].sum(0)))
        rows.append(dict(metric=f.stem, srcc=spearmanr(s, y)[0], srcc_lo=np.percentile(boot_srcc, 2.5),
                         srcc_hi=np.percentile(boot_srcc, 97.5), krcc=kendalltau(s, y)[0], plcc=plcc, rmse=rmse,
                         scene_srcc=st[:, 0].mean(), pair_acc=st[:, 1].sum() / st[:, 2].sum(),
                         pair_lo=np.percentile(boot_pair, 2.5), pair_hi=np.percentile(boot_pair, 97.5),
                         kappa=pair_kappa(*st[:, 4:].sum(0)), kappa_lo=np.percentile(boot_k, 2.5),
                         kappa_hi=np.percentile(boot_k, 97.5), top1=st[:, 3].mean(), n=len(d)))
    out = pd.DataFrame(rows).sort_values("srcc", ascending=False)
    (REPO / "mos/results").mkdir(exist_ok=True)
    out.round(4).to_csv(REPO / "mos/results" / f"{ds}.csv", index=False)
    return out


if __name__ == "__main__":
    # self-check: a perfect metric scores 1 everywhere, its negation is re-oriented by lower_better
    g = pd.DataFrame(dict(s=[1., 2, 3], mos=[1., 2, 3]))
    assert scene_stats(g)[1:4] == (3, 3, 1.0) and scene_stats(g.assign(s=[3., 2, 1]))[1] == 0
    assert pair_kappa(10, 4, 4, 4) == 1 and pair_kappa(10, 5, 5, 0) == -1
    for ds in sys.argv[1:]:
        o = evaluate(ds)
        print(f"== {ds} ({len(o)} metrics)")
        print(o.drop(columns="n").round(3).to_string(index=False))

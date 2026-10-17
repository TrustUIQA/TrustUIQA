"""Correlation harness shared by O1 (metric vs MOS), O4 (predicted vs true
delta) and all ablations.

* SRCC / KRCC on raw scores.
* PLCC after the standard VQEG 4-parameter logistic mapping (never skip it:
  raw PLCC punishes metrics for being non-linearly, but monotonically, related
  to MOS).
* Percentile bootstrap CIs on every coefficient.
* Paired bootstrap for "is metric A significantly better than metric B?".
* Intra-group (per-scene) SRCC: UIE datasets contain K enhanced versions of
  each raw image. Ranking those K versions is what an enhancement metric is
  actually used for, and global SRCC can be dominated by between-scene
  content differences. Report both.

Sign convention: metrics where lower is better (NIQE, BRISQUE) yield negative
correlations. Pass ``higher_better=False`` to flip them so tables are
comparable; we never silently take abs().
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize, stats


def logistic4(x, b1, b2, b3, b4):
    with np.errstate(over="ignore"):  # exp overflow -> inf -> correct limit b2
        return (b1 - b2) / (1.0 + np.exp(-(x - b3) / np.abs(b4))) + b2


def fit_logistic4(pred: np.ndarray, mos: np.ndarray) -> np.ndarray:
    """Fit the VQEG 4-parameter logistic and return mapped predictions."""
    pred = np.asarray(pred, float)
    mos = np.asarray(mos, float)
    sd = pred.std() or 1.0
    p0 = [mos.max(), mos.min(), np.median(pred), sd]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(logistic4, pred, mos, p0=p0, maxfev=20000)
        mapped = logistic4(pred, *popt)
        if np.all(np.isfinite(mapped)):
            return mapped
    except (RuntimeError, ValueError):
        pass
    # Fall back to linear fit (documented, rare): keeps PLCC defined.
    slope, intercept = np.polyfit(pred, mos, 1)
    return slope * pred + intercept


def srcc(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def krcc(a, b) -> float:
    return float(stats.kendalltau(a, b).statistic)


def plcc(pred, mos, fit: bool = True) -> float:
    pred = np.asarray(pred, float)
    mapped = fit_logistic4(pred, mos) if fit else pred
    if np.ptp(mapped) < 1e-12 or np.ptp(mos) == 0:
        return 0.0
    return float(stats.pearsonr(mapped, mos).statistic)


def _orient(pred, higher_better):
    pred = np.asarray(pred, float)
    return pred if higher_better else -pred


@dataclass
class CorrResult:
    n: int
    srcc: float
    srcc_lo: float
    srcc_hi: float
    plcc: float
    plcc_lo: float
    plcc_hi: float
    krcc: float
    krcc_lo: float
    krcc_hi: float

    def as_dict(self):
        return asdict(self)


def correlate(pred, mos, higher_better: bool = True, n_boot: int = 1000,
              ci: float = 0.95, seed: int = 0, bootstrap_plcc_fit: bool = True) -> CorrResult:
    """All three coefficients with percentile bootstrap CIs.

    ``bootstrap_plcc_fit=False`` reuses one logistic fit for all resamples —
    ~50x faster, slightly narrower CIs. Use True for final tables.
    """
    pred = _orient(pred, higher_better)
    mos = np.asarray(mos, float)
    mask = np.isfinite(pred) & np.isfinite(mos)
    pred, mos = pred[mask], mos[mask]
    n = len(pred)
    if n < 3:
        raise ValueError(f"need >=3 valid pairs, got {n}")

    point = (srcc(pred, mos), plcc(pred, mos), krcc(pred, mos))
    rng = np.random.default_rng(seed)
    mapped_once = None if bootstrap_plcc_fit else fit_logistic4(pred, mos)
    boots = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        p, m = pred[idx], mos[idx]
        if bootstrap_plcc_fit:
            pl = plcc(p, m)
        else:
            mm = mapped_once[idx]
            pl = float(stats.pearsonr(mm, m).statistic) if np.ptp(mm) > 1e-12 and np.ptp(m) > 0 else 0.0
        boots[i] = (srcc(p, m), pl, krcc(p, m))
    a = (1 - ci) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a], axis=0)
    return CorrResult(n, point[0], lo[0], hi[0], point[1], lo[1], hi[1],
                      point[2], lo[2], hi[2])


def paired_bootstrap_srcc(pred_a, pred_b, mos, higher_better_a=True, higher_better_b=True,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap SRCC(A) - SRCC(B) on the SAME resampled images.

    Returns the point difference, 95% CI, and a two-sided p-value
    (fraction of resamples where the sign flips, doubled).
    """
    a = _orient(pred_a, higher_better_a)
    b = _orient(pred_b, higher_better_b)
    mos = np.asarray(mos, float)
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(mos)
    a, b, mos = a[mask], b[mask], mos[mask]
    n = len(mos)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = srcc(a[idx], mos[idx]) - srcc(b[idx], mos[idx])
    d = srcc(a, mos) - srcc(b, mos)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": d, "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


def grouped_srcc(pred, mos, groups, higher_better: bool = True, min_size: int = 3) -> dict:
    """Mean/median of per-group SRCC (e.g. group = raw scene id).

    Groups smaller than ``min_size`` or with constant MOS are skipped.
    """
    pred = _orient(pred, higher_better)
    mos = np.asarray(mos, float)
    groups = np.asarray(groups)
    vals = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_size or np.std(mos[m]) == 0 or np.std(pred[m]) == 0:
            continue
        vals.append(srcc(pred[m], mos[m]))
    vals = np.asarray(vals)
    if len(vals) == 0:
        return {"n_groups": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
    return {"n_groups": int(len(vals)), "mean": float(vals.mean()),
            "median": float(np.median(vals)), "std": float(vals.std())}

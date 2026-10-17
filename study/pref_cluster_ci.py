"""Per-condition share of votes preferring the edit, with a 95% interval that resamples raters and photos together
(two-way cluster bootstrap, as analyze_votes.cluster_ci does for kappa). Wilson intervals treat the 160 votes of a
condition as independent, but they share raters and photos.
Writes mos/results/study_pref_ci.csv. Usage (repo root): python -I study/pref_cluster_ci.py study/responses/*.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_votes import checks, load, picked_clean  # noqa: E402


def cluster_share(edit, rater, image, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    ru, ri = np.unique(rater, return_inverse=True)
    gu, gi = np.unique(image, return_inverse=True)
    boot = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ru), len(ru)), minlength=len(ru))[ri] * \
            np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu))[gi]
        if w.sum():
            boot.append(np.average(edit, weights=w))
    return np.percentile(boot, [2.5, 97.5])


if __name__ == "__main__":
    lo, hi = cluster_share(np.array([1, 0] * 50), np.arange(100) % 10, np.arange(100) % 7, n_boot=200)
    assert lo < 0.5 < hi  # a 50/50 condition must not look rejected
    sessions = list({s["run"]: s for p in sys.argv[1:] for s in load(p)}.values())
    kept = [s for s in sessions if checks(s)["pass"]]
    v = pd.DataFrame([dict(c=x["c"], img=x["img"], run=s["run"], edit=not picked_clean(x))
                      for s in kept for x in s["votes"] if x["k"] == "t"])
    rows = []
    for c, g in v.groupby("c"):
        lo, hi = cluster_share(g.edit.to_numpy(float), g.run.to_numpy(), g.img.to_numpy())
        rows.append(dict(condition=c, n=len(g), pref_edit=g.edit.mean(), lo=lo, hi=hi))
    out = pd.DataFrame(rows).sort_values("pref_edit")
    out.round(4).to_csv(Path(__file__).resolve().parents[1] / "mos/results/study_pref_ci.csv", index=False)
    print(f"{len(kept)} raters\n" + out.round(3).to_string(index=False))

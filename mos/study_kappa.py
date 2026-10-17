"""Metric-vs-human agreement on our 2AFC study (clean original vs edit) for EVERY scored metric, including the new
ones in <pool>/scores_ext/ and scores_cpqi/. Same protocol as study/analyze_votes.py (gold-check filter, Cohen's
kappa, two-way cluster-bootstrap CI); that script and its .tex tables are left untouched.

Writes mos/results/study_kappa_all.csv   per metric: kappa, CI, raw agreement, n, share of pairs "fooled"
                                         (metric prefers the edit while the rater preferred the original)
       mos/results/study_pairs.csv       per condition x image x metric: original and edited score, whether the
                                         metric prefers the edit, and the raters' share preferring the edit
Usage (repo root): python -I mos/study_kappa.py study/responses/*.txt
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "study"))
from analyze_votes import CLEAN, SRC, checks, cluster_ci, kappa, load, picked_clean, scores  # noqa: E402


def all_scores():
    out = scores()  # <pool>/scores/ + the pilot's attack.csv
    for d in ("out_ablate", "out"):
        for sub in ("scores_ext", "scores_cpqi"):
            for f in Path(d, sub).glob("*.csv"):
                for r in pd.read_csv(f, dtype={"image": str}).itertuples():
                    out[(d, f.stem, r.mode, r.image, int(r.step))] = (r.score, bool(r.lower_better))
    return out


def main(paths):
    os.chdir(REPO)  # analyze_votes reads relative paths
    sessions = list({s["run"]: s for p in paths for s in load(p)}.values())
    kept = [s for s in sessions if checks(s)["pass"]]
    sc = all_scores()
    metrics = sorted({k[1] for k in sc})
    agree = defaultdict(lambda: ([], [], [], []))  # metric_edit, human_edit, rater, image
    votes = defaultdict(list)
    for s in kept:
        for v in s["votes"]:
            if v["k"] != "t":
                continue
            votes[(v["c"], v["img"])].append(not picked_clean(v))
            d, mode, step = SRC[v["c"]]
            cmode, cstep = CLEAN[d]
            for m in metrics:
                a, b = sc.get((d, m, mode, v["img"], step)), sc.get((d, m, cmode, v["img"], cstep))
                if a is None or b is None or a[0] == b[0] or not np.isfinite([a[0], b[0]]).all():
                    continue
                for col, x in zip(agree[m], ((a[0] < b[0]) if a[1] else (a[0] > b[0]), not picked_clean(v),
                                             s["run"][:8], v["img"])):
                    col.append(x)
    rows = []
    for m, cols in agree.items():
        a, h, run, img = map(np.array, cols)
        lo, hi = cluster_ci(a, h, run, img)
        rows.append(dict(metric=m, kappa=kappa(a, h), kappa_lo=lo, kappa_hi=hi, agree=(a == h).mean(),
                         metric_prefers_edit=a.mean(), fooled=(a & ~h).mean(), n=len(a)))
    kt = pd.DataFrame(rows).sort_values("kappa", ascending=False)
    pairs = []
    for (c, img), hv in votes.items():
        d, mode, step = SRC[c]
        cmode, cstep = CLEAN[d]
        for m in metrics:
            a, b = sc.get((d, m, mode, img, step)), sc.get((d, m, cmode, img, cstep))
            if a is None or b is None:
                continue
            pairs.append(dict(condition=c, image=img, metric=m, original=b[0], edited=a[0], lower_better=a[1],
                              metric_prefers_edit=(a[0] < b[0]) if a[1] else (a[0] > b[0]),
                              human_pref_edit=np.mean(hv), n_votes=len(hv)))
    (REPO / "mos/results").mkdir(exist_ok=True)
    kt.round(4).to_csv(REPO / "mos/results/study_kappa_all.csv", index=False)
    pd.DataFrame(pairs).to_csv(REPO / "mos/results/study_pairs.csv", index=False)
    print(f"{len(kept)}/{len(sessions)} sessions pass the gold check; {len(metrics)} metrics")
    print(kt.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])

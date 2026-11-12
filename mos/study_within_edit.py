"""Within-edit analysis of our paired study (reviewer yVbw): how much of a metric's study kappa comes from which of
the 12 edits it prefers, and how much from per-photo judgment.

Per metric, on the same votes as mos/study_kappa.py (gold-check filter, pairs the metric scores as equal dropped):
  kappa         Cohen's kappa as in the paper
  kappa_edit    kappa after replacing the metric's choice by its majority choice for that edit (edit preference only)
  kappa_within  within-edit kappa: chance agreement is computed inside each edit from that edit's two marginals, so a
                metric that only knows which edit it is looking at scores 0, then pooled over edits by vote count
  lo, hi        95% interval of kappa_within, resampling raters and photos together (as analyze_votes.cluster_ci)
Also: kappa_within between two raters of the same block (median over pairs), and the kappa of an "edit oracle" that
always picks the people's majority choice for each edit.

Writes mos/results/study_within_edit.csv and mos/results/study_within_edit_ref.csv
Usage (repo root): python -I mos/study_within_edit.py study/choices.csv   (or the raw sessions: study/responses/*.txt)
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "study"))
from analyze_votes import checks, kappa, load, picked_clean  # noqa: E402


def kappa_within(a, h, cond, w=None):
    """Cohen's kappa with chance agreement taken inside each condition (stratified), optional item weights."""
    a, h = np.asarray(a, float), np.asarray(h, float)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    if not w.sum():
        return float("nan")
    po = np.average(a == h, weights=w)
    pe = 0.0
    for c in np.unique(cond):
        m = cond == c
        wc = w[m].sum()
        if wc:
            pa, ph = np.average(a[m], weights=w[m]), np.average(h[m], weights=w[m])
            pe += wc * (pa * ph + (1 - pa) * (1 - ph))
    pe /= w.sum()
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cluster_ci(f, rater, image, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ru, ri = np.unique(rater, return_inverse=True)
    gu, gi = np.unique(image, return_inverse=True)
    boot = []
    for _ in range(n_boot):
        wr = np.bincount(rng.integers(0, len(ru), len(ru)), minlength=len(ru))
        wg = np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu))
        boot.append(f(wr[ri] * wg[gi]))
    return np.nanpercentile(boot, [2.5, 97.5])


def load_votes(paths):
    """Test votes of raters who pass every attention check, from the raw sessions or the released study/choices.csv."""
    if len(paths) == 1 and paths[0].endswith(".csv"):
        ch = pd.read_csv(paths[0], dtype={"image": str, "edit": str})
        failed = ch[(ch.trial_type == "attention") & (ch.chose_edit == 1)].rater.unique()
        t = ch[(ch.trial_type == "test") & ~ch.rater.isin(failed)]
        return pd.DataFrame(dict(condition=t.edit, image=t.image, human=t.chose_edit.astype(bool), rater=t.rater))
    sessions = list({s["run"]: s for p in paths for s in load(p)}.values())
    kept = [s for s in sessions if checks(s)["pass"]]
    return pd.DataFrame([dict(condition=v["c"], image=v["img"], human=not picked_clean(v), rater=s["run"][:8])
                         for s in kept for v in s["votes"] if v["k"] == "t"])


def main(paths):
    os.chdir(REPO)
    votes = load_votes(paths)
    sp = pd.read_csv(REPO / "mos/results/study_pairs.csv", dtype={"image": str})
    sp = sp[sp.original != sp.edited]  # ties excluded, as in study_kappa.py

    # references: edit oracle, and two raters of the same block
    maj = votes.groupby("condition").human.mean() > 0.5
    ref = {"edit_oracle_kappa": kappa(votes.condition.map(maj), votes.human)}
    lab = [dict(zip(zip(g.image, g.condition), g.human)) for _, g in votes.groupby("rater")]
    ir = []
    for a, b in itertools.combinations(lab, 2):
        shared = sorted(a.keys() & b.keys())
        if len(shared) >= 20:
            ir.append((kappa([a[x] for x in shared], [b[x] for x in shared]),
                       kappa_within([a[x] for x in shared], [b[x] for x in shared], np.array([x[1] for x in shared]))))
    ir = np.array(ir)
    ref.update(rater_pairs=len(ir), rater_kappa_median=np.median(ir[:, 0]), rater_within_median=np.median(ir[:, 1]),
               rater_within_min=ir[:, 1].min(), rater_within_max=ir[:, 1].max())

    rows = []
    for m, g in sp.groupby("metric"):
        d = votes.merge(g[["condition", "image", "metric_prefers_edit"]], on=["condition", "image"])
        a, h, c = d.metric_prefers_edit.to_numpy(bool), d.human.to_numpy(bool), d.condition.to_numpy()
        edit_pref = g.groupby("condition").metric_prefers_edit.mean() >= 0.5
        lo, hi = cluster_ci(lambda w: kappa_within(a, h, c, w), d.rater.to_numpy(), d.image.to_numpy())
        rows.append(dict(metric=m, n=len(d), kappa=kappa(a, h), kappa_edit=kappa(d.condition.map(edit_pref), h),
                         kappa_within=kappa_within(a, h, c), lo=lo, hi=hi))
    out = pd.DataFrame(rows).sort_values("kappa_within", ascending=False)
    out.round(4).to_csv(REPO / "mos/results/study_within_edit.csv", index=False)
    pd.Series(ref).round(4).to_csv(REPO / "mos/results/study_within_edit_ref.csv", header=["value"])
    print(pd.Series(ref).round(3).to_string())
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])

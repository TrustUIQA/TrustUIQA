"""Quality-neutral geometry changes: does a flip, a 90-degree turn, a half-size copy or a 90% centre crop change which
enhancement a metric prefers? People's preference should not change (none of these adds or removes distortion).

build  UID2021, 15 random scenes x 16 images, each saved as orig + the 4 transforms to
       uwdata/_edits/uid2021_robust/<t>/ (gitignored, research-only data); manifest
       mos/manifests/uid2021_robust.csv, scored by score_mos.py (nuiq ranks within scene = one photo's versions).
eval   per metric and transform, within-scene pairs (i, j) of originals: flip = sign(s_T(i) - s(j)) != sign(s(i) - s(j)),
       ties count as flips; plus pairwise accuracy against MOS when every image is transformed vs original.
       -> mos/results/robust.csv
Usage: python -I mos/robust.py build | eval
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "uwdata/_edits/uid2021_robust"
T = {"orig": lambda x: x, "hflip": lambda x: x[:, ::-1], "rot90": lambda x: np.ascontiguousarray(np.rot90(x)),
     "half": lambda x: cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
     "crop90": lambda x: x[round(x.shape[0] * .05):round(x.shape[0] * .95), round(x.shape[1] * .05):round(x.shape[1] * .95)]}


def build():
    m = pd.read_csv(REPO / "mos/manifests/uid2021.csv")
    sc = np.random.default_rng(0).choice(sorted(m.scene.unique()), 15, replace=False)
    rows = []
    for r in m[m.scene.isin(sc)].itertuples():
        x = cv2.imread(r.path)
        for t, f in T.items():
            p = OUT / t / f"{r.scene}__{r.method}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), f(x))
            rows.append(dict(id=f"{t}/{r.id}", path=str(p), scene=r.scene, method=r.method, t=t, src=r.id, mos=r.mos))
    pd.DataFrame(rows).to_csv(REPO / "mos/manifests/uid2021_robust.csv", index=False)
    print(len(rows), "images")


def flips(s0, sT, y):
    """s0, sT, y: one scene's original scores, transformed scores, MOS. -> (flips, pairs, MOS-agree orig, MOS-agree T,
    MOS-untied pairs)."""
    i, j = np.triu_indices(len(s0), 1)
    i, j = np.r_[i, j], np.r_[j, i]  # transform either image of the pair
    d0, dT = np.sign(s0[i] - s0[j]), np.sign(sT[i] - s0[j])
    dy = np.sign(y[:, None] - y[None])[np.triu_indices(len(y), 1)]
    a0 = np.sign(s0[:, None] - s0[None])[np.triu_indices(len(y), 1)]
    aT = np.sign(sT[:, None] - sT[None])[np.triu_indices(len(y), 1)]
    ok = dy != 0
    return ((d0 != dT) | (dT == 0)).sum(), len(i), ((a0 == dy) & ok).sum(), ((aT == dy) & ok).sum(), ok.sum()


def evaluate():
    man = pd.read_csv(REPO / "mos/manifests/uid2021_robust.csv")
    rows = []
    for f in sorted((REPO / "mos/scores/uid2021_robust").glob("*.csv")):
        d = man.merge(pd.read_csv(f), on="id")
        d["s"] = np.where(d.lower_better, -d.score, d.score)
        w = d.pivot_table(index=["scene", "src"], columns="t", values="s").join(
            d[d.t == "orig"].set_index(["scene", "src"]).mos)
        for t in T:
            if t == "orig":
                continue
            c = np.array([flips(g.orig.to_numpy(), g[t].to_numpy(), g.mos.to_numpy()) for _, g in w.groupby(level=0)]).sum(0)
            rows.append(dict(metric=f.stem, t=t, flip_rate=c[0] / c[1], pair_acc_orig=c[2] / c[4], pair_acc_t=c[3] / c[4]))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / "mos/results/robust.csv", index=False)
    return out


if __name__ == "__main__":
    # self-check: unchanged scores never flip; scores pushed below every original lose every pair they won
    s, y = np.array([1., 2, 3]), np.array([1., 2, 3])
    assert flips(s, s, y)[0] == 0 and flips(s, -s, y)[0] == 3 and flips(s, s, y)[2] == 3
    if sys.argv[1] == "build":
        build()
    else:
        print(evaluate().pivot_table(index="metric", columns="t", values="flip_rate").round(3).to_string())

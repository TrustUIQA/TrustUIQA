"""Cheap edits vs the human ranking (edits from mos/edits.py, scored into mos/scores/<ds>_edits/).

Per metric and edit, over scenes (score oriented so higher = better):
  beat_best    share of scenes where the metric scores the edited raw photo above the enhancement with the
               highest MOS (humans' best) - a metric that agrees with people should rarely do this
  beat_median  same against the scene's median-MOS enhancement
  beat_raw     edit scored above the unedited raw photo
  pct          mean share of the scene's rated images that the edit outscores
Writes mos/results/<ds>_edits.csv. Usage: python -I mos/edit_eval.py uid2021 saud sota
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def oriented(f):
    d = pd.read_csv(f)
    return d.assign(s=np.where(d.lower_better, -d.score, d.score))[["id", "s"]]


def evaluate(ds):
    man = pd.read_csv(REPO / f"mos/manifests/{ds}.csv")
    man = man[man.method != "Original"]  # rated enhancements only; the raw photo comes from the edit set
    em = pd.read_csv(REPO / f"mos/manifests/{ds}_edits.csv")
    man = man[man.scene.astype(str).isin(em.scene.astype(str))]
    rows = []
    for f in sorted((REPO / f"mos/scores/{ds}_edits").glob("*.csv")):
        if not (REPO / f"mos/scores/{ds}/{f.name}").exists():
            continue
        e = em.merge(oriented(f), on="id")
        b = man.merge(oriented(REPO / f"mos/scores/{ds}/{f.name}"), on="id")
        if len(e) < len(em) or len(b) < len(man):
            continue  # partially scored
        e["scene"], b["scene"] = e.scene.astype(str), b.scene.astype(str)
        raw = e[e.method == "raw"].set_index("scene").s
        grp = {k: g.sort_values("mos") for k, g in b.groupby("scene")}
        best = pd.Series({k: g.s.iloc[-1] for k, g in grp.items()})
        med = pd.Series({k: g.s.iloc[len(g) // 2] for k, g in grp.items()})
        for ed, g in e[e.method != "raw"].groupby("method"):
            v = g.set_index("scene").s
            pct = [np.mean(grp[k].s.to_numpy() < x) for k, x in v.items()]
            rows.append(dict(metric=f.stem, edit=ed, beat_best=np.mean(v > best[v.index]),
                             beat_median=np.mean(v > med[v.index]), beat_raw=np.mean(v > raw[v.index]),
                             pct=np.mean(pct), n=len(v)))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / f"mos/results/{ds}_edits.csv", index=False)
    return out


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        o = evaluate(ds)
        print(f"== {ds}: share of scenes where the cheap edit beats the humans' best enhancement")
        print(o.pivot(index="metric", columns="edit", values="beat_best").round(2).to_string())

"""Within-scene kappa (with 95% scene-bootstrap CI) for metrics added after the main run (RALI, MATLAB UCIQE), with
eval_mos.py's own statistics. Kept separate because eval_mos.py shares one random stream across all metrics in file
order: rerunning it with new score files would shift the published intervals of the old ones.
Writes mos/results/extra_<ds>.csv. Usage: python -I mos/eval_extra.py uid2021 saud saud2 sota [--metrics ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from mos.eval_mos import B, pair_kappa, scene_stats  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=["rali", "uciqe_matlab_fix", "uciqe"])
    a = ap.parse_args()
    for ds in a.ds:
        man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
        rows = []
        for m in a.metrics:
            f = REPO / "mos/scores" / ds / f"{m}.csv"
            if not f.exists():
                continue
            d = man.merge(pd.read_csv(f), on="id")
            assert len(d) == len(man), (ds, m, len(d))
            d["s"] = np.where(d.lower_better, -d.score, d.score)
            st = np.array([scene_stats(g)[4:] for _, g in d.groupby("scene")])
            rng = np.random.default_rng(0)
            bk = [pair_kappa(*st[rng.integers(0, len(st), len(st))].sum(0)) for _ in range(B)]
            rows.append(dict(metric=m, kappa=pair_kappa(*st.sum(0)), kappa_lo=np.percentile(bk, 2.5),
                             kappa_hi=np.percentile(bk, 97.5), n=len(d)))
        out = pd.DataFrame(rows)
        out.round(4).to_csv(REPO / "mos/results" / f"extra_{ds}.csv", index=False)
        print(f"== {ds}\n{out.round(3).to_string(index=False)}")

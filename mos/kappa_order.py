"""Does within-scene kappa depend on the manifest row order? eval_mos.py pools pairs (i, j) with i < j in file order,
so "first image" is the earlier row. The order-free version counts both orders of every pair, which makes both
marginals 0.5 and kappa = 2 * agreement - 1 on pairs neither side ties.
Writes mos/results/kappa_order.csv (set, metric, k_pub, k_sym, diff_abs, tie_rate).
Usage: python -I mos/kappa_order.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mos"))
from eval_mos import pair_kappa, scene_stats  # noqa: E402

rows = []
for ds in ["uid2021", "saud", "saud2", "sota"]:
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    for f in sorted((REPO / "mos/scores" / ds).glob("*.csv")):
        d = man.merge(pd.read_csv(f), on="id")
        d["s"] = np.where(d.lower_better, -d.score, d.score)
        if len(d) != len(man) or not np.isfinite(d.s).all():
            continue
        st = np.array([scene_stats(g) for _, g in d.groupby("scene")])
        n, a, h, ah = st[:, 4:].sum(0)
        k_pub, k_sym = pair_kappa(n, a, h, ah), 2 * (n - a - h + 2 * ah) / n - 1
        rows.append(dict(set=ds, metric=f.stem, k_pub=k_pub, k_sym=k_sym, diff_abs=abs(k_pub - k_sym),
                         tie_rate=1 - n / st[:, 2].sum()))
out = pd.DataFrame(rows)
assert abs(pair_kappa(10, 5, 5, 5) - 1) < 1e-12  # balanced marginals: both definitions give 1 for full agreement
out.round(4).to_csv(REPO / "mos/results/kappa_order.csv", index=False)
print(out.groupby("set").diff_abs.describe().round(3))

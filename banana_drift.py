"""Banana100-style drift test on the iterative Nano Banana chains (out_banana_<chain>/scores/*.csv).

Scores are min-max normalised per metric and oriented so higher = better. delta_k = mean over images of
(norm score at step k - norm score at step 1); step 1 is the baseline because step 0 is the 512 px input and every
generated step is 1200x896. A metric "detects" the drift (Banana100's success criterion) iff every delta_k < 0.
Also reports %neg_k: share of images whose own delta_k < 0.
Usage: python banana_drift.py enhance|replica|nb2_enhance|nb2_replica
"""
import sys
from pathlib import Path

import pandas as pd


def drift(df, ks):
    """df: one metric's rows (image, step, score, lower_better). Returns {f"d{k}", f"neg{k}"} per k."""
    s = df["score"] * (-1 if df["lower_better"].iloc[0] else 1)
    s = (s - s.min()) / (s.max() - s.min() or 1)
    w = df.assign(n=s).pivot_table(index="image", columns="step", values="n")
    out = {}
    for k in ks:
        d = (w[k] - w[1]).dropna()
        out[f"d{k}"], out[f"neg{k}"] = d.mean(), (d < 0).mean() * 100
    return out


def selfcheck():
    rows = [dict(image=i, step=s, score=10 - s, lower_better=False) for i in "ab" for s in range(1, 21)]
    r = drift(pd.DataFrame(rows), [5, 20])
    assert r["d5"] < 0 and r["d20"] < r["d5"] and r["neg20"] == 100
    r = drift(pd.DataFrame(rows).assign(lower_better=True), [5])  # falling "lower is better" score = improving
    assert r["d5"] > 0 and r["neg5"] == 0


if __name__ == "__main__":
    selfcheck()
    chain = sys.argv[1]
    ks = [5, 10, 20] + ([50] if chain.endswith("replica") else [])
    rows = {}
    for f in sorted(Path(f"out_banana_{chain}/scores").glob("*.csv")):
        df = pd.read_csv(f)
        df = df[df["step"] >= 1]
        if df["step"].max() >= max(ks):
            rows[f.stem] = drift(df, ks)
    if not rows:
        sys.exit(f"no scored metrics with step {max(ks)} in out_banana_{chain}/scores yet")
    t = pd.DataFrame(rows).T
    t["detects"] = (t[[f"d{k}" for k in ks]] < 0).all(axis=1)
    t = t.sort_values(f"d{max(ks)}")
    pd.set_option("display.width", 200)
    print(t.round({**{f"d{k}": 3 for k in ks}, **{f"neg{k}": 0 for k in ks}}).to_string())
    print(f"\n{t['detects'].sum()}/{len(t)} metrics detect the {chain} drift (all deltas < 0)")
    t.to_csv(f"out_banana_{chain}/drift.csv")

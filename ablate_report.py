"""Metric x condition heatmap: fraction of images a metric rates BETTER after the degradation than clean.

Writes out_ablate/fooled_by_condition.csv and out_ablate/fooled_heatmap.png.
Usage: python ablate_report.py --out out_ablate
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablate import conditions

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out_ablate")
out = Path(ap.parse_args().out)

df = pd.concat([pd.read_csv(p) for p in sorted((out / "scores").glob("*.csv"))], ignore_index=True)
df["better"] = df.score.where(~df.lower_better.astype(bool), -df.score)  # higher = metric prefers
clean = df[df["mode"] == "clean"][["metric", "image", "better"]].rename(columns={"better": "clean"})
deg = df[df["mode"] != "clean"].merge(clean, on=["metric", "image"])
deg["fooled"] = deg.better > deg.clean

cols = [c for c, _ in conditions() if c in set(deg["mode"])]
tab = deg.groupby(["metric", "mode"]).fooled.mean().unstack()[cols]
tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index]
tab.round(2).to_csv(out / "fooled_by_condition.csv")

fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 3, 0.36 * len(tab) + 2.5))
ax.imshow(tab.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(cols)), cols, rotation=70, ha="right", fontsize=7)
ax.set_yticks(range(len(tab)), tab.index, fontsize=8)
for i, row in enumerate(tab.values):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "black")
ax.set_title("% of 90 UIEB images rated better than clean after one degradation (rows: most fooled first)")
fig.tight_layout()
fig.savefig(out / "fooled_heatmap.png", dpi=160)
print(tab.mean(axis=1).round(2).to_string())
print("wrote", out / "fooled_heatmap.png")

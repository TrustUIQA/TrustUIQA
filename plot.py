"""Summarise attack results.

fooled.csv : per mode x metric, fraction of images the metric rates BETTER at the last step than clean.
trajectories.png : per mode, mean metric change vs step, in units of clean-image std, sign-flipped so up = "metric says better".
Usage: python plot.py --out out
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
out = Path(ap.parse_args().out)

att = pd.read_csv(out / "attack.csv").melt(["mode", "image", "step"], ["uiqm", "uciqe"], "metric", "score")
att["lower_better"] = False
df = pd.concat([att, *(pd.read_csv(p) for p in sorted((out / "scores").glob("*.csv")))], ignore_index=True)
df["better"] = df.score.where(~df.lower_better.astype(bool), -df.score)  # higher = metric prefers

clean = df[df.step == 0].rename(columns={"better": "clean"})
clean["sd"] = clean.groupby(["mode", "metric"]).clean.transform("std")
df = df.merge(clean[["mode", "metric", "image", "clean", "sd"]], on=["mode", "metric", "image"])
df["delta"] = df.better - df.clean
df["z"] = df.delta / df.sd

last = df[df.step == df.step.max()]
fooled = last.groupby(["mode", "metric"]).delta.apply(lambda d: (d > 0).mean()).unstack(0).sort_values("naive")
fooled.round(2).to_csv(out / "fooled.csv")
print("fraction of images rated better after attack than clean:\n", fooled.round(2))

modes = sorted(df["mode"].unique())
fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
for ax, mode in zip(axes, modes):
    traj = df[df["mode"] == mode].groupby(["metric", "step"]).z.mean().unstack(0)
    for m in traj:
        bold = m in ("uiqm", "uciqe")
        ax.plot(traj.index, traj[m], lw=2.5 if bold else 1, alpha=1 if bold else 0.6, label=m)
    ax.axhline(0, c="k", lw=0.8)
    ax.set(title=mode, xlabel="attack step", ylabel="change vs clean (clean std units, up = rated better)")
axes[-1].legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(out / "trajectories.png", dpi=150)
print("wrote", out / "trajectories.png")

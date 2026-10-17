"""Figure: how often each metric prefers the degraded image on the four conditions humans clearly reject.

Same rated trials as analyze_votes.py (gold-passing raters). Usage (from repo root): python study/fig_fooling.py
"""
import glob
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "study")
from analyze_votes import load, checks, picked_clean, scores, SRC, CLEAN

BAD = ["chain_20", "noise_strong", "chain_5", "posterize_strong"]  # humans prefer the edit <= 17%
UW = {"uiqm", "uciqe", "uranker"}  # underwater-specific, highlighted
FR = {"psnr_gt", "ssim_gt", "lpips_gt", "dists_gt"}  # need the UIEB reference

if __name__ == "__main__":
    ss = list({s["run"]: s for p in glob.glob("study/responses/*.txt") for s in load(p)}.values())
    trials = [v for s in ss if checks(s)["pass"] for v in s["votes"] if v["k"] == "t" and v["c"] in BAD]
    sc = scores()
    hum, met = [not picked_clean(v) for v in trials], defaultdict(list)
    for v in trials:
        d, mode, step = SRC[v["c"]]
        cm, cs = CLEAN[d]
        for m in {k[1] for k in sc}:
            a, b = sc.get((d, m, mode, v["img"], step)), sc.get((d, m, cm, v["img"], cs))
            if a and b and a[0] != b[0]:
                met[m].append((a[0] < b[0]) if a[1] else (a[0] > b[0]))
    rate = sorted(((100 * sum(x) / len(x), m) for m, x in met.items()), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([m for _, m in rate], [r for r, _ in rate],
           color=["tab:red" if m in UW else "tab:gray" if m in FR else "tab:blue" for _, m in rate])
    ax.axhline(100 * sum(hum) / len(hum), ls="--", c="k", lw=1, label=f"humans ({100 * sum(hum) / len(hum):.0f}%)")
    ax.set_ylabel("% pairs preferring\ndegraded image")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("study/fig_fooling.pdf")
    print("\n".join(f"{m:12}{r:5.0f}%" for r, m in rate), f"\nhumans {100 * sum(hum) / len(hum):.0f}%, n={len(hum)}")

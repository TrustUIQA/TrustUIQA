"""Paper figures + LaTeX-ready tables from RESULTS_ROOT (run after notebooks 01, 05, 06).

Writes RESULTS_ROOT/paper/{fig_*.pdf, tab_*.tex}. Each figure is skipped with a
message if its inputs are missing, so this runs on whatever results exist.

    python scripts/make_figures.py
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uwiqa import RESULTS_ROOT  # noqa: E402

OUT = RESULTS_ROOT / "paper"
OUT.mkdir(parents=True, exist_ok=True)
COL_W, FULL_W = 3.5, 7.16  # IEEE column / page width (in)
# Categorical slots, fixed order (validated reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
NICE = {"uranker": "URanker", "liqe_mix": "LIQE-mix", "liqe": "LIQE", "musiq": "MUSIQ", "arniqa": "ARNIQA",
        "uciqe": "UCIQE", "uiqm": "UIQM", "topiq_nr": "TOPIQ-NR", "uciqe_legacy": "UCIQE (legacy)",
        "clipiqa+": "CLIP-IQA+", "tres": "TReS", "qualiclip": "QualiCLIP", "maniqa": "MANIQA",
        "clipiqa": "CLIP-IQA", "niqe": "NIQE", "brisque": "BRISQUE", "clahe": "CLAHE", "udcp": "UDCP",
        "fusion": "Fusion", "gray_world": "Gray-world", "funiegan": "FUnIE-GAN", "null_jpeg95": "Null (JPEG q95)"}
nice = lambda k: NICE.get(k, k)


def have(*paths):
    miss = [p for p in paths if not p.exists()]
    if miss:
        print(f"  skip: missing {', '.join(str(m.relative_to(RESULTS_ROOT)) for m in miss)}")
    return not miss


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- O1
print("O1: metric vs MOS on UID2021")
p = RESULTS_ROOT / "tables" / "o1_uid2021.csv"
if have(p):
    t = pd.read_csv(p).sort_values("srcc")
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    y = np.arange(len(t))
    ax.barh(y, t.srcc, height=0.62, color=C[0], label="SRCC (all images)")
    ax.errorbar(t.srcc, y, xerr=[t.srcc - t.srcc_lo, t.srcc_hi - t.srcc], fmt="none", ecolor=INK, lw=0.7, capsize=1.5)
    ax.scatter(t.intra_scene_srcc, y, marker="D", s=12, color=C[1], zorder=3, label="Mean intra-scene SRCC")
    ax.set_yticks(y, [nice(m) for m in t.metric])
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Spearman correlation with MOS")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_o1_srcc.pdf")
    rows = [f"{nice(r.metric)} & {r.srcc:.3f} & [{r.srcc_lo:.3f}, {r.srcc_hi:.3f}] & {r.plcc:.3f} & "
            f"{r.krcc:.3f} & {r.intra_scene_srcc:.3f} \\\\" for r in t.sort_values("srcc", ascending=False).itertuples()]
    (OUT / "tab_o1.tex").write_text("\n".join(rows) + "\n")

# ---------------------------------------------------------------- O2
print("O2: over-enhancement sweeps")
p = RESULTS_ROOT / "stress" / "o2_uid2021_raw.csv"
if have(p):
    r = pd.read_csv(p)
    base = r[r.strength == 0].set_index(["image", "sweep", "metric"])["score"]
    r = r.join(base.rename("s0"), on=["image", "sweep", "metric"])
    scale = r.groupby("metric")["score"].transform("std").replace(0, np.nan)
    r["dz"] = (r.score - r.s0) / scale  # change from the unprocessed image, in metric std units
    sweeps = ["contrast", "saturation", "unsharp", "red_shift"]
    titles = {"contrast": "Contrast stretch", "saturation": "Saturation boost",
              "unsharp": "Unsharp masking", "red_shift": "Red over-compensation"}
    metrics = [m for m in ["uciqe", "uiqm", "uranker", "topiq_nr", "musiq", "liqe"] if m in set(r.metric)]
    fig, axs = plt.subplots(1, 4, figsize=(FULL_W, 1.75), sharey=True)
    for ax, sw in zip(axs, sweeps):
        g = r[r.sweep == sw].groupby(["metric", "strength"])["dz"].mean().unstack(0)
        for i, m in enumerate(metrics):
            ax.plot(g.index, g[m], color=C[i], marker="o", ms=2.5, label=nice(m))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(titles[sw])
        ax.set_xlabel("Distortion strength $s$")
    axs[0].set_ylabel("Score change (std units)")
    axs[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save(fig, "fig_o2_sweeps.pdf")

# ---------------------------------------------------------------- O3
print("O3: per-image utility deltas")
p = RESULTS_ROOT / "utility" / "ruod_pred" / "main" / "scores.csv"
if have(p):
    sc = pd.read_csv(p)
    w = sc.pivot_table(index="image", columns="method", values="ap")
    d = w.drop(columns="raw").sub(w["raw"], axis=0)
    order = ["null_jpeg95"] + d.drop(columns="null_jpeg95").mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    parts = ax.violinplot([d[m].values for m in order], showextrema=False, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(MUTED if order[i] == "null_jpeg95" else C[0])
        b.set_alpha(0.55)
        b.set_edgecolor("none")
    ax.scatter(np.arange(1, len(order) + 1), [d[m].mean() for m in order], color=INK, s=8, zorder=3, label="mean")
    ax.axhline(0, color=MUTED, lw=0.6)
    tau = float(np.quantile(np.abs(d["null_jpeg95"]), 0.95))
    ax.axhspan(-tau, tau, color=GRID, alpha=0.8, zorder=0, label=f"noise band ±{tau:.2f}")
    two_line = {"null_jpeg95": "Null\n(JPEG q95)", "funiegan": "FUnIE-\nGAN", "gray_world": "Gray-\nworld"}
    ax.set_xticks(np.arange(1, len(order) + 1), [two_line.get(m, nice(m)) for m in order], fontsize=6.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel(r"$\Delta$AP$_{50:95}$ vs raw")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_o3_delta.pdf")

# ---------------------------------------------------------------- O5
print("O5: selective-enhancement policies")
for sfx, tag in (("_map", "mAP-tuned"), ("", "image-AP-tuned")):
    pt, pd_ = RESULTS_ROOT / "policies" / f"policies_test{sfx}.csv", RESULTS_ROOT / "policies" / f"policies_duo{sfx}.csv"
    if not (pt.exists() and pd_.exists()):
        continue
    T, D = pd.read_csv(pt).set_index("policy"), pd.read_csv(pd_).set_index("policy")
    learned = [k for k in T.index if "@" in k]
    bt = pd.read_csv(RESULTS_ROOT / "policies" / f"bootstrap_test{sfx}.csv")
    best_learned = bt.policy[bt.policy.str.contains("@")].iloc[0] if bt.policy.str.contains("@").any() else None
    fixed = [k for k in T.index if k.startswith("all_") and k != "all_raw"]
    naive = [k for k in T.index if k.endswith("|naive")]
    rows = [("Always raw", "all_raw"), ("Best fixed enhancer", max(fixed, key=lambda k: T.at[k, "map"])),
            ("Worst fixed enhancer", min(fixed, key=lambda k: T.at[k, "map"])),
            ("NR-IQA choice (best)", max(naive, key=lambda k: T.at[k, "map"])),
            ("NR-IQA choice (worst)", min(naive, key=lambda k: T.at[k, "map"]))]
    if best_learned:
        rows.append(("Learned (ours)", best_learned))
    rows += [("Noise oracle", "noise_oracle"), ("Oracle", "oracle")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 1.9), sharey=True)
    for ax, tab, ttl in ((axs[0], T, "RUOD pred-test (in-domain)"), (axs[1], D, "DUO-clean (cross-dataset)")):
        base = tab.at["all_raw", "map"]
        vals = [tab.at[k, "map"] - base if k in tab.index else np.nan for _, k in rows]
        cols = [C[0] if lbl == "Learned (ours)" else MUTED if "racle" in lbl else C[7] if v < 0 else C[2]
                for (lbl, _), v in zip(rows, vals)]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=cols, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + (0.003 if v >= 0 else -0.003), yi, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6, color=INK)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(ttl)
        ax.set_xlabel(r"$\Delta$mAP$_{50:95}$ vs always-raw")
        ax.grid(axis="y", visible=False)
        lo = min(np.nanmin(vals) * 1.25, -0.02)
        ax.set_xlim(lo, max(np.nanmax(vals) * 1.6, 0.04))
    axs[0].set_yticks(np.arange(len(rows)), [lbl for lbl, _ in rows])
    axs[0].invert_yaxis()
    save(fig, f"fig_o5_policies{sfx}.pdf")
    keys = [k for _, k in rows]
    lines = [f"{lbl} & \\texttt{{{k.replace('_', chr(92) + '_')}}} & {T.at[k, 'frac_enhanced']:.2f} & {T.at[k, 'map']:.4f} & "
             f"{T.at[k, 'map'] - T.at['all_raw', 'map']:+.4f} & {D.at[k, 'map'] if k in D.index else float('nan'):.4f} & "
             f"{(D.at[k, 'map'] - D.at['all_raw', 'map']) if k in D.index else float('nan'):+.4f} \\\\"
             for lbl, k in rows]
    (OUT / f"tab_o5{sfx}.tex").write_text("\n".join(lines) + "\n")
    print(f"  ({tag}) best learned on val: {best_learned}")

# ---------------------------------------------------------------- O4
print("O4: predictor summary tables")
for s in ("test", "duo"):
    p = RESULTS_ROOT / "predictor" / f"summary_{s}.csv"
    if have(p):
        t = pd.read_csv(p)
        t.to_csv(OUT / f"tab_o4_{s}.csv", index=False)
print(f"\nall outputs in {OUT}")
for f in sorted(glob.glob(str(OUT / "*"))):
    print("  ", f.split("/")[-1].split("\\")[-1])

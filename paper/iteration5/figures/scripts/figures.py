"""All figures of iteration 5, redrawn from the raw result files (guide A8: Okabe-Ito colours plus markers, vector PDF).

img_grid.pdf        Fig 1: three UIEB photos x eight panels (raw, reference, cheap edits, four Nano Banana chain ends)
                    with the scores of eight metrics, two per metric group          (gitignored: pixels)
fig_kappa.pdf       within-scene kappa of the eight Fig 1 metrics and of all 73, on the five human references
fig_study.pdf       people vs metrics on the 12 edits of our study, one row per edit
fig_drift.pdf       noise, colour change and normalised scores along the four Nano Banana chains, all 73 scores as a band
img_drift.pdf       appendix: centre crops of one photo along the four chains          (gitignored: pixels)
Only UIEB photos and our own generated or edited copies of them appear. No MOS-set image is ever drawn.
Run from the repo root: python -I paper/iteration5/figures/scripts/figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import *  # noqa: E402,F403

F = IT / "figures"
OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          red="#D55E00", purple="#CC79A7", grey="#999999")
COL, TXT = 3.3, 6.9  # WACV column and text width, inches
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",  # Times-like, ships with matplotlib
                     "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2})
# Fig 1 metrics: per group, the default-checkpoint metrics with the highest and lowest mean kappa over the five
# references (fig1_metrics() recomputes and asserts this). One colour and marker per metric in every figure.
SEL = {"uiqm": (OK["orange"], "v"), "uciqe": (OK["red"], "^"), "nr_uiqa": (OK["blue"], "s"), "uranker": (OK["sky"], "X"),
       "visualquality_r1": (OK["black"], "*"), "qalign_8bit": (OK["green"], "D"), "rali": (OK["purple"], "P"),
       "maniqa": (OK["grey"], "o")}
GROUPS = ["hand-crafted underwater", "learned underwater", "vision-language", "general-purpose"]
GROUP_SHORT = {"hand-crafted underwater": "Hand-crafted UW", "learned underwater": "Learned UW",
               "vision-language": "Vision- language",
               "general-purpose": "General"}  # the first space becomes the line break in img_grid


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def fig1_metrics():
    """Best and worst default checkpoint per group by mean kappa over the five references (variants with '-' or
    '_matlab' and second checkpoints are left out, so each name is a metric as papers use it)."""
    w = kappa_wide().dropna()
    out = []
    for g in GROUPS:
        ms = [m for m in w.index if group(m) == g and "-" not in m and "matlab" not in m and not m.endswith(("_512", "_res50"))]
        mean = w.loc[ms].mean(axis=1).sort_values(ascending=False)
        out += [mean.index[0], mean.index[-1]]
    return out


# ------------------------------------------------------------------ Fig 1: image grid with scores
DRIFT_IDS = None
PANELS = [  # key, title, score folders, mode, step, image path template
    ("raw", "Raw photo", ["out_ablate"], "clean", 0, "out_ablate/clean/{i}/00.png"),
    ("ref", "UIEB reference\n(people's choice)", ["out_gt", "scratch/pool_fig1_gt"], "clean", 0, "out_gt/clean/{i}/00.png"),
    ("edit", "Simple edit\nS+S+C, once", ["out_ablate", "scratch/pool_fig1_edit"], "saturation+sharpen+clahe@moderate", 1,
     "out_ablate/saturation+sharpen+clahe@moderate/{i}/01.png"),
    ("chain", "S+S+C 20 times\n(study, people reject)", ["out"], "naive", 20, "out/naive/{i}/20.png"),
    ("prorep", "Pro replica\nstep 50", ["out_banana_replica"], "nbpro_replica", 50, "out_banana/nbpro_replica/{i}/50.png"),
    ("nb2rep", "NB2 replica\nstep 50", ["out_banana_nb2_replica"], "nb2_replica", 50, "out_banana_nb2_replica/nb2_replica/{i}/50.png"),
    ("proenh", "Pro enhance\nstep 20", ["out_banana_enhance"], "nbpro_enhance", 20, "out_banana_enhance/nbpro_enhance/{i}/20.png"),
    ("nb2enh", "NB2 enhance\nstep 20", ["out_banana_nb2_enhance"], "nb2_enhance", 20, "out_banana_nb2_enhance/nb2_enhance/{i}/20.png"),
]


def panel_scores(metrics):
    """DataFrame (image, panel) x metric of raw scores (all eight metrics are higher = better, asserted)."""
    rows = {}
    for key, _, dirs, mode, step, _ in PANELS:
        for m in metrics:
            for d in dirs:
                for sub in ("scores", "scores_ext"):
                    f = REPO / d / sub / f"{m}.csv"
                    if not f.exists():
                        continue
                    x = pd.read_csv(f, dtype={"image": str})
                    x = x[(x["mode"] == mode) & (x.step == step)]
                    assert not x.lower_better.any(), (m, f)
                    for r in x.itertuples():
                        rows.setdefault((r.image, key), {})[m] = r.score
    s = pd.DataFrame.from_dict(rows, orient="index")
    s.index = pd.MultiIndex.from_tuples(s.index, names=["image", "panel"])
    return s[metrics]


def grid_choice(s, ids):
    """Rank the drift photos by how many (metric, non-reference panel) scores beat the reference: the figure shows
    the top three (stated in the caption, with the rate over all photos)."""
    wins = {}
    for i in ids:
        t = s.loc[i]
        wins[i] = int((t.drop("ref") > t.loc["ref"]).values.sum())
    order = sorted(ids, key=lambda i: (-wins[i], i))
    return order, wins


def img_grid():
    mets = fig1_metrics()
    assert mets == list(SEL), (mets, list(SEL))
    ids = sorted(pd.read_csv(REPO / "out_banana_replica/quant.csv", dtype={"image": str}).image.unique())
    s = panel_scores(mets)
    s = s.loc[ids]
    assert s.notna().all().all() and len(s) == len(ids) * len(PANELS), s.isna().sum()
    order, wins = grid_choice(s, ids)
    show = order[:3]
    n_cmp = len(mets) * (len(PANELS) - 2 + 1)  # 7 non-reference panels x 8 metrics per photo
    beat_all = sum(wins.values()) / (len(ids) * n_cmp)
    per_panel = {k: float(np.mean([(s.loc[(i, k)] > s.loc[(i, "ref")]).mean() for i in ids])) for k, *_ in PANELS if k != "ref"}
    nP, nM = len(PANELS), len(mets)
    lab_w, gl_w, gap = 1.22, 0.5, 0.03  # label column (group name, then metric name), gap between panels
    img_w = (TXT - lab_w) / nP
    size = {i: Image.open(REPO / PANELS[0][5].format(i=i)).size for i in show}  # each row keeps its photo's frame
    img_h = {i: img_w * size[i][1] / size[i][0] for i in show}
    row_h, head, sep = 0.092, 0.30, 0.13
    H = head + sum(img_h[i] + nM * row_h + sep for i in show)
    fig = plt.figure(figsize=(TXT, H))
    cmap = LinearSegmentedColormap.from_list("rank", ["#FFFFFF", "#9ECAE1", "#2171B5"])
    for col, (key, title, *_rest) in enumerate(PANELS):
        fig.text((lab_w + (col + 0.5) * img_w) / TXT, 1 - 0.02 / H, title, ha="center", va="top", fontsize=6.6, linespacing=1.05)
    y0 = H - head  # top of the current photo's block, inches from the bottom
    for i in show:
        ih = img_h[i]
        for col, (key, _, _, _, _, path) in enumerate(PANELS):
            ax = fig.add_axes([(lab_w + col * img_w + gap / 2) / TXT, (y0 - ih) / H, (img_w - gap) / TXT, (ih - 0.01) / H])
            im = Image.open(REPO / path.format(i=i)).convert("RGB")  # the editors return about the input frame
            ax.imshow(im.resize((300, round(300 * size[i][1] / size[i][0])), Image.LANCZOS), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(key == "ref")
                sp.set_color(OK["green"])
                sp.set_linewidth(1.6)
        fig.text(0.02 / TXT, (y0 - ih / 2) / H, f"UIEB photo {i}", ha="left", va="center", fontsize=6.8, fontweight="bold")
        # score table under the images: one row per metric, cell shade = rank of that panel among the eight
        ax = fig.add_axes([0, (y0 - ih - nM * row_h - 0.02) / H, 1, nM * row_h / H])
        ax.set_xlim(0, TXT)
        ax.set_ylim(nM, 0)
        ax.axis("off")
        t = s.loc[i]
        for k, m in enumerate(mets):
            ranks = t[m].rank(ascending=False, method="min")  # 1 = scored highest of the eight panels
            c, mk = SEL[m]
            if k % 2 == 0:
                if k:
                    ax.plot([0.02, TXT], [k, k], c="#BBBBBB", lw=0.4)
                ax.text(0.02, k + 1, GROUP_SHORT[group(m)].replace(" ", "\n", 1), ha="left", va="center", fontsize=5.2,
                        color="#444444", linespacing=0.95)
            ax.plot(gl_w + 0.03, k + 0.5, mk, c=c, ms=3.0, mew=0)
            ax.text(gl_w + 0.09, k + 0.5, f"{name(m)} $\\uparrow$", ha="left", va="center", fontsize=5.6, color="k")
            for col, (key, *_r) in enumerate(PANELS):
                x0 = lab_w + col * img_w + gap / 2
                rk = ranks[key]
                ax.add_patch(Rectangle((x0, k + 0.06), img_w - gap, 0.88, color=cmap((nP - rk) / (nP - 1)), lw=0))
                v = t.loc[key, m]
                txt = f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
                ax.text(x0 + (img_w - gap) / 2, k + 0.52, txt.replace("-", "−"), ha="center", va="center", fontsize=5.6,
                        color="white" if rk == 1 else "k", fontweight="bold" if rk == 1 else "normal")
        y0 -= ih + nM * row_h + sep
    save(fig, "img_grid.pdf")
    return dict(gridIds=show, gridWins=[wins[i] for i in show], gridCmp=n_cmp, gridN=len(ids),
                gridBeatAll=round(beat_all, 4), gridPerPanel=per_panel, gridMetrics=mets)


# ------------------------------------------------------------------ Fig 2: kappa heatmap
def fig_kappa():
    w = kappa_wide()
    mets = list(SEL)
    rows = [(name(m), w.loc[m, REFS].values.astype(float), m) for m in mets]
    rows.sort(key=lambda r: -np.nanmin(r[1]))
    agg = [("Median over metrics", w[REFS].median().values, None), ("Best over metrics", w[REFS].max().values, None)]
    data = np.array([r[1] for r in rows] + [a[1] for a in agg])
    worst = np.nanmin(data[:len(rows)], axis=1)
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    cmap = LinearSegmentedColormap.from_list("div", ["#B2182B", "#FFFFFF", "#2166AC"])
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.9)
    full = np.column_stack([data, np.r_[worst, [np.nan] * len(agg)]])
    ax.imshow(full, cmap=cmap, norm=norm, aspect="auto")
    for (y, x), v in np.ndenumerate(full):
        if np.isnan(v):
            continue
        ax.text(x, y, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.3,
                color="white" if abs(v) > 0.55 else "k", fontweight="bold" if x == 5 else "normal")
    for y, (_, vals, m) in enumerate(rows):  # cells on the metric's own training set or lab: outlined
        for x, r in enumerate(REFS):
            if m in TRAINED and GROUP_OF[r] == GROUP_OF[TRAINED[m]]:
                ax.add_patch(Rectangle((x - 0.47, y - 0.45), 0.94, 0.9, fill=False, ec="k", lw=0.8, ls="--" if r != TRAINED[m] else "-"))
    ax.axhline(len(rows) - 0.5, c="k", lw=0.8)
    ax.axvline(len(REFS) - 0.5, c="k", lw=0.8)
    ax.set_xticks(range(6), [REF_NAME[r].replace("Our study", "Study") for r in REFS] + ["Worst"])
    labels = [r[0] for r in rows] + [a[0] for a in agg]
    ax.set_yticks(range(len(labels)), labels)
    for t, (_, _, m) in zip(ax.get_yticklabels(), rows + agg):
        if m:
            t.set_color("k")
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y, (_, _, m) in enumerate(rows):
        ax.plot(-0.95, y, SEL[m][1], c=SEL[m][0], ms=3.5, mew=0, clip_on=False, transform=ax.transData)
    save(fig, "fig_kappa.pdf")


# ------------------------------------------------------------------ Fig 3: study, one row per edit
COND = {"clahe_mild": "CLAHE, mild", "contrast_mild": "Contrast, mild", "sharpen_mild": "Sharpen, mild", "sat_mild": "Saturation, mild",
        "sat_mod": "Saturation, moderate", "hue_mod": "Hue shift, moderate", "noise_mild": "Noise, mild", "red_strong": "Red boost, strong",
        "posterize_strong": "Posterize, strong", "chain_5": "S+S+C 5 times", "noise_strong": "Noise, strong", "chain_20": "S+S+C 20 times"}


def fig_study():
    sp = pd.read_csv(R / "study_pairs.csv", dtype={"image": str})
    one = sp[sp.metric == "uciqe"].assign(k=lambda d: d.human_pref_edit * d.n_votes)
    h = one.groupby("condition").agg(k=("k", "sum"), n=("n_votes", "sum"))
    h["p"] = h.k / h.n
    ci = pref_ci().loc[h.index]
    assert np.allclose(ci.pref_edit, h.p, atol=1e-3)
    h["lo"], h["hi"] = ci.lo, ci.hi
    h = h.sort_values("p")
    sel = ["uciqe", "uiqm", "qalign_8bit", "rali"]
    mp = sp[sp.metric.isin(sel)].groupby(["condition", "metric"]).metric_prefers_edit.mean().unstack()
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    y = np.arange(len(h))
    rej = (h.p < 0.5).values
    for yy in y[rej]:
        ax.axhspan(yy - 0.5, yy + 0.5, color=OK["yellow"], alpha=0.22, lw=0)
    ax.barh(y, 100 * h.p, height=0.62, color="#D9D9D9", label="People (95% CI)", zorder=1)
    ax.errorbar(100 * h.p, y, xerr=[100 * (h.p - h.lo), 100 * (h.hi - h.p)], fmt="none", ecolor="#555555", elinewidth=0.6, capsize=1.2, zorder=2)
    off = dict(zip(sel, [-0.27, -0.09, 0.09, 0.27]))
    for m in sel:
        c, mk = SEL[m]
        ax.plot(100 * mp.loc[h.index, m], y + off[m], mk, c=c, ms=3.6, mew=0, label=name(m), zorder=3)
    ax.axvline(50, c="k", lw=0.6, ls=":")
    ax.set_yticks(y, [COND[c] for c in h.index])
    ax.set_ylim(-0.6, len(h) - 0.4)
    ax.set_xlim(-3, 103)
    ax.set_xticks(range(0, 101, 20))
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("Share that prefers the edit to the raw photo (%)")
    ax.text(76, 0, "most votes for\nthe raw photo", ha="center", va="center", fontsize=6.5, color="#7A6A00")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), frameon=False, ncol=5, handletextpad=0.15, columnspacing=0.6, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_study.pdf")


# ------------------------------------------------------------------ Fig 4: drift curves
def norm_curves(chain, metric):
    """Per image min-max normalised, oriented score per step, minus step 1 (banana_drift.py drift()): mean over images."""
    d = drift_scores(chain, metric)
    s = d.score * (-1 if d.lower_better.iloc[0] else 1)
    d = d.assign(n=(s - s.min()) / (s.max() - s.min() or 1)).pivot_table(index="image", columns="step", values="n")
    d = d.sub(d[1], axis=0)
    return d.drop(columns=0).mean()


def fig_drift():
    lines = ["uciqe", "uiqm", "niqe", "musiq", "qalign_8bit", "rali"]  # NIQE and MUSIQ: two general metrics that rise too
    sty = {**SEL, "niqe": ("#882255", "h"), "musiq": ("#44AA99", "<")}  # Tol wine and teal, used nowhere else
    fig, axs = plt.subplots(2, 4, figsize=(TXT, 3.0), sharex="col", gridspec_kw=dict(hspace=0.15, wspace=0.42, height_ratios=[1, 1.35]))
    for j, (c, cname) in enumerate(CHAINS.items()):
        q = pd.read_csv(REPO / f"out_banana_{c}/quant.csv", dtype={"image": str})
        q = q[q.step > 0]
        m = q.groupby("step").mean(numeric_only=True)
        a = axs[0, j]
        a.plot(m.index, m.sigma / m.sigma.iloc[0], c="k", label="Noise $\\sigma$ / step 1 (left axis)")
        a.set_title(cname, fontsize=8)
        b = a.twinx()
        b.plot(m.index, m.dE1, "--", c=OK["orange"], label="$\\Delta E^*_{ab}$ to step 1 (right axis)")
        b.spines["right"].set_visible(True)
        b.spines["top"].set_visible(False)
        b.tick_params(axis="y", colors=OK["orange"], labelsize=7)
        if j == 0:
            a.set_ylabel("Noise / step 1")
        if j == 3:
            b.set_ylabel("$\\Delta E^*_{ab}$", color=OK["orange"])
        a.set_ylim(bottom=0.8)
        b.set_ylim(bottom=0)
        a = axs[1, j]
        d, _ = drift(c)
        allc = pd.DataFrame({mt: norm_curves(c, mt) for mt in d.index})
        a.fill_between(allc.index, allc.quantile(0.1, axis=1), allc.quantile(0.9, axis=1), color="#E3E3E3", lw=0,
                       label=f"All {allc.shape[1]} scores, 10% to 90%")
        a.plot(allc.index, allc.median(axis=1), c="#888888", lw=0.9, ls="--", label=f"Median of all {allc.shape[1]} scores")
        lo_, hi_ = a.get_ylim()
        for mt in lines:
            col, mk = sty[mt]
            cv = allc[mt]
            a.plot(cv.index, cv.values, c=col, marker=mk, ms=2.6, markevery=max(1, len(cv) // 6), lw=1.1, label=name(mt))
        a.axhline(0, c="k", lw=0.6)
        a.set_xlabel("Editing step")
        y0, y1 = a.get_ylim()
        a.axhspan(0, max(y1, 0.05), color=OK["red"], alpha=0.06, lw=0)
        a.set_ylim(y0, y1)
        if j == 0:
            a.set_ylabel("Score change from step 1\n(normalized, up = better)")
            a.text(0.03, 0.97, "score rises as\nnoise grows", transform=a.transAxes, ha="left", va="top", fontsize=6, color=OK["red"])
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    hb, lb = fig.axes[8].get_legend_handles_labels()  # first twin axis (orange)
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h1 + hb + h2, l1 + lb + l2, loc="lower center", bbox_to_anchor=(0.5, -0.155), ncol=5, frameon=False, handletextpad=0.3,
               columnspacing=1.0, fontsize=6.6)
    save(fig, "fig_drift.pdf")


# ------------------------------------------------------------------ appendix: crops along the chains
def img_drift(i):
    steps = [0, 1, 5, 10, 20, 50]
    sub = {"replica": "out_banana/nbpro_replica", "nb2_replica": "out_banana_nb2_replica/nb2_replica",
           "enhance": "out_banana_enhance/nbpro_enhance", "nb2_enhance": "out_banana_nb2_enhance/nb2_enhance"}
    big = Image.open(REPO / sub["replica"] / i / "50.png").size  # editors' output size for this photo
    fig, axs = plt.subplots(4, len(steps), figsize=(TXT, 4.75), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for row, c in enumerate(CHAINS):
        for col, k in enumerate(steps):
            a = axs[row, col]
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            p = REPO / sub[c] / i / f"{k:02d}.png"
            if p.exists():
                im = Image.open(p).convert("RGB").resize(big, Image.BICUBIC)  # step 0 is the 500 px input: same frame
                cx, cy = big[0] // 2, big[1] // 2
                a.imshow(im.crop((cx - 150, cy - 150, cx + 150, cy + 150)), interpolation="nearest")
            else:
                a.text(0.5, 0.5, "chain ends\nat step 20", ha="center", va="center", fontsize=7, transform=a.transAxes)
            if row == 0:
                a.set_title("Input" if k == 0 else f"Step {k}", fontsize=8)
            if col == 0:
                a.set_ylabel(CHAINS[c], fontsize=8)
    save(fig, "img_drift.pdf")
    return dict(driftCropId=i, driftFrame=list(big), driftInW=Image.open(REPO / sub["replica"] / i / "00.png").size[0])


if __name__ == "__main__":
    assert set(SEL) <= set(kappa_wide().index)
    t = img_grid()
    fig_kappa()
    fig_study()
    fig_drift()
    t.update(img_drift(t["gridIds"][0]))
    (IT / "notes/figure_facts.json").write_text(json.dumps(t, indent=1))
    print(t, sorted(p.name for p in F.glob("*.pdf")))

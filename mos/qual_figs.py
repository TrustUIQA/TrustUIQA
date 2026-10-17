"""Qualitative panels: original vs cheap edit (and, on MOS sets, vs the people's best enhancement) with metric scores.

  study  per study condition, the image whose edit most metrics prefer while raters preferred the original:
         original | edit, raters' share preferring the edit, each metric's score (* = metric prefers that image)
  <ds>   (uid2021, saud, sota) random scenes: raw | best-MOS enhancement | worst-MOS | edits, MOS where rated
Images come from research-only datasets, so figures go to uwdata/_figs (gitignored: never pushed), each with a CSV
of the plotted numbers. Usage: python -I mos/qual_figs.py study uid2021 [saud sota]
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "uwdata/_figs"
SHOW = ["uiqm", "uciqe", "uranker", "fdum", "ccf", "nr_uiqa", "quiq", "uiqavsi", "liqe", "qalign_8bit", "dists_gt"]


def img(p, h=256):
    x = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_AREA)


def panel(rows, path, title):
    """rows: list of lists of (image_path, caption)."""
    nr, nc = len(rows), max(map(len, rows))
    fig, ax = plt.subplots(nr, nc, figsize=(2.6 * nc, 3.6 * nr), squeeze=False)
    for i, r in enumerate(rows):
        for j in range(nc):
            a = ax[i, j]
            a.axis("off")
            if j < len(r):
                a.imshow(img(r[j][0]))
                a.set_title(r[j][1], fontsize=6.5, loc="left", family="monospace")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def caption(name, sc, best=None):
    lines = [name] + [f"{m:>11} {v:8.3f}{'*' if best and best.get(m) else ''}" for m, v in sc.items()]
    return "\n".join(lines)


def study():
    pairs = pd.read_csv(REPO / "mos/results/study_pairs.csv", dtype={"image": str})
    pairs = pairs[pairs.metric.isin(SHOW)]
    sys.path.insert(0, str(REPO / "study"))
    from analyze_votes import CLEAN, SRC
    rows, recs = [], []
    for c, g in pairs.groupby("condition"):
        # image where raters most preferred the original and most metrics still prefer the edit
        t = g.groupby("image").agg(h=("human_pref_edit", "first"), m=("metric_prefers_edit", "mean"))
        im = (t.m - t.h).idxmax()
        x = g[g.image == im].set_index("metric")
        d, mode, step = SRC[c]
        cmode, cstep = CLEAN[d]
        po, pe = (REPO / d / mo / im / f"{st:02d}.png" for mo, st in ((cmode, cstep), (mode, step)))
        if not po.exists():  # step file names may be unpadded
            po, pe = (REPO / d / mo / im / f"{st}.png" for mo, st in ((cmode, cstep), (mode, step)))
        pref = {m: bool(r.metric_prefers_edit) for m, r in x.iterrows()}
        h = x.human_pref_edit.iloc[0]
        rows.append([(po, caption(f"original  (raters {1 - h:.0%})", x.original.to_dict(),
                                  {m: not v for m, v in pref.items()})),
                     (pe, caption(f"{c}  (raters {h:.0%})", x.edited.to_dict(), pref))])
        recs.append(x.assign(condition=c, image=im).reset_index())
    for k in range(0, len(rows), 4):
        panel(rows[k:k + 4], OUT / f"study_{k // 4}.png", "Our 2AFC study: original vs edit (* = metric prefers)")
    pd.concat(recs).to_csv(OUT / "study_panels.csv", index=False)


def mos_set(ds, n=4, seed=0):
    man = pd.read_csv(REPO / f"mos/manifests/{ds}.csv")
    em = pd.read_csv(REPO / f"mos/manifests/{ds}_edits.csv")
    man["scene"], em["scene"] = man.scene.astype(str), em.scene.astype(str)
    sc = {}
    for m in SHOW:
        fs = [REPO / f"mos/scores/{x}/{m}.csv" for x in (ds, f"{ds}_edits")]
        if all(f.exists() for f in fs):
            sc[m] = pd.concat(pd.read_csv(f) for f in fs).set_index("id").score
    scenes = np.random.default_rng(seed).choice(sorted(set(em.scene) & set(man.scene)), n, replace=False)
    rows, recs = [], []
    for s in scenes:
        g = man[(man.scene == s) & (man.method != "Original")].sort_values("mos")
        e = em[em.scene == s]
        items = [(e[e.method == "raw"].iloc[0], "raw (unrated)"), (g.iloc[-1], f"best MOS {g.mos.iloc[-1]:.2f}"),
                 (g.iloc[0], f"worst MOS {g.mos.iloc[0]:.2f}")] + [(r, r.method) for _, r in e[e.method != "raw"].iterrows()]
        row = []
        for r, name in items:
            v = {m: x.get(r["id"], np.nan) for m, x in sc.items()}
            row.append((r["path"], caption(name.split("@")[0][:26], v)))
            recs.append(dict(scene=s, item=name, id=r["id"], **v))
        rows.append(row)
    panel(rows, OUT / f"{ds}_edits.png", f"{ds}: raw vs people's best/worst vs cheap edits")
    pd.DataFrame(recs).to_csv(OUT / f"{ds}_edits_panels.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for a in sys.argv[1:]:
        study() if a == "study" else mos_set(a)
        print("done", a, "->", OUT)

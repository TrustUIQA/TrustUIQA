"""Montages: every image of one scene (or one test image) side by side, every scored metric printed under each.

Each tile: the image, its label (method / edit / step, MOS where people rated it), then every metric scored on that
set as  name  score  #rank. #rank = the metric's rank of that image among the images it is compared with in the
montage (1 = the metric's favourite; lower-better metrics are re-oriented). Blue line = that metric's favourite.
Green frame = the people's favourite (top MOS, or the raters' choice in our study); grey frame = the reference.

  MOS sets      uid2021 saud saud2 sota: per scene, reference photo then every enhancement by MOS. The reference is
                the raw photo (rated in UID2021 and SOTA, unrated in SAUD, absent from SAUD2.0)
  edit sets     uid2021_edits saud_edits sota_edits: raw | best / median / worst MOS enhancement | six cheap edits
  robustness    uid2021_robust: best- and worst-MOS image of a scene under each transform, ranked per transform
  study         our 2AFC pairs: UIEB reference | original | edit, raters' share
  gmad          MOS-judged gMAD pairs (mos/gmad.py): Q-Align vs UCIQE, VisualQuality-R1 vs UIQM, both directions
  UIEB pools    ablate (38 edits), attack (naive / greedy-UCIQE / greedy-UIQM chains), reedit (edits of the UIEB
                reference), banana_enhance / banana_replica (Nano Banana Pro chains); UIEB reference first
Images are research-only, so montages go to uwdata/_figs/montage/<set>/ (gitignored: never pushed). Every number
is read from the score CSVs (mos/scores/<set>/, <pool>/scores*/, mos/results/study_pairs.csv).
Usage: python -I mos/montage.py [set ...]   (default: all sets)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "uwdata/_figs/montage"
FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf")
FONT_B = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono-Bold.ttf")
TW, TH, PAD, LH = 288, 216, 8, 12  # tile image box, gap, text line height
BLUE, GREEN, GREY, INK = (25, 85, 200), (20, 150, 60), (110, 110, 110), (20, 20, 20)
LEGEND = ("score #rank: #1 = this metric's favourite among the images it is ranked with here (lower-better metrics "
          "re-oriented)\nblue = favourite; green frame = people's favourite; grey frame = reference; - = not scored")
POOLS = dict(ablate="out_ablate", attack="out", reedit="out_gt", banana_enhance="out_banana_enhance",
             banana_replica="out_banana_replica")


def ranks(v, lb):
    """1 = best; NaN stays NaN."""
    return pd.Series(v, dtype=float).mul(-1 if not lb else 1).rank(method="min").to_numpy()


def fmt(v):
    if not np.isfinite(v):
        return "     -"
    a = abs(v)
    return f"{v:6.0f}" if a >= 1000 else f"{v:6.1f}" if a >= 100 else f"{v:6.2f}" if a >= 10 else f"{v:6.3f}"


def tile_img(p):
    x = cv2.imread(str(p))
    if x is None:
        return Image.new("RGB", (TW, TH), (230, 230, 230))
    s = min(TW / x.shape[1], TH / x.shape[0])
    x = cv2.resize(x, (max(1, round(x.shape[1] * s)), max(1, round(x.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))


def render(job):
    """job: out path, title, tiles [dict(path, label, sub, s={metric: score}, group, human, ref)], metrics, lb, per_row."""
    out, title, tiles, metrics, lb, per_row = job
    f, fb, ft = ImageFont.truetype(FONT, 9), ImageFont.truetype(FONT_B, 9), ImageFont.truetype(FONT_B, 13)
    rk = {}  # (tile index, metric) -> rank within the tile's group
    for g in {t["group"] for t in tiles}:
        idx = [i for i, t in enumerate(tiles) if t["group"] == g]
        for m in metrics:
            for i, r in zip(idx, ranks([tiles[i]["s"].get(m, np.nan) for i in idx], lb[m])):
                rk[i, m] = r
    nl = -(-len(metrics) // 2)
    cw, ch = TW + PAD, 30 + TH + 6 + nl * LH + PAD
    nc, nr = min(len(tiles), per_row), -(-len(tiles) // per_row)
    W, H0 = max(nc * cw + PAD, 1000), 58
    im = Image.new("RGB", (W, H0 + nr * ch), "white")
    d = ImageDraw.Draw(im)
    d.text((PAD, 6), title, font=ft, fill=INK)
    d.text((PAD, 26), LEGEND, font=f, fill=GREY)
    for i, t in enumerate(tiles):
        x, y = PAD + (i % per_row) * cw, H0 + (i // per_row) * ch
        d.text((x, y), t["label"][:42], font=fb, fill=INK)
        d.text((x, y + 13), t.get("sub", "")[:42], font=f, fill=INK)
        im.paste(tile_img(t["path"]), (x, y + 28))
        if t.get("human") or t.get("ref"):
            d.rectangle([x - 3, y + 25, x + TW + 2, y + 30 + TH], outline=GREEN if t.get("human") else GREY, width=3)
        for j, m in enumerate(metrics):
            v, r = t["s"].get(m, np.nan), rk[i, m]
            txt = f"{m[:14]:<14}{fmt(v)} " + (f"#{r:.0f}" if np.isfinite(r) else "")
            d.text((x + (j // nl) * (TW // 2), y + 30 + TH + 6 + (j % nl) * LH), txt,
                   font=fb if r == 1 else f, fill=BLUE if r == 1 else INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=88)
    return out


def mos_scores(*sets):
    """{metric: {id: score}} over the given score sets (first set wins for a metric's lower_better), lb."""
    sc, lb = {}, {}
    for s in sets:
        for p in sorted((REPO / "mos/scores" / s).glob("*.csv")):
            x = pd.read_csv(p)
            sc.setdefault(p.stem, {}).update(dict(zip(x["id"], x.score)))
            lb.setdefault(p.stem, bool(x.lower_better.iloc[0]))
    return sc, lb


def manifest(s):
    m = pd.read_csv(REPO / f"mos/manifests/{s}.csv")
    m["scene"] = m.scene.astype(str)
    return m


def tile(r, sc, label, sub="", **kw):
    return dict(path=r["path"], label=label, sub=sub, s={m: v.get(r["id"], np.nan) for m, v in sc.items()}, group=0,
                **kw)


def mos_jobs(ds):
    man = manifest(ds)
    has_edits = (REPO / f"mos/manifests/{ds}_edits.csv").exists()
    sc, lb = mos_scores(ds, *([f"{ds}_edits"] if has_edits else []))
    raw = manifest(f"{ds}_edits").query("method == 'raw'").set_index("scene") if has_edits else None
    jobs = []
    for s, g in man.groupby("scene"):
        g = g.sort_values("mos", ascending=False)
        tiles, top = [], g.mos.max()
        o = g[g.method == "Original"]
        if len(o):  # rated raw photo
            tiles.append(tile(o.iloc[0], sc, "REFERENCE: raw photo", f"MOS {o.mos.iloc[0]:.2f}", ref=True,
                              human=o.mos.iloc[0] == top))
        elif raw is not None and s in raw.index:
            tiles.append(tile(raw.loc[s], sc, "REFERENCE: raw photo", "unrated", ref=True))
        e = g[g.method != "Original"]
        for k, (_, r) in enumerate(e.iterrows(), 1):
            tiles.append(tile(r, sc, r.method, f"MOS {r.mos:.2f}  (#{k} of {len(e)} by people)", human=r.mos == top))
        note = "" if tiles[0].get("ref") else "  (no raw photo in this dataset)"
        jobs.append((OUT / ds / f"{s}.jpg", f"{ds} | scene {s} | every enhancement, by MOS{note}", tiles, sorted(sc),
                     lb, 8))
    return jobs


def edit_jobs(es):
    ds = es.removesuffix("_edits")
    man, em = manifest(ds), manifest(es)
    sc, lb = mos_scores(ds, es)
    jobs = []
    for s, e in em.groupby("scene"):
        g = man[(man.scene == s) & (man.method != "Original")].sort_values("mos", ascending=False)
        tiles = [tile(e[e.method == "raw"].iloc[0], sc, "REFERENCE: raw photo", "input to every edit", ref=True)]
        for name, r in (("best", g.iloc[0]), ("median", g.iloc[len(g) // 2]), ("worst", g.iloc[-1])):
            tiles.append(tile(r, sc, f"{name} MOS: {r.method}", f"MOS {r.mos:.2f}", human=name == "best"))
        for _, r in e[e.method != "raw"].iterrows():
            tiles.append(tile(r, sc, r.method.split("@")[0], "cheap edit of the raw photo (no MOS)"))
        jobs.append((OUT / es / f"{s}.jpg", f"{ds} | scene {s} | people's best/median/worst vs cheap edits", tiles,
                     sorted(sc), lb, 5))
    return jobs


def robust_jobs(rs="uid2021_robust"):
    man, (sc, lb) = manifest(rs), mos_scores(rs)
    order = ["orig", "hflip", "rot90", "half", "crop90"]
    jobs = []
    for s, g in man.groupby("scene"):
        o = g[g.t == "orig"].sort_values("mos")
        tiles = []
        for name, src in (("best", o.src.iloc[-1]), ("worst", o.src.iloc[0])):
            for t in order:
                r = g[(g.src == src) & (g.t == t)].iloc[0]
                tiles.append(tile(r, sc, f"{t}: {name} MOS ({r.method})", f"MOS {r.mos:.2f}", human=name == "best")
                             | {"group": t})
        jobs.append((OUT / rs / f"{s}.jpg", f"UID2021 | scene {s} | best vs worst MOS image under each transform "
                     "(#rank within each transform column: a metric that flips #1/#2 across columns changed its mind)",
                     tiles, sorted(sc), lb, len(order)))
    return jobs


def pool_scores(pool):
    """Wide table (mode, image, step) x metric over <pool>/scores*/ (first file wins per metric), lb."""
    parts, lb = [], {}
    for p in sorted((REPO / pool).glob("scores*/*.csv")):
        x = pd.read_csv(p, dtype={"image": str})
        m = x.metric.iloc[0]
        if m not in lb:
            lb[m] = bool(x.lower_better.iloc[0])
            parts.append(x)
    w = pd.concat(parts).pivot_table(index=["mode", "image", "step"], columns="metric", values="score")
    return w, lb


def step_path(pool, mode, img, step):
    p = REPO / pool / mode / img / f"{step:02d}.png"
    return p if p.exists() else REPO / pool / mode / img / f"{step}.png"


def pool_jobs(name):
    pool = POOLS[name]
    w, lb = pool_scores(pool)
    gw, glb = pool_scores("out_gt")
    gt = gw.xs(("clean", 0), level=("mode", "step"))
    metrics = sorted(set(lb) | set(glb))
    lb = glb | lb
    nsteps = w.reset_index().groupby("mode").step.nunique()
    jobs = []
    for img, g in w.groupby(level="image"):
        ref = dict(path=REPO / "out_gt/clean" / img / "00.png", label="REFERENCE: UIEB reference image",
                   sub="(reference-890)", s=gt.loc[img].dropna().to_dict() if img in gt.index else {}, group=0,
                   ref=True)
        single = [k for k in g.index if nsteps[k[0]] == 1]
        chains = sorted({k[0] for k in g.index if nsteps[k[0]] > 1})
        groups = ([("edits", single)] if single else []) + [(c, sorted(k for k in g.index if k[0] == c)) for c in chains]
        for gname, keys in groups:
            keys = sorted(keys, key=lambda k: (k[0] != "clean", k[0], k[2]))
            tiles = [] if pool == "out_gt" and gname == "edits" else [ref]
            for mode, _, step in keys:
                s = g.loc[(mode, img, step)].dropna().to_dict()
                label = ("UIEB raw photo" if mode == "clean" and pool != "out_gt" else
                         "REFERENCE: UIEB reference image" if mode == "clean" else
                         f"{mode} step {step}" + (" (input)" if step == 0 else "") if gname != "edits" else mode)
                tiles.append(dict(path=step_path(pool, mode, img, step), label=label, s=s, group=0,
                                  ref=mode == "clean" and pool == "out_gt"))
            jobs.append((OUT / name / f"{img}_{gname}.jpg", f"UIEB image {img} | {name}: {gname}", tiles, metrics,
                         lb, 8))
    return jobs


def study_jobs():
    pairs = pd.read_csv(REPO / "mos/results/study_pairs.csv", dtype={"image": str})
    lb = pairs.groupby("metric").lower_better.first().astype(bool).to_dict()
    gw, _ = pool_scores("out_gt")
    gt = gw.xs(("clean", 0), level=("mode", "step"))
    sys.path.insert(0, str(REPO / "study"))
    from analyze_votes import CLEAN, SRC
    jobs = []
    for (c, im), x in pairs.groupby(["condition", "image"]):
        x = x.set_index("metric")
        d, mode, step = SRC[c]
        cmode, cstep = CLEAN[d]
        h, n = x.human_pref_edit.iloc[0], x.n_votes.iloc[0]
        tiles = [dict(path=REPO / "out_gt/clean" / im / "00.png", label="REFERENCE: UIEB reference image",
                      sub="not shown to raters", s=gt.loc[im].dropna().to_dict() if im in gt.index else {}, group=0,
                      ref=True),
                 dict(path=step_path(d, cmode, im, cstep), label="original (UIEB raw)",
                      sub=f"raters {1 - h:.0%} of {n} votes", s=x.original.to_dict(), group=0, human=h < 0.5),
                 dict(path=step_path(d, mode, im, step), label=f"edit: {c}", sub=f"raters {h:.0%} of {n} votes",
                      s=x.edited.to_dict(), group=0, human=h > 0.5)]
        jobs.append((OUT / "study" / f"{c}_{im}.jpg", f"Our 2AFC study | {c} | UIEB image {im}", tiles, sorted(lb),
                     lb, 3))
    return jobs


GMAD = [("qalign_8bit", "uciqe"), ("uciqe", "qalign_8bit"), ("visualquality_r1", "uiqm"), ("uiqm", "visualquality_r1")]


def gmad_jobs(_):
    """gMAD pairs (mos/gmad.py): per MOS set and (attacker, defender), the attacker's 6 most separated pairs among
    pairs the defender scores the same. Each row is one pair; only the two metrics are printed."""
    sys.path.insert(0, str(REPO / "mos"))
    from gmad import load, picks
    jobs = []
    for ds in ("uid2021", "saud", "saud2", "sota"):
        y, diff, man, i, j = load(ds)
        sc, lb = mos_scores(ds)
        man["scene"] = man.scene.astype(str)
        for a, d in GMAD:
            top = picks(diff[a], diff[d], y)[:6]
            tiles = []
            for n, p in enumerate(top):
                for q in (i[p], j[p]):
                    r = man.iloc[q]
                    tiles.append(tile(r, sc, f"scene {r.scene}: {r.method}", f"MOS {r.mos:.2f}",
                                      human=r.mos == max(man.mos.iloc[i[p]], man.mos.iloc[j[p]])) | {"group": n})
            hit = (np.sign(diff[a][top]) == y[top]).mean()
            jobs.append((OUT / "gmad" / f"{ds}_{a}_vs_{d}.jpg",
                         f"{ds} gMAD | {d} rates each pair ~equal, {a} splits them most | people side with {a} in "
                         f"{hit:.0%}", tiles, [a, d], lb, 2))
    return jobs


BUILD = dict(uid2021=mos_jobs, saud=mos_jobs, saud2=mos_jobs, sota=mos_jobs, uid2021_edits=edit_jobs,
             saud_edits=edit_jobs, sota_edits=edit_jobs, uid2021_robust=lambda _: robust_jobs(),
             study=lambda _: study_jobs(), gmad=gmad_jobs, **{k: pool_jobs for k in POOLS})

if __name__ == "__main__":
    # self-check: rank 1 = best for both orientations, NaN unranked
    assert list(ranks([3, 1, 2], False)) == [1, 3, 2] and list(ranks([3, 1, 2], True)) == [3, 1, 2]
    assert np.isnan(ranks([1, np.nan], False)[1])
    for name in sys.argv[1:] or BUILD:
        jobs = BUILD[name](name)
        with Pool(6) as p:
            for k, _ in enumerate(p.imap_unordered(render, jobs, chunksize=4), 1):
                if k % 100 == 0:
                    print(name, k, "/", len(jobs), flush=True)
        print("done", name, len(jobs), "montages ->", OUT / name, flush=True)

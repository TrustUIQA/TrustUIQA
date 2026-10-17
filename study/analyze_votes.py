"""Analyse rating-study sessions: quality checks, human preference per condition, metric-human agreement.

Each argument is a response code (base64 text file) or a JSON session file exported from the db.
Usage (from repo root): python study/analyze_votes.py code1.txt [code2.txt ...]
"""
import base64
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# study condition -> (score dir, mode, step); clean reference is step 0 of the same family
SRC = {c: ("out_ablate", m, 1) for c, m in {
    "sharpen_mild": "sharpen@mild", "clahe_mild": "clahe@mild", "sat_mild": "saturation@mild",
    "sat_mod": "saturation@moderate", "contrast_mild": "contrast@mild", "noise_mild": "noise@mild",
    "noise_strong": "noise@strong", "posterize_strong": "posterize@strong", "hue_mod": "hue@moderate",
    "red_strong": "red_boost@strong"}.items()}
SRC.update(chain_5=("out", "naive", 5), chain_20=("out", "naive", 20))
CLEAN = {"out_ablate": ("clean", 0), "out": ("naive", 0)}


def load(path):
    """A JSON session, or one or more whitespace-separated response codes (bad/truncated codes are skipped)."""
    txt = Path(path).read_text().strip()
    if txt.startswith("{"):
        return [json.loads(txt)]
    out = []
    for code in txt.split():
        try:
            out.append(json.loads(base64.b64decode(code)))
        except ValueError:
            print(f"skip unreadable code in {path}: {code[:20]}...")
    return out


def picked_clean(v):
    return (v["pick"] == "L") == v["cl"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa(a, b, w=None):
    """Cohen's kappa for two binary label vectors; optional per-item weights (cluster bootstrap multiplicities)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    if not w.sum():
        return float("nan")
    m = lambda x: np.average(x, weights=w)
    pe = m(a) * m(b) + (1 - m(a)) * (1 - m(b))
    return (m(a == b) - pe) / (1 - pe) if pe < 1 else float("nan")


def cluster_ci(a, h, rater, image, n_boot=1000, seed=0):
    """95% CI for kappa(a, h) resampling raters and images together (two-way cluster bootstrap):
    trials are not independent, they share a rater and an image."""
    rng = np.random.default_rng(seed)
    ru, ri = np.unique(rater, return_inverse=True)
    gu, gi = np.unique(image, return_inverse=True)
    boot = []
    for _ in range(n_boot):
        wr = np.bincount(rng.integers(0, len(ru), len(ru)), minlength=len(ru))
        wg = np.bincount(rng.integers(0, len(gu), len(gu)), minlength=len(gu))
        boot.append(kappa(a, h, wr[ri] * wg[gi]))
    return np.nanpercentile(boot, [2.5, 97.5])


def kripp_alpha(units):
    """Krippendorff's alpha, nominal, binary labels; units = per-item label lists (missing raters allowed)."""
    units = [u for u in units if len(u) >= 2]
    n, ones = sum(map(len, units)), sum(map(sum, units))
    do = sum(2 * sum(u) * (len(u) - sum(u)) / (len(u) - 1) for u in units) / n
    de = 2 * ones * (n - ones) / (n * (n - 1))
    return 1 - do / de if de else float("nan")


def checks(s):
    vs = s["votes"]
    gold = [picked_clean(v) for v in vs if v["k"] == "g"]
    ident = [v["pick"] for v in vs if v["k"] == "i"]
    first = {(v["img"], v["c"]): picked_clean(v) for v in vs if v["k"] == "t"}
    rep = [picked_clean(v) == first[(v["img"], v["c"])] for v in vs if v["k"] == "r"]
    ms = pd.Series([v["ms"] for v in vs])
    left = sum(v["pick"] == "L" for v in vs) / len(vs)
    return {"run": s["run"][:8], "block": s["block"] + 1,  # stored 0-based; +1 matches the #b1-#b4 link
"screen": s["meta"]["screen"], "n": len(vs), "gold": f"{sum(gold)}/{len(gold)}",
            "repeat_consistent": f"{sum(rep)}/{len(rep)}", "identical_L": f"{ident.count('L')}/{len(ident)}",
            "left_rate": round(left, 2), "median_ms": int(ms.median()), "fast<500ms": int((ms < 500).sum()),
            "pass": all(gold)}


def scores():
    out = {}
    for d in ("out_ablate", "out"):
        for f in Path(d, "scores").glob("*.csv"):
            df = pd.read_csv(f, dtype={"image": str})
            for (mode, img, step), r in df.set_index(["mode", "image", "step"]).iterrows():
                out[(d, f.stem, mode, img, step)] = (r.score, r.lower_better)
    # pilot run kept UIQM/UCIQE in attack.csv, not out/scores
    att = pd.read_csv("out/attack.csv", dtype={"image": str})
    for r in att.itertuples():
        for m in ("uiqm", "uciqe"):
            out[("out", m, r.mode, r.image, r.step)] = (getattr(r, m), False)
    return out


def main(paths):
    sessions = list({s["run"]: s for p in paths for s in load(p)}.values())  # same code pasted twice counts once
    qc = pd.DataFrame([checks(s) for s in sessions])
    print("== quality checks ==\n" + qc.to_string(index=False))
    kept = [s for s, ok in zip(sessions, qc["pass"]) if ok]
    trials = [v for s in kept for v in s["votes"] if v["k"] == "t"]
    if not trials:
        return print("no session passed the gold check")

    hum = defaultdict(list)
    for v in trials:
        hum[v["c"]].append(not picked_clean(v))
    # Thurstone Case V vs clean: z = Phi^-1(p), 0 = same as clean; (k+.5)/(n+1) keeps 0/n finite
    rows = {}
    for c, x in hum.items():
        k, n = sum(x), len(x)
        lo, hi = wilson(k, n)
        rows[c] = {"k/n": f"{k}/{n}", "pref_edit": f"{k / n:.0%}", "95% CI": f"{lo:.0%}-{hi:.0%}",
                   "thurstone_z": round(norm.ppf((k + .5) / (n + 1)), 2)}
    print("\n== human: preference for edited image (pairwise 'MOS') ==")
    ht = pd.DataFrame(rows).T.sort_values("thurstone_z")
    print(ht.to_string())
    ht.to_latex("study/table_human.tex")

    # inter-rater: Cohen's kappa per rater pair on shared items, Krippendorff's alpha over everyone
    lab = [{(v["img"], v["c"]): not picked_clean(v) for v in s["votes"] if v["k"] == "t"} for s in kept]
    print("\n== inter-rater agreement ==")
    for (i, a), (j, b) in itertools.combinations(enumerate(lab), 2):
        shared = sorted(a.keys() & b.keys())
        if len(shared) >= 20:
            print(f"Cohen kappa {kept[i]['run'][:8]} vs {kept[j]['run'][:8]} (block {kept[i]['block'] + 1}, "
                  f"{len(shared)} pairs): {kappa([a[x] for x in shared], [b[x] for x in shared]):.2f}, "
                  f"raw agreement {np.mean([a[x] == b[x] for x in shared]):.0%}")
    items = defaultdict(list)
    for d in lab:
        for x, y in d.items():
            items[x].append(y)
    print(f"Krippendorff alpha (nominal, {sum(len(u) >= 2 for u in items.values())} multiply-rated pairs): "
          f"{kripp_alpha(list(items.values())):.2f}")

    sc = scores()
    metrics = sorted({k[1] for k in sc})
    # per metric: columns metric_edit, human_edit, rater run, image id
    agree, fooled = defaultdict(lambda: ([], [], [], [])), defaultdict(lambda: defaultdict(list))
    for s in kept:
        for v in s["votes"]:
            if v["k"] != "t":
                continue
            d, mode, step = SRC[v["c"]]
            cmode, cstep = CLEAN[d]
            for m in metrics:
                a, b = sc.get((d, m, mode, v["img"], step)), sc.get((d, m, cmode, v["img"], cstep))
                if a is None or b is None or a[0] == b[0]:
                    continue
                metric_edit = (a[0] < b[0]) if a[1] else (a[0] > b[0])
                for col, x in zip(agree[m], (metric_edit, not picked_clean(v), s["run"][:8], v["img"])):
                    col.append(x)
                fooled[m][v["c"]].append(metric_edit and picked_clean(v))
    # sensitivity subsets (point estimates): drop low-quality raters, drop phone raters
    q = qc[qc["pass"]].set_index("run")
    lowq = {r for r, x in q.iterrows() if int(x.repeat_consistent.split("/")[0]) < 3 or x.left_rate > 0.6}
    phone = {r for r, x in q.iterrows() if x.screen == "Phone"}
    rows = {}
    for m, cols in agree.items():
        a, h, run, img = map(np.array, cols)
        lo, hi = cluster_ci(a, h, run, img)
        keep = lambda drop: ~np.isin(run, list(drop))
        rows[m] = {"agree": f"{(a == h).mean():.0%}", "kappa": round(kappa(a, h), 2), "95% CI": f"{lo:.2f} to {hi:.2f}",
                   "no_lowq": round(kappa(a[keep(lowq)], h[keep(lowq)]), 2),
                   "no_phone": round(kappa(a[keep(phone)], h[keep(phone)]), 2), "n": len(a)}
    print("\n== metric vs human: Cohen's kappa on test pairs (0 = chance, <0 = prefers what humans reject) ==")
    print(f"CI: two-way cluster bootstrap over raters and images. no_lowq drops {len(lowq)} raters "
          f"(repeats <3/5 or left rate >0.6): {sorted(lowq)}; no_phone drops {len(phone)} phone raters")
    kt = pd.DataFrame(rows).T.sort_values("kappa", ascending=False)
    print(kt.to_string())
    kt.to_latex("study/table_kappa.tex")
    print("\n== truly fooled: metric prefers edit AND human preferred clean (share of pairs) ==")
    tab = pd.DataFrame({m: {c: sum(x) / len(x) for c, x in cs.items()} for m, cs in fooled.items()})
    ft = (tab.T.sort_index() * 100).round(0).astype("Int64")
    print(ft.to_string())
    ft.to_latex("study/table_fooled.tex")


if __name__ == "__main__":
    main(sys.argv[1:])

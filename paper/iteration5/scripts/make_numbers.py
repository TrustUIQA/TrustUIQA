"""Every number and table of iteration 5, recomputed from the raw result files (guide rule A1: no hand-typed numbers).

Writes
  tables/numbers.tex   one \\newcommand per number used in the text (macro names start with \\n)
  tables/*.tex         every table of the paper and the supplement
  notes/numbers.json   the same numbers, for notes/claim_source_map.md and scripts/check.py
Run from the repo root: python -I paper/iteration5/scripts/make_numbers.py
"""
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403

T = IT / "tables"
N = {}  # macro name -> (text value, source file)


def put(key, val, src):
    assert re.fullmatch(r"[A-Za-z]+", key), key
    assert key not in N, key
    N[key] = (str(val), src)


def f2(x):
    return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"


def m2(x):  # inside tables (math minus)
    return f"$-${abs(x):.2f}" if round(x, 2) < 0 else f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def big(n):
    return f"{n:,}"


def save(name, body):
    (T / f"{name}.tex").write_text(body)


def mark(vals, fmt, higher=True):
    """bold best, underline second best (guide A8.2)."""
    s = pd.Series(vals).dropna()
    order = s.sort_values(ascending=not higher).index.tolist()
    out = {}
    for k, v in vals.items():
        t = fmt(v) if pd.notna(v) else "n/a"
        if order and k == order[0]:
            t = f"\\textbf{{{t}}}"
        elif len(order) > 1 and k == order[1]:
            t = f"\\underline{{{t}}}"
        out[k] = t
    return out


# ------------------------------------------------------------------ data and references
def references():
    rows = []
    tot_img = tot_sc = 0
    for s in SETS:
        m = pd.read_csv(REPO / f"mos/manifests/{s}.csv")
        rated = m[m.mos.notna()]
        tot_img += len(rated)
        tot_sc += m.scene.nunique()
        meth = rated.method.nunique()
        put(f"nImg{s.capitalize().replace('2021', 'A').replace('2', 'B')}", big(len(rated)), f"mos/manifests/{s}.csv")
        rows.append((cref(s), big(m.scene.nunique()), str(meth), big(len(rated))))
    put("nImgMOS", big(tot_img), "mos/manifests/*.csv")
    put("nScenesMOS", big(tot_sc), "mos/manifests/*.csv")
    sp = pd.read_csv(R / "study_pairs.csv", dtype={"image": str})
    one = sp[sp.metric == "uciqe"]
    votes = int(one.n_votes.sum())
    av = (IT / "notes/study_analyze_votes.txt").read_text()
    qc = re.findall(r"^\S+\s+\d\s+.*\s(True|False)$", av.split("== human")[0], re.M)
    put("nRaters", sum(q == "True" for q in qc), "notes/study_analyze_votes.txt (quality checks)")
    put("nRatersAll", len(qc), "notes/study_analyze_votes.txt (quality checks)")
    ir = [float(x) for x in re.findall(r"^Cohen kappa .*\): (-?[\d.]+),", av, re.M)]
    put("interRaterMin", f"{min(ir):.2f}", "notes/study_analyze_votes.txt (rater pairs on the same block)")
    put("interRaterMax", f"{max(ir):.2f}", "same")
    put("nRaterPairs", len(ir), "same")
    put("interRaterMed", f"{np.median(ir):.2f}", "same")
    q = re.findall(r"^\w{8}\s+(\d)\s.*?\s(\d)/5\s+(\d)/4\s", av.split("== human")[0], re.M)
    assert len(q) == len(qc), (len(q), len(qc))
    blk = [sum(b == str(k) for b, _r, _i in q) for k in range(1, 5)]
    assert sum(n * (n - 1) // 2 for n in blk) == len(ir)
    put("blockSizes", ", ".join(map(str, blk[:-1])) + f" and {blk[-1]}", "notes/study_analyze_votes.txt (quality checks, block column)")
    put("nBlockBig", max(blk), "same (largest block)")
    put("nBlockSmall", min(blk), "same (smallest block)")
    put("nRaterPairsBig", max(blk) * (max(blk) - 1) // 2, "same (rater pairs within the largest block)")
    put("repeatSame", pct(sum(int(r) for _b, r, _i in q) / (5 * len(q))), "same (repeat_consistent)")
    put("identLeft", pct(sum(int(i) for _b, _r, i in q) / (4 * len(q))), "same (identical_L)")
    ff = IT / "notes/figure_facts.json"  # written by figures/scripts/figures.py
    if ff.exists():
        t = json.loads(ff.read_text())
        put("gridN", t["gridN"], "notes/figure_facts.json (figures.py img_grid)")
        put("gridCmp", t["gridCmp"], "same")
        put("gridWins", ", ".join(map(str, t["gridWins"])), "same")
        put("gridBeatAll", pct(t["gridBeatAll"]), "same")
        put("gridBeatEdit", pct(t["gridPerPanel"]["edit"]), "same")
        put("gridBeatProRep", pct(t["gridPerPanel"]["prorep"]), "same")
        put("driftCropId", t["driftCropId"], "notes/figure_facts.json (figures.py img_drift)")
        put("driftFrame", "{}\\times{}".format(*t["driftFrame"]), "same")
        put("driftInW", t["driftInW"], "same")
    put("nChoices", big(votes), "mos/results/study_pairs.csv")
    put("nConditions", one.condition.nunique(), "mos/results/study_pairs.csv")
    put("nStudyImg", one.image.nunique(), "mos/results/study_pairs.csv")
    per = one.groupby("condition").n_votes.sum()
    assert per.nunique() == 1
    put("nVotesPerCond", int(per.iloc[0]), "mos/results/study_pairs.csv")
    rows.append(("Our study", str(one.image.nunique()), f"{one.condition.nunique()} edits", big(votes) + " choices"))
    body = "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nReference & Scenes & Methods & Images \\\\\n\\midrule\n"
    body += "".join(" & ".join(r) + " \\\\\n" for r in rows[:-1]) + "\\midrule\n" + " & ".join(rows[-1]) + " \\\\\n"
    save("tab_refs", body + "\\bottomrule\n\\end{tabular}\n")


def metrics():
    w = kappa_wide()
    put("nMetrics", len(w), "mos/results/<ds>.csv + extra_<ds>.csv")
    put("nAllFive", int(w.notna().all(axis=1).sum()), "same + study_kappa_all.csv")
    g = pd.Series({m: group(m) for m in w.index}).value_counts()
    put("nHandUW", g["hand-crafted underwater"], "scripts/common.py groups")
    put("nLearnedUW", g["learned underwater"], "scripts/common.py groups")
    put("nVLM", g["vision-language"], "scripts/common.py groups")
    put("nGeneral", g["general-purpose"], "scripts/common.py groups")
    miss = {r: w.index[w[r].isna()].tolist() for r in REFS}
    put("nSota", int(w["sota"].notna().sum()), "mos/results/sota.csv")
    assert miss["sota"] == ["liqe_mix"] and miss["study"] == ["liqe_mix"], miss
    return w


# ------------------------------------------------------------------ RQ1 agreement
def agreement(w):
    k = kappa_table()
    a5 = w.dropna()
    worst = a5.min(axis=1).sort_values(ascending=False)
    put("kWorstBest", f2(worst.iloc[0]), "kappa tables")
    put("kWorstBestName", name(worst.index[0]), "kappa tables")
    put("kWorstSecond", f2(worst.iloc[1]), "kappa tables")
    put("kWorstSecondName", name(worst.index[1]), "kappa tables")
    st = pd.read_csv(R / "study_kappa_all.csv").set_index("metric").reindex(w.index).dropna(subset=["n"])
    N = st.n.max()
    acc = (st.agree * st.n + 0.5 * (N - st.n)) / N  # a metric tie counts as half a correct choice
    best = w["study"].idxmax()
    assert best == "visualquality_r1" and acc.idxmax() != best
    put("vqStudyTie", pct(1 - st.n[best] / N), "mos/results/study_kappa_all.csv (pairs without a metric preference)")
    put("accTieBestName", name(acc.idxmax()), "same, accuracy with ties as half")
    put("accTieBest", f2(acc.max()), "same")
    put("vqAccTie", f2(acc[best]), "same")
    put("vqAccTieRank", int((acc > acc[best]).sum()) + 1, "same, rank")
    ko = pd.read_csv(R / "kappa_order.csv").pivot(index="metric", columns="set", values="k_sym")
    ws = a5.copy()
    for s in SETS:
        ws[s] = ko[s].reindex(ws.index)
    sw = ws.min(axis=1).sort_values(ascending=False)
    put("kSymWorstBest", f2(sw.iloc[0]), "mos/results/kappa_order.csv k_sym on the opinion-score sets, study kappa")
    put("kSymWorstBestName", name(sw.index[0]), "same")
    put("kWorstCap", f"{np.ceil(worst.iloc[0] * 100) / 100:.2f}", "kappa tables (rounded up)")
    for r in REFS:
        put(f"kMed{r.capitalize().replace('2021', 'A').replace('2', 'B')}", f2(w[r].median()), f"kappa table {r}")
        put(f"kBest{r.capitalize().replace('2021', 'A').replace('2', 'B')}", f2(w[r].max()), f"kappa table {r}")
        put(f"kBestName{r.capitalize().replace('2021', 'A').replace('2', 'B')}", name(w[r].idxmax()), f"kappa table {r}")
    # UCIQE and UIQM
    for m in ["uciqe", "uiqm"]:
        for r in REFS:
            put(f"k{m.capitalize()}{r.capitalize().replace('2021', 'A').replace('2', 'B')}", f2(w.loc[m, r]), f"kappa table {r}")
    hi = pd.Series({s: k[s].loc["uciqe", "kappa_hi"] for s in SETS})
    put("nUciqeBelowChance", int((hi < 0).sum()), "kappa_hi < 0 on MOS sets")
    lo = pd.Series({s: k[s].loc["uciqe", "kappa_lo"] for s in SETS})
    put("nUciqeAtOrBelow", int((lo <= 0).sum()), "kappa_lo <= 0 on MOS sets")
    # share of metrics whose 95% CI includes or lies below zero on SOTA
    so = k["sota"].loc[[m for m in w.index if m in k["sota"].index]]
    put("nSotaLowHalf", int((so.kappa < 0.2).sum()), "sota.csv kappa < 0.2")
    # method ranking
    mr = pd.read_csv(R / "method_rank.csv")
    mr = mr[~mr.metric.isin(DROP | {UCIQE_M})]
    for s in SETS:
        d = mr[mr.ds == s]
        tag = s.capitalize().replace("2021", "A").replace("2", "B")
        put(f"nTopHit{tag}", int(d.top_hit.sum()), "mos/results/method_rank.csv")
        put(f"nRank{tag}", len(d), "mos/results/method_rank.csv")
        put(f"tauUciqe{tag}", f2(d.set_index("metric").loc["uciqe", "ktau"]), "mos/results/method_rank.csv")
        put(f"tauUiqm{tag}", f2(d.set_index("metric").loc["uiqm", "ktau"]), "mos/results/method_rank.csv")
        put(f"nMethods{tag}", int(d.n_methods.iloc[0]), "mos/results/method_rank.csv")
    neg = int((mr[mr.metric == "uciqe"].ktau < 0).sum())
    put("nUciqeTauNeg", neg, "mos/results/method_rank.csv")
    # gMAD aggressiveness without the removed scores (row mean of the saved win matrices)
    rows = []
    for s in SETS:
        g = pd.read_csv(R / f"gmad_{s}.csv", index_col=0)
        keep = [m for m in g.index if m not in DROP and m != UCIQE_M]
        g = g.loc[keep, keep]
        agg = g.mean(axis=1, skipna=True).sort_values(ascending=False)
        rows.append((s, agg))
        put(f"gmadUciqeRank{s.capitalize().replace('2021', 'A').replace('2', 'B')}", int(list(agg.index).index("uciqe")) + 1, f"gmad_{s}.csv")
        put(f"gmadN{s.capitalize().replace('2021', 'A').replace('2', 'B')}", len(agg), f"gmad_{s}.csv")
    body = "\\begin{tabular}{@{}l" + "l" * len(SETS) + "@{}}\n\\toprule\nRank & " + " & ".join(cref(s) for s in SETS) + " \\\\\n\\midrule\n"
    for i in range(5):
        body += f"{i + 1} & " + " & ".join(f"{mcite(a.index[i])} ({a.iloc[i]:.2f})" for _, a in rows) + " \\\\\n"
    body += "\\midrule\nUCIQE & " + " & ".join(f"{list(a.index).index('uciqe') + 1} of {len(a)} ({a['uciqe']:.2f})" for _, a in rows) + " \\\\\n"
    save("tab_gmad", body + "\\bottomrule\n\\end{tabular}\n")
    # 2AFC on our study
    t = pd.read_csv(R / "twoafc.csv")
    t = t[t.metric.isin(w.index)].sort_values("score", ascending=False)
    put("twoafcBest", f"{t.score.iloc[0]:.2f}", "mos/results/twoafc.csv")
    put("twoafcBestName", name(t.metric.iloc[0]), "mos/results/twoafc.csv")
    put("twoafcCeiling", f"{t.ceiling.iloc[0]:.2f}", "mos/results/twoafc.csv")
    put("twoafcUciqe", f"{t.set_index('metric').loc['uciqe', 'score']:.2f}", "mos/results/twoafc.csv")
    put("twoafcN", int(t.n.iloc[0]), "mos/results/twoafc.csv")
    put("twoafcAtCeiling", int((t.score >= t.ceiling).sum()), "mos/results/twoafc.csv")
    put("twoafcMedian", f"{t.score.median():.2f}", "mos/results/twoafc.csv")
    # full kappa table (supplement)
    lo_, hi_ = kappa_wide("kappa_lo"), kappa_wide("kappa_hi")
    order = sorted(w.index, key=lambda m: (["hand-crafted underwater", "learned underwater", "vision-language", "general-purpose"].index(group(m)), name(m).lower()))
    body = "\\begin{longtable}{@{}l" + "c" * 5 + "@{}}\n\\caption{Within-scene Cohen's $\\kappa$ of every metric with people, with 95\\% scene-bootstrap intervals (study: rater and image cluster bootstrap). n/a: not run. Groups as in Sec.~3.2 of the main paper.}\\label{tab:kappa_all}\\\\\n\\toprule\nMetric & " + " & ".join(cref(r) for r in REFS) + " \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nMetric & " + " & ".join(cref(r) for r in REFS) + " \\\\\n\\midrule\n\\endhead\n"
    cur = None
    for m in order:
        if group(m) != cur:
            cur = group(m)
            body += f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{cur[0].upper() + cur[1:]}}}}} \\\\\n"
        cells = [("n/a" if pd.isna(w.loc[m, r]) else f"{m2(w.loc[m, r])} {{\\scriptsize [{m2(lo_.loc[m, r])}, {m2(hi_.loc[m, r])}]}}") for r in REFS]
        body += mcite(m) + " & " + " & ".join(cells) + " \\\\\n"
    save("tab_kappa_all", body + "\\bottomrule\n\\end{longtable}\n")
    # UIEB-style "which metric is best per reference" list for the figure caption
    return worst


# ------------------------------------------------------------------ RQ2 learned underwater metrics
def learned(w):
    ins, lab, other = [], [], []
    for m in LEARNED_UW:
        if m not in TRAINED:
            continue
        tr = TRAINED[m]
        for r in REFS:
            v = w.loc[m, r]
            (ins if r == tr else lab if GROUP_OF[r] == GROUP_OF[tr] else other).append(v)
    put("kInMin", f2(min(ins)), "kappa tables, metrics with known training set")
    put("kInMax", f2(max(ins)), "kappa tables")
    put("kLabMin", f2(min(lab)), "kappa tables (SAUD2.0, same lab as SAUD)")
    put("kLabMax", f2(max(lab)), "kappa tables")
    put("kOtherMax", f2(max(other)), "kappa tables")
    put("nKnownTrain", len(TRAINED), "mos/cross_eval.py TRAINED")
    o = pd.read_csv(R / "overlap.csv")
    put("nShareSaudSaudB", int(((o.set_a == "saud") & (o.set_b == "saud2")).sum()), "mos/results/overlap.csv")
    cols = REFS
    body = "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\nMetric (trained on) & " + " & ".join(cref(r).replace("Our study", "Study") for r in cols) + " \\\\\n\\midrule\n"
    allm = LEARNED_UW + ["qalign_8bit", "visualquality_r1", "rali"]
    seen = lambda m, r: m in TRAINED and GROUP_OF[r] == GROUP_OF[TRAINED[m]]  # noqa: E731  shaded cells are not ranked
    marks = {r: mark({m: (np.nan if seen(m, r) else w.loc[m, r]) for m in allm}, m2) for r in cols}
    marks = {r: {m: (m2(w.loc[m, r]) if seen(m, r) else t) for m, t in marks[r].items()} for r in cols}
    for m in allm:
        if m == "qalign_8bit":
            body += "\\midrule\n"
        tr = f" ({REF_NAME[TRAINED[m]]})" if m in TRAINED else ""
        cells = []
        for r in cols:
            c = marks[r][m]
            if m in TRAINED and r == TRAINED[m]:
                c = f"\\cellcolor{{insample}}{c}"
            elif m in TRAINED and GROUP_OF[r] == GROUP_OF[TRAINED[m]]:
                c = f"\\cellcolor{{samelab}}{c}"
            cells.append(c)
        body += mcite(m) + tr + " & " + " & ".join(cells) + " \\\\\n"
    save("tab_learned", body + "\\bottomrule\n\\end{tabular}\n")
    # disjoint check
    d = pd.read_csv(R / "disjoint.csv")
    p = d.pivot_table(index=["metric", "trained_on", "test_set"], columns="subset", values="kappa")
    put("kDisjointMax", f2(p["disjoint"].max()), "mos/results/disjoint.csv")
    put("kDisjointDropMax", f"{(p['all'] - p['disjoint']).max():.2f}", "mos/results/disjoint.csv")
    sub = p.xs("saud2", level="test_set")
    put("kDisjointSaudBMin", f2(sub["disjoint"].min()), "mos/results/disjoint.csv")
    put("kDisjointSaudBMax", f2(sub["disjoint"].max()), "mos/results/disjoint.csv")
    # fusion (refit by scripts/fusion.py without the removed scores)
    f = pd.read_csv(T / "fusion.csv")
    tr = f[f.train == "saud2"].set_index("test")
    for s in ["saud", "sota", "uid2021"]:
        tag = s.capitalize().replace("2021", "A")
        put(f"fusSaudBto{tag}", f2(tr.loc[s, "kappa"]), "tables/fusion.csv")
    off = f[~((f.train.map(GROUP_OF)) == (f.test.map(GROUP_OF)))]
    put("fusOtherMax", f2(off.kappa.max()), "tables/fusion.csv")
    put("fusOtherMaxPair", f"{REF_NAME[off.loc[off.kappa.idxmax(), 'train']]} to {REF_NAME[off.loc[off.kappa.idxmax(), 'test']]}", "tables/fusion.csv")
    put("fusNFeats", f"{f.n_feats.min()} to {f.n_feats.max()}", "tables/fusion.csv")
    put("fusBeats", int((f.kappa > f.best_single_kappa).sum()), "tables/fusion.csv kappa > best_single_kappa")
    put("fusPairs", len(f), "tables/fusion.csv")
    body = "\\begin{tabular}{@{}llcc@{}}\n\\toprule\nTrained on & Tested on & Forest $\\kappa$ & Best single \\\\\n\\midrule\n"
    for _, r in f.iterrows():
        body += f"{cref(r.train)} & {cref(r.test)} & {m2(r.kappa)} {{\\scriptsize [{m2(r.kappa_lo)}, {m2(r.kappa_hi)}]}} & {mcite(r.best_single)} ({m2(r.best_single_kappa)}) \\\\\n"
    save("tab_fusion", body + "\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------ RQ3 gaming: cheap edits, study, gradient attack
EDIT_NAME = {"saturation+sharpen@moderate": "S+S", "saturation+sharpen+clahe@moderate": "S+S+C",
             "saturation+sharpen+noise@moderate": "S+S+N", "clahe@moderate": "CLAHE", "red_boost@moderate": "Red boost",
             "noise@moderate": "Noise"}


def edits(w):
    E = ["uid2021", "saud", "sota"]
    e = pd.concat([pd.read_csv(R / f"{s}_edits.csv").assign(ds=s) for s in E])
    em = e[e.metric.isin(DROP | {UCIQE_M})]
    e = e[~e.metric.isin(DROP | {UCIQE_M})]
    n = e.groupby("ds").metric.nunique()
    assert n.nunique() == 1
    put("nEditMetrics", int(n.iloc[0]), "mos/results/<ds>_edits.csv")
    for s in E:
        tag = s.capitalize().replace("2021", "A")
        put(f"nEditScenes{tag}", int(e[e.ds == s].n.max()), f"mos/results/{s}_edits.csv")
    mx = e.groupby(["ds", "metric"]).beat_best.max()
    for s in E:
        tag = s.capitalize().replace("2021", "A")
        put(f"nEditHalf{tag}", int((mx[s] > 0.5).sum()), f"mos/results/{s}_edits.csv")
    ssc = e[(e.metric == "uciqe") & (e.edit == "saturation+sharpen+clahe@moderate")].set_index("ds").beat_best
    for s in E:
        put(f"uciqeSSC{s.capitalize().replace('2021', 'A')}", pct(ssc[s]), f"mos/results/{s}_edits.csv")
    put("uciqeSSCMax", pct(ssc.max()), "mos/results/<ds>_edits.csv")
    msc = em[(em.metric == UCIQE_M) & (em.edit == "saturation+sharpen+clahe@moderate")].set_index("ds").beat_best
    put("uciqeMatSSCMin", pct(msc.min()), "mos/results/<ds>_edits.csv uciqe_matlab_fix")
    put("uciqeMatSSCMax", pct(msc.max()), "mos/results/<ds>_edits.csv uciqe_matlab_fix")
    nz = e[e.edit == "noise@moderate"].groupby("metric").beat_best.max()  # noise is never an enhancement
    for m in ["uciqe", "fdum", "quiq"]:
        put(f"noiseWin{m.capitalize()}", pct(nz[m]), "mos/results/<ds>_edits.csv, noise@moderate, max over sets")
    # metrics where no edit beats the best in over half the scenes on any of the three sets
    safe = mx.groupby("metric").max()
    put("nEditSafe", int((safe <= 0.5).sum()), "mos/results/<ds>_edits.csv")
    put("editSafeNames", ", ".join(name(m) for m in safe[safe <= 0.5].sort_values().index), "same")
    # table: share of scenes where the edit beats the people's best
    sel = ["uiqm", "uciqe", "ccf", "fdum", "uranker", "nr_uiqa", "twicemix", "niqe", "brisque", "musiq", "clipiqa+", "qalign_8bit", "visualquality_r1", "rali"]
    eds = ["saturation+sharpen@moderate", "saturation+sharpen+clahe@moderate", "clahe@moderate"]
    piv = e.pivot_table(index="metric", columns=["edit", "ds"], values="beat_best")
    body = "\\begin{tabular}{@{}l" + "ccc" * len(eds) + "@{}}\n\\toprule\n & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{EDIT_NAME[x]}}}" for x in eds) + " \\\\\n"
    body += "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(eds))) + "\n"
    body += "Metric & " + " & ".join("UID & SAUD & SOTA" for _ in eds) + " \\\\\n\\midrule\n"
    for m in sel:
        cells = []
        for x in eds:
            for s in E:
                v = piv.loc[m, (x, s)]
                c = f"{100 * v:.0f}"
                cells.append(f"\\textbf{{{c}}}" if v > 0.5 else c)
        body += mcite(m) + " & " + " & ".join(cells) + " \\\\\n"
        if m in ("fdum", "twicemix", "clipiqa+"):
            body += "\\midrule\n"
    body += "\\midrule\n\\multicolumn{" + str(1 + 3 * len(eds)) + "}{@{}l}{Metrics (of " + str(int(n.iloc[0])) + ") where some edit beats the best in over half the scenes: " + \
        ", ".join(f"{REF_NAME[s]} {int((mx[s] > 0.5).sum())}" for s in E) + "} \\\\\n"
    save("tab_edits", body + "\\bottomrule\n\\end{tabular}\n")
    # full table, supplement
    body = "\\begin{longtable}{@{}l" + "c" * 6 + "@{}}\n\\caption{Share of scenes (\\%) in which the simple edit of the raw photo scores above the people's best enhancement, all metrics, maximum over the three MOS sets and, in brackets, the set where it is reached. Values are rounded, so a printed 50 can be just above half. NUIQ and LIQE-mix were not run on the edits.}\\label{tab:edits_all}\\\\\n\\toprule\nMetric & " + " & ".join(EDIT_NAME[x] for x in EDIT_NAME) + " \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nMetric & " + " & ".join(EDIT_NAME[x] for x in EDIT_NAME) + " \\\\\n\\midrule\n\\endhead\n"
    for m in sorted(piv.index, key=lambda m: name(m).lower()):
        cells = []
        for x in EDIT_NAME:
            v = piv.loc[m, x]
            cells.append(f"{100 * v.max():.0f} {{\\scriptsize ({REF_NAME[v.idxmax()].replace('UID2021', 'UID')})}}")
        body += mcite(m) + " & " + " & ".join(cells) + " \\\\\n"
    save("tab_edits_all", body + "\\bottomrule\n\\end{longtable}\n")


COND = {"clahe_mild": "CLAHE, mild", "contrast_mild": "Contrast, mild", "sharpen_mild": "Sharpen, mild", "sat_mild": "Saturation, mild",
        "sat_mod": "Saturation, moderate", "hue_mod": "Hue shift, moderate", "noise_mild": "Noise, mild", "red_strong": "Red boost, strong",
        "posterize_strong": "Posterize, strong", "chain_5": "S+S+C chain, 5 steps", "noise_strong": "Noise, strong", "chain_20": "S+S+C chain, 20 steps"}


def study(w):
    sp = pd.read_csv(R / "study_pairs.csv", dtype={"image": str})
    h = sp[sp.metric == "uciqe"].assign(k=lambda d: d.human_pref_edit * d.n_votes).groupby("condition").agg(k=("k", "sum"), n=("n_votes", "sum"))
    h["k"] = h.k.round().astype(int)
    sel = ["uciqe", "uiqm", "niqe", "qalign_8bit", "rali"]
    mp = sp[sp.metric.isin(sel)].groupby(["condition", "metric"]).metric_prefers_edit.mean().unstack()
    h = h.join(mp).sort_values("k", ascending=False)
    body = "\\begin{tabular}{@{}lc" + "c" * len(sel) + "@{}}\n\\toprule\n & People & \\multicolumn{" + str(len(sel)) + "}{c}{Metric prefers the edit (\\%)} \\\\\n\\cmidrule(lr){3-" + str(2 + len(sel)) + "}\n"
    body += "Edit of the raw photo & prefer edit (\\%) & " + " & ".join(mcite(m) for m in sel) + " \\\\\n\\midrule\n"
    ci = pref_ci()
    assert np.allclose(ci.pref_edit[h.index], h.k / h.n, atol=1e-3)
    for c, r in h.iterrows():
        lo, hi = ci.lo[c], ci.hi[c]
        body += f"{COND[c]} & {100 * r.k / r.n:.0f} {{\\scriptsize [{100 * lo:.0f}, {100 * hi:.0f}]}} & " + " & ".join(f"{100 * r[m]:.0f}" for m in sel) + " \\\\\n"
    save("tab_study", body + "\\bottomrule\n\\end{tabular}\n")
    rej = h[h.k / h.n < 0.5]  # conditions people reject
    put("nRejected", len(rej), "mos/results/study_pairs.csv")
    put("nRejectedSure", int((ci.hi < 0.5).sum()), "mos/results/study_pref_ci.csv (rater x photo bootstrap upper bound below half)")
    eq = sp[sp.metric == "uciqe"].groupby("condition").human_pref_edit.mean()  # each photo weighted equally
    flip = eq.index[(eq >= 0.5) != (h.k / h.n >= 0.5).reindex(eq.index)]
    assert list(flip) == ["hue_mod"], flip  # text names this edit
    put("nRejectedPhoto", int((eq < 0.5).sum()), "mos/results/study_pairs.csv (mean over photos of per-photo vote share)")
    put("hueVote", pct(h.loc["hue_mod", "k"] / h.loc["hue_mod", "n"]), "same, vote-weighted")
    put("huePhoto", pct(eq["hue_mod"]), "same, photo-weighted")
    put("surePhotoMax", pct(eq[ci.index[ci.hi < 0.5]].max()), "same, photo-weighted, edits with upper bound below half")
    put("uciqeOnRejected", pct(rej.uciqe.mean()), "mos/results/study_pairs.csv (mean over rejected conditions)")
    put("uiqmOnRejected", pct(rej.uiqm.mean()), "mos/results/study_pairs.csv")
    put("humanChainTwenty", pct(h.loc["chain_20", "k"] / h.loc["chain_20", "n"]), "mos/results/study_pairs.csv")
    put("uciqeChainTwenty", pct(h.loc["chain_20", "uciqe"]), "mos/results/study_pairs.csv")
    put("humanNoiseStrong", pct(h.loc["noise_strong", "k"] / h.loc["noise_strong", "n"]), "mos/results/study_pairs.csv")
    put("uciqeNoiseStrong", pct(h.loc["noise_strong", "uciqe"]), "mos/results/study_pairs.csv")
    st = pd.read_csv(R / "study_kappa_all.csv").set_index("metric")
    put("uciqePrefersEdit", pct(st.loc["uciqe", "metric_prefers_edit"]), "mos/results/study_kappa_all.csv")
    put("humanPrefersEdit", pct(h.k.sum() / h.n.sum()), "mos/results/study_pairs.csv")


ATT = ["musiq", "liqe", "topiq_nr", "clipiqa+", "rali"]


def gradattack():
    g = pd.concat(pd.read_csv(R / f"gradattack_{s}.csv") for s in ["uid2021", "saud", "sota"])
    s = g[g.attacked == g.metric].assign(cb=lambda d: d.clean > d.best, ab=lambda d: d.adv > d.best)
    sc = s.groupby("ds").scene.nunique()
    assert sc.nunique() == 1
    put("nAttScenes", int(sc.iloc[0]), "mos/results/gradattack_<ds>.csv")
    put("nAttScenesAll", int(sc.sum()), "mos/results/gradattack_<ds>.csv")
    put("nAttMetrics", s.attacked.nunique(), "mos/results/gradattack_<ds>.csv")
    a = s.groupby(["eps"]).ab.mean()
    assert (a == 1).all(), a
    put("attWin", pct(a.min()), "mos/results/gradattack_<ds>.csv")
    c = s[s.eps == 2].groupby(["attacked", "ds"]).cb.mean()
    put("cleanMin", pct(c.min()), "mos/results/gradattack_<ds>.csv")
    put("cleanMax", pct(c.max()), "mos/results/gradattack_<ds>.csv")
    mean = g.groupby(["ds", "attacked", "eps"]).psnr.mean()
    for e in (2, 8):
        put(f"psnrMean{'Two' if e == 2 else 'Eight'}", f"{mean.xs(e, level='eps').min():.1f} to {mean.xs(e, level='eps').max():.1f}", "mos/results/gradattack_<ds>.csv")
        pe = g[g.eps == e].psnr
        put(f"psnrMin{'Two' if e == 2 else 'Eight'}", f"{pe.min():.1f}", "mos/results/gradattack_<ds>.csv")
    body = "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Clean raw photo beats best (\\%)} & \\multicolumn{2}{c}{Attacked, $\\epsilon$ (\\%)} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nAttacked metric & UID2021 & SAUD & SOTA & 2/255 & 8/255 \\\\\n\\midrule\n"
    for m in ATT:
        ab = s[s.attacked == m].groupby("eps").ab.mean()
        body += mcite(m) + " & " + " & ".join(f"{100 * c[(m, d)]:.0f}" for d in ["uid2021", "saud", "sota"]) + f" & {100 * ab[2]:.0f} & {100 * ab[8]:.0f} \\\\\n"
    save("tab_attack", body + "\\bottomrule\n\\end{tabular}\n")
    # transfer: each attacked image scored by the other metrics (supplement)
    t = g.assign(ab=g.adv > g.best, cb=g.clean > g.best)
    tm = t[t.eps == 8].groupby(["attacked", "metric", "ds"])[["cb", "ab"]].mean()
    targets = [m for m in ["musiq", "liqe", "topiq_nr", "clipiqa+", "rali", "uiqm", "uciqe"] if m in t.metric.unique()]
    body = "\\begin{tabular}{@{}l" + "c" * len(targets) + "@{}}\n\\toprule\nAttacked $\\downarrow$ / scored by $\\rightarrow$ & " + " & ".join(mcite(m) for m in targets) + " \\\\\n\\midrule\n"
    clean_row = t[t.eps == 8].groupby(["metric"]).cb.mean()
    body += "Clean (no attack) & " + " & ".join(f"{100 * clean_row[m]:.0f}" for m in targets) + " \\\\\n\\midrule\n"
    for a_ in ATT:
        row = t[(t.eps == 8) & (t.attacked == a_)].groupby("metric").ab.mean()
        body += mcite(a_) + " & " + " & ".join(f"{100 * row[m]:.0f}" if m in row.index else "n/a" for m in targets) + " \\\\\n"
    save("tab_transfer", body + "\\bottomrule\n\\end{tabular}\n")
    rr = t[(t.eps == 8) & (t.attacked == "rali")].groupby("metric").ab.mean()
    put("raliToMusiq", pct(rr["musiq"]), "mos/results/gradattack_<ds>.csv (8/255, all three sets)")
    put("raliToLiqe", pct(rr["liqe"]), "same")
    put("cleanMusiq", pct(clean_row["musiq"]), "same")
    put("cleanLiqe", pct(clean_row["liqe"]), "same")


# ------------------------------------------------------------------ RQ4 drift
def drift_all():
    det, rows, q = {}, [], {}
    for c, cname in CHAINS.items():
        d, p = drift(c)
        assert len(d) == 73, (c, len(d))
        det[c] = set(d.index[d.detects])
        tag = cname.replace(" ", "").replace("Pro", "Pro").replace("NBB", "NB")
        tag = {"replica": "ProRep", "nb2_replica": "NbRep", "enhance": "ProEnh", "nb2_enhance": "NbEnh"}[c]
        put(f"nDetect{tag}", len(det[c]), f"out_banana_{c}/drift_all.csv")
        dl = last_delta(d)
        put(f"uciqeD{tag}", ("+" if d.loc["uciqe", dl] > 0 else "") + f2(d.loc["uciqe", dl]), f"out_banana_{c}/drift_all.csv")
        put(f"uciqeMatD{tag}", ("+" if d.loc[UCIQE_M, dl] > 0 else "") + f2(d.loc[UCIQE_M, dl]), f"out_banana_{c}/drift_all.csv")
        pr = p.groupby("metric").passed.mean()
        put(f"perImgBest{tag}", pct(pr.max()), f"out_banana_{c}/drift_per_image.csv")
        put(f"perImgBestName{tag}", " and ".join(name(m) for m in pr[pr == pr.max()].index), f"out_banana_{c}/drift_per_image.csv")
        put(f"raliPerImg{tag}", pct(pr["rali"]), f"out_banana_{c}/drift_per_image.csv")
        put(f"uciqePerImg{tag}", pct(pr["uciqe"]), f"out_banana_{c}/drift_per_image.csv")
        # noise quantification (banana_quant.py), step 1 = first edit
        qq = pd.read_csv(REPO / f"out_banana_{c}/quant.csv", dtype={"image": str})
        inp = qq[qq.step == 0].set_index("image").aspect
        qq = qq[qq.step > 0]
        first, last = qq.step.min(), qq.step.max()
        m = qq.groupby("step").mean(numeric_only=True)
        a, b = qq[qq.step == first].set_index("image"), qq[qq.step == last].set_index("image")
        q[c] = dict(steps=int(last), s1=m.sigma[first], sl=m.sigma[last], ratio=m.sigma[last] / m.sigma[first],
                    up=int((b.sigma > a.sigma).sum()), n=len(a), dE=m.dE1[last], a1=m.cast_a[first], al=m.cast_a[last],
                    aspect=(qq.aspect / qq.image.map(inp) - 1).abs().gt(0.02).mean())
        put(f"steps{tag}", int(last), f"out_banana_{c}/quant.csv")
        put(f"sigmaRatio{tag}", f"{q[c]['ratio']:.1f}", f"out_banana_{c}/quant.csv")
        put(f"dE{tag}", f"{q[c]['dE']:.1f}", f"out_banana_{c}/quant.csv")
        put(f"sigmaUp{tag}", f"{q[c]['up']} of {q[c]['n']}", f"out_banana_{c}/quant.csv")
    put("nDriftImg", q["replica"]["n"], "out_banana_replica/quant.csv")
    put("nDriftImages", big(sum(v["n"] * v["steps"] for v in q.values())), "out_banana_<chain>/quant.csv (generated steps)")
    put("sigmaRatioMin", f"{min(v['ratio'] for v in q.values()):.1f}", "out_banana_<chain>/quant.csv")
    put("sigmaRatioMax", f"{max(v['ratio'] for v in q.values()):.1f}", "out_banana_<chain>/quant.csv")
    put("dEMin", f"{min(v['dE'] for v in q.values()):.1f}", "out_banana_<chain>/quant.csv")
    put("dEMax", f"{max(v['dE'] for v in q.values()):.1f}", "out_banana_<chain>/quant.csv")
    put("nSigmaUp", f"{sum(v['up'] for v in q.values())} of {sum(v['n'] for v in q.values())}", "out_banana_<chain>/quant.csv")
    d0 = drift("replica")[0]
    put("nDriftScores", len(d0), "out_banana_<chain>/drift_all.csv rows")
    put("nDriftMetrics", len(d0) - 1, "drift_all.csv rows minus the MATLAB UCIQE diagnostic")
    assert UCIQE_M in d0.index and "liqe_mix" not in d0.index
    pro = det["replica"] & det["enhance"]
    nb2 = det["nb2_replica"] & det["nb2_enhance"]
    four = pro & nb2
    put("nBothPro", len(pro), "drift_all.csv")
    put("nBothNb", len(nb2), "drift_all.csv")
    put("nAllFour", len(four), "drift_all.csv")
    put("allFourNames", ", ".join(name(m) for m in sorted(four, key=lambda m: name(m).lower())), "drift_all.csv")
    defaults = [m for m in four if "-" not in m]
    put("nAllFourDefault", len(defaults), "drift_all.csv (names without a checkpoint suffix)")
    put("allFourDefaultNames", ", ".join(name(m) for m in defaults), "drift_all.csv")
    last = {c: drift(c)[0] for c in CHAINS}
    last = pd.DataFrame({c: d[last_delta(d)] for c, d in last.items()})
    put("nLastDownAll", int((last < 0).all(axis=1).sum()), "drift_all.csv (mean change at the last step below 0 on all four chains)")
    put("nLastDownProRep", int((last.replica < 0).sum()), "out_banana_replica/drift_all.csv (last step only)")
    put("uciqeUpAll", int(all(m not in det[c] for c in CHAINS for m in ["uciqe"])), "drift_all.csv")
    hand = [m for m in HAND_UW if any(m in det[c] for c in CHAINS)]
    put("handDetectAny", " and ".join(name(m) for m in hand) or "none", "drift_all.csv")
    # main drift table: noise + selected scores (check mark = detects by the mean rule)
    sel = ["uiqm", "uciqe", "uiconm", "niqe", "brisque", "musiq", "uranker", "qalign_8bit", "visualquality_r1", "rali"]
    body = "\\begin{tabular}{@{}l" + "c" * 4 + "@{}}\n\\toprule\n & " + " & ".join(CHAINS.values()) + " \\\\\n\\midrule\n"
    body += "Steps & " + " & ".join(str(q[c]["steps"]) for c in CHAINS) + " \\\\\n"
    body += "Noise $\\sigma$ (8-bit levels), first to last step & " + " & ".join(f"{q[c]['s1']:.1f} to {q[c]['sl']:.1f}" for c in CHAINS) + " \\\\\n"
    body += "Images whose noise grows & " + " & ".join(f"{q[c]['up']}/{q[c]['n']}" for c in CHAINS) + " \\\\\n"
    body += "Color change $\\Delta E^*_{ab}$, last to first step & " + " & ".join(f"{q[c]['dE']:.1f}" for c in CHAINS) + " \\\\\n"
    m1 = lambda x: f"$-${abs(x):.1f}" if round(x, 1) < 0 else f"{x:.1f}"  # noqa: E731
    body += "Mean $a^*$, first to last step & " + " & ".join(f"{m1(q[c]['a1'])} to {m1(q[c]['al'])}" for c in CHAINS) + " \\\\\n"
    body += "\\midrule\nScores that detect drift (of " + str(len(d0)) + ") & " + " & ".join(str(len(det[c])) for c in CHAINS) + " \\\\\n\\midrule\n"
    for m in sel:
        cells = []
        for c in CHAINS:
            d, p = drift(c)
            dl = last_delta(d)
            sym = "\\cmark" if m in det[c] else "\\xmark"
            cells.append(f"{sym} {m2(d.loc[m, dl])}")
        body += mcite(m) + " & " + " & ".join(cells) + " \\\\\n"
    save("tab_drift", body + "\\bottomrule\n\\end{tabular}\n")
    # supplement: every score, every chain
    body = "\\begin{longtable}{@{}l" + "c" * 4 + "@{}}\n\\caption{Drift test for all " + str(len(d0)) + " scores: " + str(len(d0) - 1) + " metrics (LIQE-mix was not run on the chains) and, as a diagnostic, the patched MATLAB UCIQE code. Each cell: mean change of the min-max normalized score from step 1 to the last step (negative = the score falls as the photo degrades), per-image pass rate in brackets, \\cmark\\ when the score detects the drift (below step 1 at every checked step).}\\label{tab:drift_all}\\\\\n\\toprule\nScore & " + " & ".join(CHAINS.values()) + " \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nScore & " + " & ".join(CHAINS.values()) + " \\\\\n\\midrule\n\\endhead\n"
    D = {c: drift(c) for c in CHAINS}
    for m in sorted(D["replica"][0].index, key=lambda m: name(m).lower()):
        cells = []
        for c in CHAINS:
            d, p = D[c]
            pr = p[p.metric == m].passed.mean()
            cells.append(("\\cmark\\ " if m in det[c] else "") + f"{m2(d.loc[m, last_delta(d)])} {{\\scriptsize ({100 * pr:.0f})}}")
        body += mcite(m) + " & " + " & ".join(cells) + " \\\\\n"
    save("tab_drift_all", body + "\\bottomrule\n\\end{longtable}\n")
    return det


# ------------------------------------------------------------------ Why: UCIQE implementations, geometry
def why(w):
    wm = kappa_wide(with_matlab=True)
    rows = []
    for s in SETS:
        o = pd.read_csv(REPO / f"scratch/octave_scores/mos/scores/{s}/uciqe_matlab.csv")
        a = pd.read_csv(REPO / f"mos/scores/{s}/uciqe.csv").set_index("id").score
        rho = a.corr(o.set_index("id").score, method="spearman")  # pairwise-complete: images where MATLAB returns a score
        rows.append((s, o.score.isna().mean(), rho))
        put(f"nanShare{s.capitalize().replace('2021', 'A').replace('2', 'B')}", pct(o.score.isna().mean()), f"scratch/octave_scores/mos/scores/{s}/uciqe_matlab.csv")
    put("nanMin", pct(min(r[1] for r in rows)), "scratch/octave_scores")
    put("nanMax", pct(max(r[1] for r in rows)), "scratch/octave_scores")
    put("implRhoMin", f"{min(r[2] for r in rows):.2f}", "mos/scores/<ds>/uciqe.csv vs scratch/octave_scores uciqe_matlab.csv, Spearman")
    put("implRhoMax", f"{max(r[2] for r in rows):.2f}", "mos/scores/<ds>/uciqe.csv vs scratch/octave_scores uciqe_matlab.csv, Spearman")
    body = "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n & " + " & ".join(cref(r).replace("Our study", "Study") for r in REFS) + " \\\\\n\\midrule\n"
    body += "NaN share, MATLAB code as released & " + " & ".join(pct(v) for _, v, _ in rows) + " & n/a \\\\\n"
    body += "$\\kappa$, our NumPy port & " + " & ".join(m2(wm.loc["uciqe", r]) for r in REFS) + " \\\\\n"
    body += "$\\kappa$, MATLAB code, NaN patched & " + " & ".join(m2(wm.loc[UCIQE_M, r]) for r in REFS) + " \\\\\n"
    save("tab_uciqe_impl", body + "\\bottomrule\n\\end{tabular}\n")
    put("kUciqeImplDiff", f"{(wm.loc[UCIQE_M, REFS].round(2) - wm.loc['uciqe', REFS].round(2)).abs().max():.2f}", "kappa tables, both UCIQE rows (printed values)")
    # the largest port UCIQE among the last Pro replica frames, and its share of near-black pixels (L* < 1)
    sys.path.insert(0, str(REPO))
    import cv2
    from PIL import Image
    from uwmetrics import uciqe
    best = max(((uciqe(np.asarray(Image.open(p).convert("RGB"))), p) for p in (REPO / "out_banana_replica/nbpro_replica").glob("*/50.png")),
               key=lambda t: t[0])
    L = cv2.cvtColor(np.asarray(Image.open(best[1]).convert("RGB"), np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    put("uciqeDarkId", best[1].parent.name, "out_banana_replica/nbpro_replica/*/50.png, uwmetrics.uciqe")
    put("uciqeDarkVal", f"{best[0]:.2f}", "same")
    put("uciqeDarkShare", pct((L < 1).mean()), "same, share of pixels with L* < 1")
    ko = pd.read_csv(R / "kappa_order.csv")  # mos/kappa_order.py
    ko = ko[ko.metric.isin(w.index)]
    put("kOrderMedMax", f"{ko.groupby('set').diff_abs.median().max():.2f}", "mos/results/kappa_order.csv")
    put("kOrderMax", f"{ko.diff_abs.max():.2f}", "same")
    put("kOrderRankMin", f"{ko.groupby('set').apply(lambda g: g.k_pub.rank().corr(g.k_sym.rank())).min():.2f}", "same, Spearman per set")
    vs = pd.concat([pd.read_csv(REPO / f"mos/scores/{s}/visualquality_r1.csv").score for s in SETS])
    put("vqBelowOne", pct((vs < 1).mean()), "mos/scores/<ds>/visualquality_r1.csv (scores below 1, four MOS sets)")
    vq = ko[ko.metric == "visualquality_r1"].tie_rate
    put("vqTieMin", pct(vq.min()), "same, pairs the metric ties")
    put("vqTieMax", pct(vq.max()), "same")
    for r in REFS:
        put(f"kUciqeMat{r.capitalize().replace('2021', 'A').replace('2', 'B')}", f2(wm.loc[UCIQE_M, r]), "kappa tables")
    rb = pd.read_csv(R / "robust.csv")
    rb = rb[rb.metric.isin(w.index)]
    fl = rb.pivot_table(index="metric", columns="t", values="flip_rate")
    for t in fl.columns:
        put(f"flipMed{t.capitalize().replace('90', 'N').replace('crop', 'Crop')}", pct(fl[t].median()), "mos/results/robust.csv")
        put(f"flipMax{t.capitalize().replace('90', 'N').replace('crop', 'Crop')}", pct(fl[t].max()), "mos/results/robust.csv")
    put("nRobust", len(fl), "mos/results/robust.csv")
    body = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nShare of within-scene pairs whose order flips & H. flip & 90$^\\circ$ turn & Half size & 90\\% crop \\\\\n\\midrule\n"
    for lab, f in [("Median over metrics", "median"), ("Maximum over metrics", "max")]:
        body += lab + " & " + " & ".join(pct(getattr(fl[t], f)()) for t in ["hflip", "rot90", "half", "crop90"]) + " \\\\\n"
    for m in ["uiqm", "uciqe", "musiq", "qalign_8bit", "rali"]:
        if m in fl.index:
            body += mcite(m) + " & " + " & ".join(pct(fl.loc[m, t]) for t in ["hflip", "rot90", "half", "crop90"]) + " \\\\\n"
    save("tab_robust", body + "\\bottomrule\n\\end{tabular}\n")


def compute():
    """Editor tokens and list-price cost (out_banana/usage.csv), and the wall time of the final scoring run."""
    u = pd.read_csv(REPO / "out_banana/usage.csv")
    put("nEditorImages", big(len(u)), "out_banana/usage.csv (one row per edited image)")
    # USD per million tokens (text input, image output), https://ai.google.dev/gemini-api/docs/pricing, read 2026-10-17
    price = {"gemini-3-pro-image": (2.0, 120.0), "gemini-3.1-flash-image": (0.5, 60.0)}
    assert set(u.model) == set(price), set(u.model)
    cost = sum(g.input_tokens.sum() / 1e6 * price[m][0] + g.output_tokens.sum() / 1e6 * price[m][1] for m, g in u.groupby("model"))
    put("editorCost", f"{cost:.0f}", "out_banana/usage.csv tokens x Gemini API list prices (scripts/make_numbers.py compute())")
    log = REPO / "scratch/logs/fill_all73.log"  # final run that scored every image with every metric missing a score
    with open(log, "rb") as f:
        first = f.readline().decode()
        f.seek(-4096, 2)
        last = f.read().decode(errors="ignore").strip().splitlines()[-1]
    t0, t1 = (pd.Timestamp("2026-" + x[:14]) for x in (first, last))
    assert last.endswith("all done"), last
    put("gpuHoursFinal", f"{(t1 - t0).total_seconds() / 3600:.0f}", "scratch/logs/fill_all73.log (first to last timestamp)")


def main():
    (IT / "notes").mkdir(exist_ok=True)
    T.mkdir(exist_ok=True)
    references()
    w = metrics()
    agreement(w)
    learned(w)
    edits(w)
    study(w)
    gradattack()
    drift_all()
    why(w)
    compute()
    tex ="% generated by scripts/make_numbers.py from the raw result files: do not edit by hand\n"
    for k, (v, src) in sorted(N.items()):
        tex += f"\\newcommand{{\\n{k}}}{{{v}\\xspace}} % src: {src}\n"
    (T / "numbers.tex").write_text(tex)
    (IT / "notes/numbers.json").write_text(json.dumps({k: {"value": v, "source": s} for k, (v, s) in sorted(N.items())}, indent=1))
    texs = {p.relative_to(IT).as_posix(): p.read_text() for p in [*IT.glob("sec/*.tex"), IT / "main.tex"]}
    rows = ["# Claim source map", "", "Generated by scripts/make_numbers.py. Every number in the text is a macro. Tables are generated from the same sources (see each tables/tab_*.tex header).", "",
            "| Macro | Value | Source | Used in |", "|---|---|---|---|"]
    for k, (v, src) in sorted(N.items()):
        used = ", ".join(f for f, t in sorted(texs.items()) if re.search(r"\\n" + k + r"(?![a-zA-Z])", t)) or "tables only or unused"
        rows.append(f"| `\\n{k}` | {v.replace('|', '/')} | {src.replace('|', '/')} | {used} |")
    (IT / "notes/claim_source_map.md").write_text("\n".join(rows) + "\n")
    print(len(N), "numbers;", len(list(T.glob("tab_*.tex"))), "tables")


if __name__ == "__main__":
    assert f2(-0.081) == "$-$0.08" and pct(0.954) == "95\\%"
    main()

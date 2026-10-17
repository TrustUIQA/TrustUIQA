"""O1: correlation of every metric with MOS, per dataset.

Writes results/tables/o1_<dataset>.csv and .md with SRCC/PLCC/KRCC + 95%
bootstrap CIs, intra-scene SRCC, and a paired-bootstrap significance test of
each metric against the best one.

    python scripts/benchmark.py --dataset uid2021
    python scripts/benchmark.py --dataset uid2021 --fast   # quick look
"""
import argparse

import pandas as pd

from uwiqa import RESULTS_ROOT
from uwiqa.data import load_manifest
from uwiqa.eval import correlate, grouped_srcc, paired_bootstrap_srcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fast", action="store_true", help="200 boots, single logistic fit")
    ap.add_argument("--gate", type=float, default=0.7, help="Week-4 gate SRCC for topiq_nr")
    args = ap.parse_args()

    man = load_manifest(args.dataset).dropna(subset=["mos"])
    scores = pd.read_csv(RESULTS_ROOT / "scores" / f"{args.dataset}.csv")
    wide = scores.pivot_table(index="image", columns="metric", values="score")
    hb = scores.groupby("metric")["higher_better"].first().astype(bool).to_dict()
    df = man.merge(wide, left_on="image", right_index=True, how="inner")
    if df.empty:
        raise SystemExit("no overlap between manifest and scores")

    n_boot = 200 if args.fast else args.n_boot
    rows = []
    for m in wide.columns:
        sub = df[["mos", "scene", m]].dropna()
        r = correlate(sub[m], sub["mos"], higher_better=hb[m], n_boot=n_boot,
                      bootstrap_plcc_fit=not args.fast).as_dict()
        g = grouped_srcc(sub[m], sub["mos"], sub["scene"], higher_better=hb[m])
        rows.append({"metric": m, **r, "intra_scene_srcc": g["mean"],
                     "n_scenes": g["n_groups"]})
    tab = pd.DataFrame(rows).sort_values("srcc", ascending=False).reset_index(drop=True)

    best = tab.loc[0, "metric"]
    sig = []
    for m in tab["metric"]:
        if m == best:
            sig.append(float("nan"))
            continue
        sub = df[["mos", best, m]].dropna()
        sig.append(paired_bootstrap_srcc(sub[best], sub[m], sub["mos"], hb[best], hb[m])["p"])
    tab[f"p_vs_{best}"] = sig

    out = RESULTS_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"o1_{args.dataset}.csv", index=False)
    fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    md = ["| metric | SRCC | PLCC | KRCC | intra-scene SRCC | p vs best |", "|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r.metric} | {fmt(r, 'srcc')} | {fmt(r, 'plcc')} | {fmt(r, 'krcc')} | "
                  f"{r.intra_scene_srcc:.3f} | {r[f'p_vs_{best}']:.3g} |")
    (out / f"o1_{args.dataset}.md").write_text("\n".join(md))
    print("\n".join(md))

    if "topiq_nr" in tab["metric"].values:
        s = tab.set_index("metric").loc["topiq_nr", "srcc"]
        print(f"\nWeek-4 gate (topiq_nr SRCC >= {args.gate}): "
              f"{'PASS' if s >= args.gate else 'FAIL'} ({s:.3f})")


if __name__ == "__main__":
    main()

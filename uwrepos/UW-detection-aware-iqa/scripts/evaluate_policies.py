"""Headline experiment: does per-image selective enhancement beat all-raw / all-enhanced?

Policies (dataset mAP on RUOD pred_test and DUO clean, from saved detections):
  all_raw, all_<m>         fixed choices
  oracle                   per-image best variant by true AP (upper bound, not deployable)
  noise_oracle             per-image best of raw vs the invisible JPEG re-encode: how much
                           "oracle gain" pure label noise produces (the honest control)
  <model>|d, <model>|p     learned predictor: pick argmax predicted delta (d) or P(help) (p);
                           enhance only if it exceeds a margin tuned on RUOD pred_val
  iqa_delta-<k>|tuned      same rule with Q(enh) - Q(raw) as the score
  iqa_delta-<k>|naive      common practice: always use the enhancer with the best IQA gain
The best learned policy is chosen on pred_val (never on test) and compared with all_raw
and the best fixed enhancer by a paired bootstrap.

    python scripts/evaluate_policies.py
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from uwiqa import RESULTS_ROOT
from uwiqa.detect.metrics import Dets, yolo_labels_to_gt
from uwiqa.detect.policy import MatchCache
from uwiqa.model.predictor import NULL, load_utility

import build_utility_labels as bul


def gts_for(set_name, images, classes):
    _, root = bul.load_set(set_name)
    out = {}
    for rel in images:
        g = yolo_labels_to_gt(bul.label_path(root, rel).read_text())
        if classes is not None:
            k = np.isin(g.cls, classes)
            g = Dets(g.boxes[k], g.cls[k])
        out[rel] = g
    return out


def choose(S, methods, delta, prior):
    """argmax over methods of S (+ tiny prior to break ties); raw unless the max exceeds delta."""
    S = S + prior[None, :] * 1e-9
    j = S.argmax(1)
    best = S[np.arange(len(S)), j]
    return np.where(best > delta, np.array(methods)[j], "raw")


def tune_delta(S, methods, objective, prior, n_cands=41):
    """Margin maximising objective(choice) on validation (also allows never/always enhancing)."""
    best_scores = (S + prior[None, :] * 1e-9).max(1)
    cands = np.r_[-np.inf, np.quantile(best_scores, np.linspace(0, 1, n_cands)), np.inf]
    obj = [objective(choose(S, methods, d, prior)) for d in cands]
    return float(cands[int(np.argmax(obj))])


class APTable:
    def __init__(self, wide):
        self.w = wide

    def lookup_mean(self, choice):
        return float(np.mean([self.w.at[i, m] for i, m in zip(self.w.index, choice)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--metric", default="ap")
    ap.add_argument("--objective", default="image_ap", choices=["image_ap", "map"],
                    help="what the margin is tuned to maximise on pred_val: mean per-image AP or dataset mAP")
    args = ap.parse_args()
    out = RESULTS_ROOT / "policies"
    sfx = "" if args.objective == "image_ap" else f"_{args.objective}"
    out.mkdir(parents=True, exist_ok=True)

    W_r, _ = load_utility("ruod_pred", args.metric)
    W_d, _ = load_utility("duo_clean", args.metric)
    split = pd.read_csv(RESULTS_ROOT / "splits" / "ruod_splits.csv").set_index("image")["split"]
    methods = [c for c in W_r.columns if c not in ("raw", NULL)]
    sets = {
        "val": W_r.loc[split.reindex(W_r.index).values == "pred_val"],
        "test": W_r.loc[split.reindex(W_r.index).values == "pred_test"],
        "duo": W_d,
    }
    prior = sets["val"][methods].sub(sets["val"]["raw"], axis=0).mean().values  # tie-break only

    pred_files = sorted(glob.glob(str(RESULTS_ROOT / "predictor" / "preds" / "*.csv")))
    scorers = {}  # name -> split -> S (n, M), plus naive flag
    for f in pred_files:
        model = Path(f).stem
        P = pd.read_csv(f, index_col="image")
        for col, tag in (("d_", "d"), ("p_", "p")):
            cols = [col + m for m in methods]
            if not set(cols) <= set(P.columns):
                continue
            if model.startswith("iqa_") and tag == "p":  # baselines store the same score twice
                continue
            S = {s: P[P.split == s].reindex(sets[s].index)[cols].values for s in sets}
            name = model if model.startswith("iqa_") else f"{model}|{tag}"
            scorers[name] = S

    ap_val = APTable(sets["val"])
    if args.objective == "map":
        val_imgs = list(sets["val"].index)
        print(f"[val] building match cache for {len(val_imgs)} images (objective = dataset mAP) ...", flush=True)
        val_cache = MatchCache(str(RESULTS_ROOT / "utility" / "ruod_pred" / "main" / "preds_*.npz"),
                               gts_for("ruod_pred", val_imgs, None))
        objective = lambda ch: val_cache.map(val_imgs, list(ch))["map"]
        n_cands = 21
    else:
        objective, n_cands = ap_val.lookup_mean, 41
    tuned = {}
    rows, choices = {"test": [], "duo": []}, {"test": {}, "duo": {}}
    val_obj = {}
    for s, set_name, classes in (("test", "ruod_pred", None), ("duo", "duo_clean", [0, 1, 2, 3])):
        W = sets[s]
        imgs = list(W.index)
        print(f"\n[{s}] building match cache for {len(imgs)} images ...", flush=True)
        cache = MatchCache(str(RESULTS_ROOT / "utility" / set_name / "main" / "preds_*.npz"),
                           gts_for(set_name, imgs, classes))
        pol = {"all_raw": ["raw"] * len(imgs)}
        pol.update({f"all_{m}": [m] * len(imgs) for m in methods})
        allv = ["raw"] + methods
        best = W[allv].values.argmax(1)  # ties -> raw (first column)
        pol["oracle"] = list(np.array(allv)[best])
        pol["noise_oracle"] = list(np.where(W[NULL].values > W["raw"].values, NULL, "raw"))
        for name, S in scorers.items():
            if np.isnan(S[s]).any() or np.isnan(S["val"]).any():
                continue
            if name not in tuned:  # tune once, reuse for test and duo
                d0 = tune_delta(S["val"], methods, objective, prior, n_cands)
                tuned[name] = (d0, objective(choose(S["val"], methods, d0, prior)))
            d = tuned[name][0]
            pol[f"{name}|tuned" if name.startswith("iqa_") else name] = list(choose(S[s], methods, d, prior))
            val_obj[name] = tuned[name][1]
            if name.startswith("iqa_delta"):
                pol[f"{name}|naive"] = list(choose(S[s], methods, -np.inf, prior))
        base = cache.map(imgs, pol["all_raw"])
        for name, ch in pol.items():
            r = cache.map(imgs, ch)
            rows[s].append({"policy": name, "frac_enhanced": float(np.mean(np.array(ch) != "raw")),
                            "map50": r["map50"], "map": r["map"], "d_map_vs_raw": r["map"] - base["map"],
                            "mean_image_ap": float(np.mean([W.at[i, m] for i, m in zip(imgs, ch)]))})
        choices[s] = (cache, imgs, pol)

    learned = {k: v for k, v in val_obj.items() if not k.startswith("iqa_")}
    best_learned = max(learned, key=learned.get) if learned else None
    best_iqa = max((k for k in val_obj if k.startswith("iqa_")), key=val_obj.get, default=None)
    print(f"\nchosen on pred_val: best learned = {best_learned}, best IQA = {best_iqa}")

    for s, label in (("test", "RUOD pred_test"), ("duo", "DUO clean (margins tuned on RUOD)")):
        tab = pd.DataFrame(rows[s]).sort_values("map", ascending=False)
        tab.to_csv(out / f"policies_{s}{sfx}.csv", index=False)
        print(f"\n=== {label}: dataset mAP by policy ===")
        print(tab.round(4).to_string(index=False))
        cache, imgs, pol = choices[s]
        best_fixed = max((p for p in pol if p.startswith("all_") and p != "all_raw"),
                         key=lambda p: cache.map(imgs, pol[p])["map"])
        comps = []
        for a in [best_learned, f"{best_iqa}|tuned" if best_iqa and best_iqa.startswith("iqa_") else best_iqa,
                  "oracle", "noise_oracle"]:
            if a is None or a not in pol:
                continue
            for b in ("all_raw", best_fixed):
                r = cache.bootstrap_diff(imgs, pol[a], pol[b], n_boot=args.n_boot)
                comps.append({"policy": a, "vs": b, **r})
        comp = pd.DataFrame(comps)
        comp.to_csv(out / f"bootstrap_{s}{sfx}.csv", index=False)
        print(f"\npaired bootstrap of mAP50-95 difference ({args.n_boot} resamples):")
        print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

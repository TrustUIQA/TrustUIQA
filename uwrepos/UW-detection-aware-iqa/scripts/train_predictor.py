"""O4/O5: train the detection-aware NR predictor and compare it with NR-IQA baselines.

Train on RUOD pred_train, tune on pred_val, report on pred_test (in-domain) and
duo_clean (cross-dataset, nothing tuned on DUO).

Models (per backbone: clip_b32, dinov2_s14, resnet18, clip_b32+dinov2_s14):
  ridge   linear regression of the per-method delta
  logreg  per-method logistic "helps" classifier
  mlp     multi-task MLP (delta + helps + ranking loss), 3-seed ensemble
  mlp_norank   ablation without the ranking loss
Baselines (no training): for UIQM/UCIQE/TOPIQ/LIQE/URanker,
  iqa_raw-<k>    -Q(raw)            (worse-looking raw image => enhancement helps)
  iqa_delta-<k>  Q(enh_m) - Q(raw)  (the usual practice; needs the enhanced image)

    python scripts/train_predictor.py
"""
import argparse
import re

import numpy as np
import pandas as pd
import torch

from uwiqa import RESULTS_ROOT
from uwiqa.model.predictor import (NULL, Standardizer, best_threshold, evaluate, fit_logistic,
                                   fit_mlp, fit_ridge, load_embeddings, load_iqa, load_utility,
                                   noise_tau, predict_logistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*",
                    default=["clip_b32", "dinov2_s14", "resnet18", "clip_b32+dinov2_s14"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--metric", default="ap", choices=["ap", "ap50", "f1"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = RESULTS_ROOT / "predictor"
    (out / "preds").mkdir(parents=True, exist_ok=True)

    _, D_r = load_utility("ruod_pred", args.metric)
    _, D_d = load_utility("duo_clean", args.metric)
    split = pd.read_csv(RESULTS_ROOT / "splits" / "ruod_splits.csv").set_index("image")["split"]
    methods = [c for c in D_r.columns if c != NULL]
    idx = {s: D_r.index[split.reindex(D_r.index).values == f"pred_{s}"] for s in ("train", "val", "test")}
    idx["duo"] = D_d.index
    tau = noise_tau(D_r.loc[idx["train"], NULL].values)
    print(f"methods {methods}; noise threshold tau = {tau:.4f} (95th pct of |delta| for {NULL})")

    def DH(s):
        D = (D_r if s != "duo" else D_d).loc[idx[s], methods].values
        return D, (D > tau).astype(float)

    data = {s: DH(s) for s in idx}
    for s, (D, H) in data.items():
        print(f"  {s:5s} n={len(D):5d}  helps-rate per method: "
              + ", ".join(f"{m}={h:.2f}" for m, h in zip(methods, H.mean(0))))

    results = {"test": [], "duo": []}
    per_method = {"test": [], "duo": []}

    def record(name, preds, kind):
        """preds: split -> (dhat or None, score). Thresholds tuned on val."""
        thr = [best_threshold(data["val"][1][:, j], preds["val"][1][:, j]) for j in range(len(methods))]
        for s in ("test", "duo"):
            dh, sc = preds[s]
            t = evaluate(*data[s], dh, sc, thr, methods)
            per_method[s].append(t.assign(model=name))
            results[s].append({"model": name, "kind": kind, **t.drop(columns=["method"]).mean().to_dict()})
        rows = []
        for s in ("val", "test", "duo"):
            dh, sc = preds[s]
            df = pd.DataFrame(sc, columns=[f"p_{m}" for m in methods], index=idx[s])
            if dh is not None:
                df[[f"d_{m}" for m in methods]] = dh
            rows.append(df.assign(split=s))
        fn = re.sub(r"[^A-Za-z0-9_.+@-]", "_", name)
        pd.concat(rows).rename_axis("image").to_csv(out / "preds" / f"{fn}.csv")

    # ---------------- learned predictors
    for bb in args.backbones:
        E_r, E_d = load_embeddings("ruod_pred", bb), load_embeddings("duo_clean", bb)
        for s_, E in (("ruod_pred", E_r), ("duo_clean", E_d)):
            need = np.concatenate([idx[k] for k in idx if (k == "duo") == (s_ == "duo_clean")])
            miss = int((~pd.Index(need).isin(E.index)).sum())
            if miss:
                raise SystemExit(f"{miss} images of {s_} have no features yet: "
                                 f"run scripts/extract_features.py --set {s_} to completion first")
        X = {s: (E_r if s != "duo" else E_d).loc[idx[s]].values for s in idx}
        st = Standardizer().fit(X["train"])
        X = {s: st(v) for s, v in X.items()}
        (Dtr, Htr), (Dva, Hva) = data["train"], data["val"]

        rid, alpha = fit_ridge(X["train"], Dtr, X["val"], Dva)
        p = {s: rid.predict(X[s]) for s in ("val", "test", "duo")}
        record(f"ridge@{bb}", {s: (v, v) for s, v in p.items()}, "learned")

        lr = fit_logistic(X["train"], Htr, X["val"], Hva)
        record(f"logreg@{bb}", {s: (None, predict_logistic(lr, X[s])) for s in ("val", "test", "duo")},
               "learned")

        for variant, rw in (("mlp", 0.5), ("mlp_norank", 0.0)):
            outs = [fit_mlp(X["train"], Dtr, Htr, X["val"], Dva, tau, rank_w=rw, seed=k, device=dev)
                    for k in range(args.seeds)]
            ens = {}
            for s in ("val", "test", "duo"):
                pr = [f(X[s]) for f in outs]
                ens[s] = (np.mean([a for a, _ in pr], 0), np.mean([b for _, b in pr], 0))
            record(f"{variant}@{bb}", ens, "learned")
        print(f"  trained {bb} (ridge alpha={alpha})", flush=True)

    # ---------------- NR-IQA baselines
    Q_r, Q_d = load_iqa("ruod_pred"), load_iqa("duo_clean")
    for k in sorted(Q_r.columns.get_level_values(0).unique()):
        def raw_only(s):
            q = (Q_r if s != "duo" else Q_d).loc[idx[s], (k, "raw")].values
            return np.repeat(-q[:, None], len(methods), 1)

        def delta_q(s):
            Q = Q_r if s != "duo" else Q_d
            return np.stack([Q.loc[idx[s], (k, m)].values - Q.loc[idx[s], (k, "raw")].values
                             for m in methods], 1)

        for name, f in ((f"iqa_raw-{k}", raw_only), (f"iqa_delta-{k}", delta_q)):
            record(name, {s: (f(s), f(s)) for s in ("val", "test", "duo")}, "baseline")

    for s, label in (("test", "RUOD pred_test (in-domain)"), ("duo", "DUO clean (cross-dataset)")):
        tab = pd.DataFrame(results[s]).sort_values("auroc", ascending=False)
        tab.to_csv(out / f"summary_{s}.csv", index=False)
        pd.concat(per_method[s]).to_csv(out / f"per_method_{s}.csv", index=False)
        print(f"\n=== {label}: mean over {len(methods)} enhancers (sorted by AUROC) ===")
        print(tab.drop(columns=["pos_rate"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

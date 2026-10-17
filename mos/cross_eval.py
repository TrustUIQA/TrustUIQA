"""Protocol and generalisation checks on the MOS datasets (MOS is the human reference throughout).

  protocol  Abhinav et al.'s SAUD2.0 protocol per metric: random 75/25 split, map score -> MOS on the 75% with
            (a) a cubic polynomial (main paper) and (b) a 300-tree random forest on the degree-3 polynomial
            expansion (supplement), report test SRCC/KRCC/PLCC/RMSE. 10 splits, split over images (their protocol,
            the same photo's other enhancements can sit in train) and over scenes (no photo in both).
            -> mos/results/protocol_<ds>.csv
  disjoint  Metrics trained on one MOS set, scored on another set's scenes, all vs only those whose source photo is
            NOT in the training set (mos/results/overlap.csv; a lower bound on overlap, so "disjoint" is optimistic).
            -> mos/results/disjoint.csv
  fusion    All metrics shared by two sets, fused by a random forest trained on set A's MOS (z-scored), tested on
            set B's scenes not shared with A. Metrics trained on A or B are left out. Within-scene numbers only, as
            the MOS scales differ. -> mos/results/fusion.csv
Usage: python -I mos/cross_eval.py protocol saud2 [uid2021 ...] | disjoint | fusion
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from eval_mos import REPO, pair_kappa, scene_stats  # noqa: E402

SETS = ("uid2021", "saud", "saud2", "sota")
TRAINED = {"uiqavsi": "saud", "nr_uiqa": "saud", "nuiq": "saud", "nmfc": "saud", "quiq": "uid2021"}
SPLITS = 10


def load(ds):
    """Manifest (scene as str, as overlap.py reads it) + {metric: score oriented higher = better}, finite only."""
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    man["scene"] = man.scene.astype(str)
    sc = {}
    for f in sorted((REPO / "mos/scores" / ds).glob("*.csv")):
        d = man[["id"]].merge(pd.read_csv(f), on="id", how="left")
        s = np.where(d.lower_better.astype(bool), -d.score, d.score).astype(float)
        if len(d) == len(man) and np.isfinite(s).all():
            sc[f.stem] = s
    return man, sc


def shared(a, b):
    """Scenes of set b whose photo also appears in set a."""
    o = pd.read_csv(REPO / "mos/results/overlap.csv", dtype=str)
    return set(o[(o.set_a == a) & (o.set_b == b)].scene_b) | set(o[(o.set_b == a) & (o.set_a == b)].scene_a)


def within(scene, s, y, boot=0, seed=0):
    """Within-scene SRCC, pairwise accuracy and Cohen's kappa against MOS (eval_mos definitions); kappa CI if boot."""
    d = pd.DataFrame(dict(scene=scene, s=s, mos=y))
    st = np.array([scene_stats(g) for _, g in d.groupby("scene")])
    out = dict(scene_srcc=st[:, 0].mean(), pair_acc=st[:, 1].sum() / st[:, 2].sum(), kappa=pair_kappa(*st[:, 4:].sum(0)),
               scenes=len(st))
    if boot:
        rng = np.random.default_rng(seed)
        k = [pair_kappa(*st[rng.integers(0, len(st), len(st)), 4:].sum(0)) for _ in range(boot)]
        out.update(kappa_lo=np.percentile(k, 2.5), kappa_hi=np.percentile(k, 97.5))
    return out


def protocol(ds):
    man, sc = load(ds)
    y, scenes = man.mos.to_numpy(float), man.scene.to_numpy()
    us = np.unique(scenes)
    rows = []
    for m, s in sc.items():
        x = ((s - s.mean()) / (s.std() + 1e-12))[:, None]
        X3 = PolynomialFeatures(3, include_bias=False).fit_transform(x)
        for split in ("image", "scene"):
            res = []
            for r in range(SPLITS):
                rng = np.random.default_rng(r)
                if split == "image":
                    te = np.zeros(len(y), bool)
                    te[rng.permutation(len(y))[: len(y) // 4]] = True
                else:
                    te = np.isin(scenes, rng.permutation(us)[: len(us) // 4])
                cub = np.polyval(np.polyfit(x[~te, 0], y[~te], 3), x[te, 0])
                rf = RandomForestRegressor(300, random_state=r, n_jobs=8).fit(X3[~te], y[~te]).predict(X3[te])
                for mapper, p in (("cubic", cub), ("rf", rf)):
                    res.append(dict(mapper=mapper, srcc=abs(spearmanr(p, y[te])[0]), krcc=abs(kendalltau(p, y[te])[0]),
                                    plcc=abs(pearsonr(p, y[te])[0]), rmse=np.sqrt(np.mean((p - y[te]) ** 2))))
            g = pd.DataFrame(res).groupby("mapper")
            for mapper, v in g:
                rows.append(dict(metric=m, split=split, mapper=mapper, **{k: v[k].mean() for k in ("srcc", "krcc", "plcc", "rmse")},
                                 srcc_sd=v.srcc.std(), raw_srcc=spearmanr(s, y)[0]))
        print(ds, m, flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / f"mos/results/protocol_{ds}.csv", index=False)
    return out


def disjoint():
    rows = []
    for b in SETS:
        man, sc = load(b)
        for m, a in TRAINED.items():
            if m not in sc or a == b:
                continue
            drop = shared(a, b)
            keep = ~man.scene.isin(drop).to_numpy()
            y = man.mos.to_numpy(float)
            for subset, k in (("all", np.ones(len(man), bool)), ("disjoint", keep)):
                rows.append(dict(metric=m, trained_on=a, test_set=b, subset=subset, srcc=spearmanr(sc[m][k], y[k])[0],
                                 **within(man.scene.to_numpy()[k], sc[m][k], y[k], boot=1000)))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / "mos/results/disjoint.csv", index=False)
    return out


def fusion():
    data = {ds: load(ds) for ds in SETS}
    rows = []
    for a in SETS:
        for b in SETS:
            if a == b:
                continue
            (ma, sa), (mb, sb) = data[a], data[b]
            feats = sorted(m for m in set(sa) & set(sb) if TRAINED.get(m) not in (a, b))
            if len(feats) < 3:
                continue
            Xa, Xb = np.column_stack([sa[m] for m in feats]), np.column_stack([sb[m] for m in feats])
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-12  # train-set scaling only
            ya = ma.mos.to_numpy(float)
            rf = RandomForestRegressor(300, random_state=0, n_jobs=8).fit((Xa - mu) / sd, (ya - ya.mean()) / ya.std())
            keep = ~mb.scene.isin(shared(a, b)).to_numpy()
            p, yb = rf.predict((Xb - mu) / sd)[keep], mb.mos.to_numpy(float)[keep]
            best = max(feats, key=lambda m: within(ma.scene.to_numpy(), sa[m], ya)["kappa"])  # picked on A, like the RF
            rows.append(dict(train=a, test=b, n_feats=len(feats), srcc=spearmanr(p, yb)[0],
                             **within(mb.scene.to_numpy()[keep], p, yb, boot=1000), best_single=best,
                             best_single_kappa=within(mb.scene.to_numpy()[keep], sb[best][keep], yb)["kappa"]))
            print(a, b, len(feats), flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO / "mos/results/fusion.csv", index=False)
    return out


if __name__ == "__main__":
    # self-check: a metric equal to MOS is perfect within scene; its negation is perfectly wrong
    t = within(np.array(["a"] * 3 + ["b"] * 3), np.array([2., 1, 3, 3, 1, 2]), np.array([2., 1, 3, 5, 3, 4]))
    assert t["kappa"] == 1 and t["pair_acc"] == 1
    assert within(np.array(["a"] * 3), np.array([3., 2, 1]), np.array([1., 2, 3]))["pair_acc"] == 0
    cmd = sys.argv[1]
    if cmd == "protocol":
        for ds in sys.argv[2:]:
            o = protocol(ds)
            print(o.pivot_table(index="metric", columns=["split", "mapper"], values="srcc").round(3).to_string())
    else:
        print(globals()[cmd]().round(3).to_string(index=False))

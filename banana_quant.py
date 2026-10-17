"""Noise and failure quantification on the Nano Banana Pro chains (Banana100 Sec. 3-4 style), per image and step:

  sigma     Immerkaer (1996) noise estimate on the grey image, 8-bit units (Laplacian-difference mask)
  chroma    mean Lab chroma sqrt(a^2+b^2); cast_a / cast_b = mean a*, b* (underwater cast: b* or a* far from 0)
  dE1       mean CIEDE76 colour difference to step 1 (images resized to step 1's size)
  aspect    w/h; step 0 is the 512 px input, every generated step is the model's own size
plus the drift test of banana_drift.py over ALL scored metrics (scores/, scores_ext/, scores_cpqi/, i.e. also the
learned underwater metrics and RALI), and a per-image pass/fail table (Banana100 Fig. 5-6): a metric passes an
image if its own delta_k < 0 at every checkpoint.
Writes out_banana_<chain>/quant.csv, drift_all.csv, drift_per_image.csv. Usage: python -I banana_quant.py enhance|replica|nb2_enhance|nb2_replica
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from banana_drift import drift  # noqa: E402

M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def sigma(grey):
    h, w = grey.shape
    r = np.abs(cv2.filter2D(grey.astype(np.float64), -1, M, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    return np.sqrt(np.pi / 2) * r.sum() / (6 * (w - 2) * (h - 2))


def selfcheck():
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), 128.0)
    assert sigma(flat) < 1e-9
    assert abs(sigma(flat + rng.normal(0, 10, flat.shape)) - 10) < 0.5


def lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


if __name__ == "__main__":
    selfcheck()
    chain = sys.argv[1]
    d = Path(f"out_banana_{chain}")
    ks = [5, 10, 20] + ([50] if chain.endswith("replica") else [])
    rows = []
    for img in sorted(p for p in (d / (chain if chain.startswith("nb2_") else f"nbpro_{chain}")).iterdir() if p.is_dir()):
        files = sorted(img.glob("*.png"), key=lambda p: int(p.stem))
        ref = None
        for f in files:
            x = cv2.imread(str(f))
            L = lab(x)
            step = int(f.stem)
            if step == 1:
                ref = L
            r = dict(image=img.name, step=step, sigma=sigma(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
                     chroma=float(np.hypot(L[..., 1], L[..., 2]).mean()), cast_a=float(L[..., 1].mean()),
                     cast_b=float(L[..., 2].mean()), aspect=x.shape[1] / x.shape[0], w=x.shape[1], h=x.shape[0])
            if ref is not None:
                y = L if L.shape == ref.shape else lab(cv2.resize(x, ref.shape[1::-1], interpolation=cv2.INTER_AREA))
                r["dE1"] = float(np.sqrt(((y - ref) ** 2).sum(-1)).mean())
            rows.append(r)
    q = pd.DataFrame(rows)
    q.round(4).to_csv(d / "quant.csv", index=False)

    # drift over every scored metric, and per-image pass/fail
    allm, per = {}, []
    for sub in ("scores", "scores_ext", "scores_cpqi"):
        for f in sorted((d / sub).glob("*.csv")):
            if f.stem in allm:
                continue
            df = pd.read_csv(f, dtype={"image": str})
            df = df[df.step >= 1]
            if df.step.max() < max(ks):
                continue
            allm[f.stem] = drift(df, ks)
            s = df.score * (-1 if df.lower_better.iloc[0] else 1)
            w = df.assign(n=(s - s.min()) / (s.max() - s.min() or 1)).pivot_table(index="image", columns="step",
                                                                                 values="n")
            ok = np.logical_and.reduce([(w[k] - w[1]) < 0 for k in ks])
            per += [dict(metric=f.stem, image=i, passed=bool(v)) for i, v in zip(w.index, ok)]
    t = pd.DataFrame(allm).T
    t["detects"] = (t[[f"d{k}" for k in ks]] < 0).all(axis=1)
    t.sort_values(f"d{max(ks)}").to_csv(d / "drift_all.csv")
    pd.DataFrame(per).to_csv(d / "drift_per_image.csv", index=False)

    g = q.groupby("step")[["sigma", "chroma", "cast_a", "cast_b", "dE1"]].mean()
    print(g.loc[[s for s in [0, 1] + ks if s in g.index]].round(2).to_string())
    print(f"aspect differs from the input by >2%: "
          f"{(abs(q[q.step >= 1].aspect / q[q.step == 0].set_index('image').aspect.reindex(q[q.step >= 1].image).values - 1) > .02).mean():.0%} of generated images")
    pi = pd.DataFrame(per).groupby("metric").passed.mean()
    print(f"{t.detects.sum()}/{len(t)} metrics detect the {chain} drift; "
          f"metrics passing every image: {list(pi[pi == 1].index)}; best per-image pass rate {pi.max():.0%}")

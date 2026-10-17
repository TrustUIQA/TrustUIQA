"""NumPy port of Z. J. Wang's public MATLAB UCIQE (uwrepos/UCIQE: UCIQE.m + rgb2lab_n.m), line by line.

The MATLAB code computes saturation = chroma ./ L with no guard, so any pure-black pixel (L = 0, chroma = 0) gives
0/0 = NaN and the image's UCIQE becomes NaN (18-85% of the images in our sets, measured with Octave).
`omit_nan=True` is the obvious one-line patch (mean over the defined pixels, i.e. nanmean); everything else is
unchanged (L in 0-100, max-min luminance contrast, coefficients 0.4680 / 0.2745 / 0.2576).
--verify checks the port against the Octave scores of mos/uciqe_matlab.py (scratch/octave_scores/) on the first 200
images Octave scored as finite (12 Oct: max |diff| 5e-9, i.e. the 8-decimal print).
Writes uciqe_matlab_fix.csv (score_mos.py format) to mos/scores/<ds>/ or <pool>/scores_ext/.
Usage: python -I mos/uciqe_wang.py uid2021 saud ... [--pool] [--verify]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from mos.score_mos import save  # noqa: E402

MAT = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])


def rgb2lab_n(rgb):
    x = rgb.astype(np.float64)
    if x.max() > 1.0:
        x = x / 255
    X, Y, Z = np.moveaxis(x @ MAT.T, -1, 0)
    X, Z = X / 0.950456, Z / 1.088754
    T = 0.008856
    f = lambda t: np.where(t > T, np.cbrt(t), 7.787 * t + 16 / 116)  # noqa: E731
    L = np.where(Y > T, 116 * np.cbrt(Y) - 16.0, 903.3 * Y)
    return L, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))


def uciqe(rgb, omit_nan=False):
    L, a, b = rgb2lab_n(rgb)
    chroma = np.sqrt(a ** 2 + b ** 2)
    u_c = chroma.mean()
    sigma_c = np.sqrt(np.mean(chroma ** 2 - u_c ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = chroma / L
    u_s = np.nanmean(sat) if omit_nan else sat.mean()
    return 0.4680 * sigma_c + 0.2745 * (L.max() - L.min()) + 0.2576 * u_s


def selfcheck():
    im = np.full((4, 4, 3), 120, np.uint8)
    assert np.isfinite(uciqe(im))
    im[0, 0] = 0
    assert np.isnan(uciqe(im)) and np.isfinite(uciqe(im, omit_nan=True))


def items(ds, pool):
    if pool:
        paths = sorted((REPO / ds).glob("*/*/*.png"))
        return paths, [f"{p.parts[-3]}/{p.parts[-2]}/{int(p.stem)}" for p in paths], REPO / ds / "scores_ext"
    man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
    return [Path(p) for p in man.path], list(man.id), REPO / "mos/scores" / ds


if __name__ == "__main__":
    selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    for ds in a.ds:
        paths, ids, out = items(ds, a.pool)
        if a.verify:
            o = pd.read_csv(REPO / "scratch/octave_scores" / out.relative_to(REPO) / "uciqe_matlab.csv",
                            dtype={"image": str})
            if a.pool:
                o["id"] = o["mode"] + "/" + o.image + "/" + o.step.astype(str)
            o = o.set_index("id").score
            ok = [i for i, k in enumerate(ids) if np.isfinite(o[k])][:200]
            d = [abs(uciqe(cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB)) - o[ids[i]]) for i in ok]
            print(f"{ds}: port vs Octave on {len(ok)} finite images, max |diff| {max(d):.2e}; "
                  f"Octave NaN share {np.isnan(o).mean():.0%}", flush=True)
            continue
        dst = out / "uciqe_matlab_fix.csv"
        if dst.exists():
            continue
        s = [uciqe(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), omit_nan=True)
             for p in tqdm(paths, desc=ds, mininterval=30)]
        save(dst, ids, s, False, a.pool)
        print(f"done {ds} ({len(s)}), NaN left: {int(np.isnan(s).sum())}", flush=True)

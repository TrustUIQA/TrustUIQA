"""UIQM (Panetta et al. 2016) and UCIQE (Yang & Sowmya 2015). Input: RGB uint8/float in [0, 255].

Follows the widely copied FUnIE-GAN uqim_utils.py for UIQM, vectorised, with two fixes:
trimmed-mean slice off-by-one and UISM blue weight 0.114 (FUnIE has typo 0.144).
ponytail: UIQM/UCIQE implementations disagree across repos; cross-check one image against the
authors' MATLAB before quoting absolute numbers. Trends (score up/down per step) do not depend on it.
"""
import math

import cv2
import numpy as np
from scipy import ndimage


def _trimmed_mean(x, a=0.1):
    x = np.sort(x)
    k = x.size
    return x[math.ceil(a * k): k - math.floor(a * k)].mean()


def _uicm(x):
    r, g, b = (x[..., i].ravel() for i in range(3))
    rg, yb = r - g, (r + g) / 2 - b
    mrg, myb = _trimmed_mean(rg), _trimmed_mean(yb)
    var = ((rg - mrg) ** 2).mean() + ((yb - myb) ** 2).mean()
    return -0.0268 * math.hypot(mrg, myb) + 0.1586 * math.sqrt(var)


def _block_maxmin(x, w=10):
    k2, k1 = x.shape[0] // w, x.shape[1] // w
    b = x[: k2 * w, : k1 * w].reshape(k2, w, k1, w, -1)
    return b.max(axis=(1, 3, 4)), b.min(axis=(1, 3, 4)), k1 * k2


def _eme(ch):
    mx, mn, n = _block_maxmin(ch)
    ok = (mx > 0) & (mn > 0)
    return 2.0 / n * np.log(mx[ok] / mn[ok]).sum()


def _uism(x):
    total = 0.0
    for i, lam in enumerate((0.299, 0.587, 0.114)):
        c = x[..., i]
        mag = np.hypot(ndimage.sobel(c, 0), ndimage.sobel(c, 1))
        mag *= 255.0 / max(mag.max(), 1e-12)
        total += lam * _eme(mag * c)
    return total


def _uiconm(x):
    mx, mn, n = _block_maxmin(x)
    top, bot = mx - mn, mx + mn
    ok = (top > 0) & (bot > 0)
    r = top[ok] / bot[ok]
    return -1.0 / n * (r * np.log(r)).sum()


def uiqm(rgb):
    x = np.asarray(rgb, np.float64)
    return 0.0282 * _uicm(x) + 0.2953 * _uism(x) + 3.5753 * _uiconm(x)


def uciqe(rgb):
    lab = cv2.cvtColor(np.asarray(rgb, np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # L 0..100, a/b signed
    L = lab[..., 0] / 100.0
    chroma = np.hypot(lab[..., 1], lab[..., 2]) / 100.0
    lo, hi = np.percentile(L, [1, 99])
    sat = np.divide(chroma, L, out=np.zeros_like(chroma), where=L > 0)
    return 0.4680 * chroma.std() + 0.2745 * (hi - lo) + 0.2576 * sat.mean()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gray = np.full((64, 64, 3), 128.0)
    assert abs(_uicm(gray)) < 1e-9 and uciqe(gray) < 1e-3
    img = rng.uniform(40, 200, (64, 64, 3))
    img[..., 2] += 40  # blue cast
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 1)
    sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255
    assert uciqe(sat) > uciqe(np.clip(img, 0, 255)), "saturation must raise UCIQE"
    print("ok", round(uiqm(img), 3), round(uciqe(img), 3))

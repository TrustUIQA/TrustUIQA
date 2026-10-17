"""UCIQE — Underwater Colour Image Quality Evaluation.

Yang & Sowmya, "An underwater color image quality evaluation metric",
IEEE TIP 24(12):6062-6071, 2015.

    UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s     (CIELab)

Published UCIQE numbers are NOT comparable across papers because the public
ports disagree on Lab scaling, the luminance-contrast definition and the
saturation term. We therefore ship two variants and report both:

``variant="paper"`` (default; what we argue is correct)
    skimage CIELab (D65). L/100, chroma/100. con_l = mean of top 1% L minus
    mean of bottom 1% L. Saturation is the CIE definition C/sqrt(C^2+L^2),
    which is bounded in [0,1] (the original C/L explodes on dark pixels).

``variant="legacy_cv2"``
    Mirrors the most-copied Python port as we understand it: OpenCV 8-bit Lab
    divided by 255 *without removing the +128 a/b offset*, percentile-index
    contrast, saturation = C/L. C/L explodes on dark pixels, so values >1 are
    common. Diff it against the exact port you cite before quoting numbers.

Note: the paper's third coefficient is 0.2576 (some secondary sources print
0.2575; the difference is immaterial).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import rgb2lab

C1, C2, C3 = 0.4680, 0.2745, 0.2576


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {img.shape}")
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0 + 1e-6:
        raise ValueError("UCIQE expects [0,255] scale; got a float image in [0,1]")
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _uciqe_paper(rgb: np.ndarray) -> dict:
    lab = rgb2lab(rgb)  # L in [0,100], a/b roughly [-128,127]
    L = lab[..., 0] / 100.0
    C = np.hypot(lab[..., 1], lab[..., 2]) / 100.0
    sigma_c = float(C.std())
    Ls = np.sort(L.ravel())
    k = max(1, int(round(0.01 * Ls.size)))
    con_l = float(Ls[-k:].mean() - Ls[:k].mean())
    denom = np.sqrt(C ** 2 + L ** 2)
    sat = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
    mu_s = float(sat.mean())
    return {"sigma_c": sigma_c, "con_l": con_l, "mu_s": mu_s}


def _uciqe_legacy_cv2(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    L = lab[..., 0] / 255.0
    a = lab[..., 1] / 255.0
    b = lab[..., 2] / 255.0
    C = np.sqrt(a ** 2 + b ** 2)
    sigma_c = float(C.std())
    Lf = L.ravel()
    order = np.argsort(Lf)
    con_l = float(Lf[order[int(len(Lf) * 0.99)]] - Lf[order[int(len(Lf) * 0.01)]])
    Cf = C.ravel()
    sat = np.divide(Cf, Lf, out=np.zeros_like(Cf), where=Lf != 0)
    return {"sigma_c": sigma_c, "con_l": con_l, "mu_s": float(sat.mean())}


def uciqe(img: np.ndarray, variant: str = "paper", return_components: bool = False):
    rgb = _to_uint8_rgb(img)
    if variant == "paper":
        comp = _uciqe_paper(rgb)
    elif variant == "legacy_cv2":
        comp = _uciqe_legacy_cv2(rgb)
    else:
        raise ValueError(f"unknown UCIQE variant {variant!r}")
    score = C1 * comp["sigma_c"] + C2 * comp["con_l"] + C3 * comp["mu_s"]
    if return_components:
        return {"uciqe": score, **comp}
    return score

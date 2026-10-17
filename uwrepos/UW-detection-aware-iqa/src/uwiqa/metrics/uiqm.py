"""UIQM — Underwater Image Quality Measure.

Panetta, Gao & Agaian, "Human-visual-system-inspired underwater image quality
measures", IEEE J. Oceanic Eng. 41(3):541-551, 2016.

    UIQM = c1*UICM + c2*UISM + c3*UIConM,  c = (0.0282, 0.2953, 3.5753)

Implementation follows the de-facto reference Python port used by most UIE
papers (Islam et al., FUnIE-GAN repo, ``uqim_utils.py``) so that our numbers
are comparable with published tables. Deliberate deviations, all documented:

* UISM channel weights use the paper's (0.299, 0.587, 0.114). The reference
  port has a typo (0.144 for blue); pass ``legacy_blue_weight=True`` to
  reproduce it exactly.
* Block statistics are vectorised (identical maths, ~100x faster).

Input: HxWx3 RGB, uint8 or float in [0, 255]. Do NOT pass [0, 1] floats —
UICM is scale-dependent and you will get silently wrong numbers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

C1, C2, C3 = 0.0282, 0.2953, 3.5753


def _as_float_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {img.shape}")
    if img.dtype != np.uint8 and img.max() <= 1.0 + 1e-6 and img.size > 0:
        raise ValueError("UIQM expects [0,255] scale; got a float image in [0,1]")
    return img.astype(np.float64)


def _alpha_trimmed_mean(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    x = np.sort(x.ravel())
    k = x.size
    lo, hi = int(np.ceil(alpha_l * k)), int(np.floor(alpha_r * k))
    return float(x[lo:k - hi].mean())


def _alpha_trimmed_var(x: np.ndarray, mu: float) -> float:
    return float(np.mean((x.ravel() - mu) ** 2))


def uicm(img: np.ndarray) -> float:
    """Colourfulness measure on opponent channels RG and YB."""
    img = _as_float_rgb(img)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = (r + g) / 2.0 - b
    mu_rg, mu_yb = _alpha_trimmed_mean(rg), _alpha_trimmed_mean(yb)
    s_rg, s_yb = _alpha_trimmed_var(rg, mu_rg), _alpha_trimmed_var(yb, mu_yb)
    return -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(s_rg + s_yb)


def _blocks(x: np.ndarray, w: int) -> np.ndarray:
    """Crop to a multiple of w and reshape into (nby, nbx, w, w[, C]) blocks."""
    nby, nbx = x.shape[0] // w, x.shape[1] // w
    x = x[: nby * w, : nbx * w]
    if x.ndim == 2:
        return x.reshape(nby, w, nbx, w).swapaxes(1, 2)
    return x.reshape(nby, w, nbx, w, x.shape[2]).swapaxes(1, 2)


def _block_max_min(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    blk = _blocks(x, w)
    axes = tuple(range(2, blk.ndim))
    return blk.max(axis=axes), blk.min(axis=axes)


def _eme(x: np.ndarray, w: int) -> float:
    k1, k2 = x.shape[1] / w, x.shape[0] / w
    mx, mn = _block_max_min(x, w)
    ok = (mx > 0) & (mn > 0)
    return float(2.0 / (k1 * k2) * np.sum(np.log(mx[ok] / mn[ok])))


def _sobel_mag(ch: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(ch, axis=1)
    dy = ndimage.sobel(ch, axis=0)
    mag = np.hypot(dx, dy)
    m = mag.max()
    return mag * (255.0 / m) if m > 0 else mag


def uism(img: np.ndarray, window: int = 10, legacy_blue_weight: bool = False) -> float:
    """Sharpness measure: EME of Sobel-edge-weighted channels."""
    img = _as_float_rgb(img)
    wb = 0.144 if legacy_blue_weight else 0.114
    weights = (0.299, 0.587, wb)
    total = 0.0
    for c, lam in enumerate(weights):
        ch = img[..., c]
        edge = np.round(_sobel_mag(ch) * ch)
        total += lam * _eme(edge, window)
    return total


def uiconm(img: np.ndarray, window: int = 10) -> float:
    """Contrast measure: logAMEE over all channels per block (reference port)."""
    img = _as_float_rgb(img)
    k1, k2 = img.shape[1] / window, img.shape[0] / window
    mx, mn = _block_max_min(img, window)
    top, bot = mx - mn, mx + mn
    ok = (top > 0) & (bot > 0)
    ratio = top[ok] / bot[ok]
    return float(-1.0 / (k1 * k2) * np.sum(ratio * np.log(ratio)))


def uiqm(img: np.ndarray, window: int = 10, legacy_blue_weight: bool = False,
         return_components: bool = False):
    a = uicm(img)
    b = uism(img, window, legacy_blue_weight)
    c = uiconm(img, window)
    score = C1 * a + C2 * b + C3 * c
    if return_components:
        return {"uiqm": score, "uicm": a, "uism": b, "uiconm": c}
    return score

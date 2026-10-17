"""Classical / physics-based UIE baselines (CPU, no weights needed).

All functions: uint8 RGB in, uint8 RGB out.

* clahe        — CLAHE on the L channel of CIELab
* gray_world   — gray-world white balance
* udcp         — Underwater Dark Channel Prior, Drews Jr. et al., ICCVW 2013
* fusion       — Colour balance + multi-scale fusion, Ancuti et al., TIP 2018

These are faithful-but-compact re-implementations, not the authors' code.
Say so in the paper.
"""
from __future__ import annotations

import cv2
import numpy as np


def _f(img):  # uint8 -> float32 [0,1]
    return img.astype(np.float32) / 255.0


def _u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _gray_world(x: np.ndarray) -> np.ndarray:
    means = x.reshape(-1, 3).mean(0) + 1e-6
    return np.clip(x * (means.mean() / means), 0, 1)


def gray_world(img: np.ndarray) -> np.ndarray:
    return _u8(_gray_world(_f(img)))


# ---------------------------------------------------------------- UDCP
def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 40, eps: float = 1e-3):
    box = lambda a: cv2.boxFilter(a, -1, (r, r))
    mi, mp = box(guide), box(src)
    cov = box(guide * src) - mi * mp
    var = box(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(a) * guide + box(b)


def udcp(img: np.ndarray, patch: int = 15, omega: float = 0.95, t0: float = 0.1) -> np.ndarray:
    x = _f(img)
    kernel = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(x[..., 1:].min(axis=2), kernel)  # G,B only
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = x.reshape(-1, 3)[idx].mean(0)
    A = np.maximum(A, 1e-3)
    t = 1 - omega * cv2.erode((x[..., 1:] / A[1:]).min(axis=2), kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    t = np.clip(_guided_filter(gray, t.astype(np.float32)), t0, 1)[..., None]
    return _u8((x - A) / t + A)


# ---------------------------------------------------------------- Ancuti fusion
def _red_compensate(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    r, g = x[..., 0], x[..., 1]
    x = x.copy()
    x[..., 0] = np.clip(r + alpha * (g.mean() - r.mean()) * (1 - r) * g, 0, 1)
    return x


def _norm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-6)


def _weights(x: np.ndarray) -> np.ndarray:
    lum = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    w_lap = np.abs(cv2.Laplacian(lum, cv2.CV_32F))
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    blur = cv2.GaussianBlur(lab, (5, 5), 0)
    w_sal = np.linalg.norm(lab.reshape(-1, 3).mean(0) - blur, axis=2)
    w_sat = np.sqrt(((x - lum[..., None]) ** 2).mean(axis=2))
    return w_lap + _norm(w_sal) + w_sat


def _gauss_pyr(x, n):
    p = [x]
    for _ in range(n - 1):
        p.append(cv2.pyrDown(p[-1]))
    return p


def _lap_pyr(x, n):
    g = _gauss_pyr(x, n)
    out = [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(n - 1)]
    return out + [g[-1]]


def fusion(img: np.ndarray, gamma: float = 2.0, levels: int | None = None) -> np.ndarray:
    x = _gray_world(_red_compensate(_f(img))).astype(np.float32)
    in1 = np.power(x, gamma).astype(np.float32)                       # gamma-corrected
    blur = cv2.GaussianBlur(x, (0, 0), 2)
    in2 = ((x + _norm(x - blur)) / 2).astype(np.float32)               # sharpened
    w = np.stack([_weights(in1), _weights(in2)]) + 0.1
    w = (w / w.sum(0, keepdims=True)).astype(np.float32)
    if levels is None:
        levels = max(1, int(np.log2(min(img.shape[:2]))) - 4)
    out = None
    for inp, wk in zip((in1, in2), w):
        lp = _lap_pyr(inp, levels)
        gw = _gauss_pyr(wk, levels)
        fused = [l * g[..., None] for l, g in zip(lp, gw)]
        out = fused if out is None else [a + b for a, b in zip(out, fused)]
    res = out[-1]
    for lev in reversed(out[:-1]):
        res = cv2.pyrUp(res, dstsize=lev.shape[1::-1]) + lev
    return _u8(res)


CLASSICAL = {"clahe": clahe, "gray_world": gray_world, "udcp": udcp, "fusion": fusion}

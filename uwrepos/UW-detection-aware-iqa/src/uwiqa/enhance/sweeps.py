"""Controlled over-enhancement sweeps for the O2 stress test.

Each op maps (uint8 RGB, strength s >= 0) -> uint8 RGB, with s=0 the identity.
A metric is "gameable" if its score rises monotonically with s past the point
where the image is visibly over-processed.
"""
from __future__ import annotations

import cv2
import numpy as np


def saturation(img: np.ndarray, s: float) -> np.ndarray:
    """Scale HSV saturation by (1+s). Float HSV: the 8-bit path quantises hue."""
    hsv = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + s), 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, s: float) -> np.ndarray:
    """Linear contrast stretch about the per-image mean by factor (1+s)."""
    x = img.astype(np.float32)
    mu = x.mean(axis=(0, 1), keepdims=True)
    return np.clip((x - mu) * (1 + s) + mu, 0, 255).astype(np.uint8)


def red_shift(img: np.ndarray, s: float) -> np.ndarray:
    """Over-compensate the red channel (the classic UIQM-inflating artefact)."""
    x = img.astype(np.float32)
    x[..., 0] = x[..., 0] * (1 + s) + 40 * s
    return np.clip(x, 0, 255).astype(np.uint8)


def unsharp(img: np.ndarray, s: float) -> np.ndarray:
    """Unsharp-mask over-sharpening (halo artefacts)."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + s, blur, -s, 0)


SWEEPS = {"saturation": saturation, "contrast": contrast,
          "red_shift": red_shift, "unsharp": unsharp}
DEFAULT_STRENGTHS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def monotonic_inflation(strengths, scores) -> dict:
    """Spearman of score vs strength + fraction of increasing steps."""
    from scipy.stats import spearmanr
    s = np.asarray(scores, float)
    steps = np.diff(s)
    return {"srcc_vs_strength": float(spearmanr(strengths, s).statistic),
            "frac_increasing": float((steps > 0).mean()),
            "gain_at_max": float(s[-1] - s[0])}

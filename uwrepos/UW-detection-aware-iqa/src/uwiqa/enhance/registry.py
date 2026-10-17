"""Enhancer registry for utility labelling: name -> callable(uint8 RGB) -> uint8 RGB.

``raw`` is the identity (the baseline every delta is measured against).
``null_jpeg95`` is a JPEG q95 round-trip: a visually invisible change whose
delta distribution estimates the label noise floor (how much detection
scores move for reasons that have nothing to do with enhancement).
Deep enhancers register themselves here when their weights are available.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .classical import CLASSICAL


def _jpeg_roundtrip(img: np.ndarray, q: int = 95) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


ENHANCERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": lambda x: x,
    "null_jpeg95": _jpeg_roundtrip,
    **CLASSICAL,
}
CPU_ENHANCERS = list(ENHANCERS)  # all of the above are CPU-only and picklable via name lookup


def register(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    ENHANCERS[name] = fn


def resize_long_side(img: np.ndarray, long_side: int | None) -> np.ndarray:
    if not long_side or max(img.shape[:2]) <= long_side:
        return img
    s = long_side / max(img.shape[:2])
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)

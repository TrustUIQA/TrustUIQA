"""One interface over hand-crafted (UIQM/UCIQE) and pyiqa NR metrics.

Every metric exposes ``higher_better`` so the correlation harness can orient
scores without guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from .uciqe import uciqe
from .uiqm import uiqm

# Core O1 set. qalign (7B LMM) is excluded by default: too heavy for a T4.
PYIQA_DEFAULT = [
    "niqe", "brisque", "musiq", "clipiqa", "clipiqa+", "topiq_nr",
    "liqe", "liqe_mix", "maniqa", "tres", "arniqa", "qualiclip", "uranker",
]
HANDCRAFTED = {
    "uiqm": (lambda a: uiqm(a), True),
    "uciqe": (lambda a: uciqe(a, "paper"), True),
    "uciqe_legacy": (lambda a: uciqe(a, "legacy_cv2"), True),
}
DEFAULT_METRICS = list(HANDCRAFTED) + PYIQA_DEFAULT


def load_rgb(path, max_side: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BICUBIC)
    return np.asarray(img)


@dataclass
class Metric:
    name: str
    higher_better: bool
    fn: Callable  # (path) -> float


class MetricRunner:
    """Lazily instantiates metrics; pyiqa models live on ``device``."""

    def __init__(self, names=None, device: str | None = None, max_side: int | None = None):
        self.names = list(names or DEFAULT_METRICS)
        self.max_side = max_side
        self._device = device
        self._metrics: dict[str, Metric] = {}

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def get(self, name: str) -> Metric:
        if name in self._metrics:
            return self._metrics[name]
        if name in HANDCRAFTED:
            f, hb = HANDCRAFTED[name]
            m = Metric(name, hb, lambda p, f=f: float(f(load_rgb(p, self.max_side))))
        else:
            import pyiqa
            import torch
            model = pyiqa.create_metric(name, device=self.device, as_loss=False)

            def run(p, model=model):
                img = load_rgb(p, self.max_side)
                if min(img.shape[:2]) < 224:  # LIQE and others assert a short side >= 224
                    s = 224 / min(img.shape[:2])
                    img = np.asarray(Image.fromarray(img).resize(
                        (round(img.shape[1] * s), round(img.shape[0] * s)), Image.BICUBIC))
                t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float().div(255).unsqueeze(0)
                with torch.no_grad():
                    return float(model(t.to(self.device)).item())

            m = Metric(name, not bool(model.lower_better), run)
        self._metrics[name] = m
        return m

    def higher_better(self) -> dict[str, bool]:
        return {n: self.get(n).higher_better for n in self.names}

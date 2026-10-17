"""Dataset-level mAP of per-image enhancement policies, from saved detections.

A policy picks one variant (``raw`` or an enhancer) per image; the dataset
mAP is then computed over the chosen variants' detections. Matching
(prediction -> GT, per class and IoU threshold) is computed once per
(image, variant) and cached, so evaluating many policies and bootstrap
resamples is cheap.
"""
from __future__ import annotations

import glob

import numpy as np

from .metrics import IOU_THRS, Dets, _ap_from_tp, _match


class MatchCache:
    def __init__(self, npz_glob: str, gts: dict, methods=None):
        """gts: image -> Dets (already class-filtered). Loads every (image, method) in the npz shards."""
        self.gts = gts
        self.classes = np.unique(np.concatenate([g.cls for g in gts.values()]))
        self.m = {}
        for f in sorted(glob.glob(npz_glob)):
            z = np.load(f)
            for k in z.files:
                img, meth = k.rsplit("|", 1)
                if img not in gts or (methods is not None and meth not in methods):
                    continue
                a = z[k]
                p = Dets(a[:, :4], a[:, 5].astype(int), a[:, 4])
                g = gts[img]
                self.m[(img, meth)] = {c: _match(p, g, c)[:2] for c in self.classes}
        self.n_gt = {img: {c: int((g.cls == c).sum()) for c in self.classes} for img, g in gts.items()}

    def methods(self):
        return sorted({m for _, m in self.m})

    def map(self, images, choice) -> dict:
        """images: list; choice: list of method names aligned with images."""
        aps = []
        for c in self.classes:
            n = sum(self.n_gt[i][c] for i in images)
            if n == 0:
                continue
            tps, confs = zip(*[self.m[(i, m)][c] for i, m in zip(images, choice)])
            conf = np.concatenate(confs)
            order = np.argsort(-conf, kind="stable")
            tp = np.concatenate(tps, axis=1)[:, order]
            aps.append([_ap_from_tp(tp[t], n) for t in range(len(IOU_THRS))])
        a = np.array(aps)
        return {"map50": float(a[:, 0].mean()), "map": float(a.mean())}

    def bootstrap_diff(self, images, choice_a, choice_b, n_boot=200, seed=0, key="map"):
        """Paired bootstrap over images of mAP(A) - mAP(B): mean, 95% CI, P(diff <= 0)."""
        rng = np.random.default_rng(seed)
        images, ca, cb = np.asarray(images), np.asarray(choice_a), np.asarray(choice_b)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(images), len(images))
            d.append(self.map(images[i], ca[i])[key] - self.map(images[i], cb[i])[key])
        d = np.array(d)
        return {"diff_mean": float(d.mean()), "lo": float(np.quantile(d, 0.025)),
                "hi": float(np.quantile(d, 0.975)), "p_le_0": float((d <= 0).mean())}

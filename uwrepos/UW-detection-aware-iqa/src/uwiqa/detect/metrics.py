"""Detection scoring used for utility labels (per image) and for policy
evaluation (dataset level).

Boxes are normalised xyxy in [0,1], so scores are independent of the working
resolution. AP follows COCO conventions: greedy matching by descending
confidence, 101-point interpolated precision, IoU thresholds .50:.05:.95,
averaged over classes that have ground truth (per image: classes present in
that image). Predictions of classes absent from the ground truth do not
lower AP (as in COCO) but do lower ``f1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_PTS = np.linspace(0, 1, 101)


@dataclass
class Dets:
    boxes: np.ndarray            # (N,4) xyxy normalised
    cls: np.ndarray              # (N,) int
    conf: np.ndarray | None = None  # (N,) float; None for ground truth

    def __len__(self):
        return len(self.cls)


def yolo_labels_to_gt(txt: str) -> Dets:
    rows = [list(map(float, ln.split()[:5])) for ln in txt.splitlines() if ln.strip()]
    if not rows:
        return Dets(np.zeros((0, 4)), np.zeros(0, int))
    a = np.array(rows)
    cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return Dets(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), a[:, 0].astype(int))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]).clip(0) * (x[:, 3] - x[:, 1]).clip(0)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def _match(pred: Dets, gt: Dets, c: int, thr_list=IOU_THRS):
    """For class c: TP flags (len(thr), n_pred_c) in confidence order, confs, n_gt."""
    pm, gm = pred.cls == c, gt.cls == c
    order = np.argsort(-pred.conf[pm], kind="stable")
    pb, conf = pred.boxes[pm][order], pred.conf[pm][order]
    gb = gt.boxes[gm]
    iou = box_iou(pb, gb)
    tp = np.zeros((len(thr_list), len(pb)), bool)
    for t, thr in enumerate(thr_list):
        used = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            if len(gb) == 0:
                break
            cand = np.where(~used, iou[i], -1)
            j = int(cand.argmax())
            if cand[j] >= thr:
                used[j] = True
                tp[t, i] = True
    return tp, conf, int(gm.sum())


def _ap_from_tp(tp_sorted: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP from TP flags already sorted by confidence."""
    if n_gt == 0:
        return np.nan
    if len(tp_sorted) == 0:
        return 0.0
    ctp = np.cumsum(tp_sorted)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp_sorted) + 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # monotone envelope
    idx = np.searchsorted(rec, REC_PTS, side="left")
    return float(np.mean(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)))


def image_scores(pred: Dets, gt: Dets, f1_conf: float = 0.25) -> dict:
    """Per-image AP50, AP50:95 (mean over GT classes) and F1 at a conf threshold."""
    classes = np.unique(gt.cls)
    ap = np.full((len(classes), len(IOU_THRS)), np.nan)
    for k, c in enumerate(classes):
        tp, _, n = _match(pred, gt, c)
        ap[k] = [_ap_from_tp(tp[t], n) for t in range(len(IOU_THRS))]
    keep = pred.conf >= f1_conf
    p = Dets(pred.boxes[keep], pred.cls[keep], pred.conf[keep])
    tp50 = sum(_match(p, gt, c, [0.5])[0].sum() for c in np.unique(np.r_[gt.cls, p.cls]))
    fp, fn = len(p) - tp50, len(gt) - tp50
    f1 = 2 * tp50 / (2 * tp50 + fp + fn) if (tp50 + fp + fn) else 1.0
    return {"ap50": float(np.nanmean(ap[:, 0])) if len(classes) else np.nan,
            "ap": float(np.nanmean(ap)) if len(classes) else np.nan,
            "f1": float(f1), "n_pred": int(len(p)), "n_gt": int(len(gt))}


def dataset_map(preds: list[Dets], gts: list[Dets], classes=None) -> dict:
    """COCO-style dataset mAP over many images (for selective-enhancement policies)."""
    classes = np.unique(np.concatenate([g.cls for g in gts])) if classes is None else np.asarray(classes)
    per_class = {}
    for c in classes:
        tps, confs, n_gt = [], [], 0
        for p, g in zip(preds, gts):
            tp, conf, n = _match(p, g, c)
            tps.append(tp)
            confs.append(conf)
            n_gt += n
        if n_gt == 0:
            continue
        conf = np.concatenate(confs)
        order = np.argsort(-conf, kind="stable")
        tp = np.concatenate(tps, axis=1)[:, order]
        per_class[int(c)] = [_ap_from_tp(tp[t], n_gt) for t in range(len(IOU_THRS))]
    a = np.array(list(per_class.values()))
    return {"map50": float(a[:, 0].mean()), "map": float(a.mean()),
            "per_class_map": {c: float(np.mean(v)) for c, v in per_class.items()}}

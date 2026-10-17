"""O3: per-image detection utility of every enhancer, with the frozen raw detector.

For each image: decode -> resize to --work-size (long side) -> apply every
enhancer in memory -> detect with the frozen YOLO -> score against GT.
Enhanced images are never written to disk (Kaggle's 20 GB output limit).

Outputs in RESULTS_ROOT/utility/<set>/<tag>/ (resumable, one shard per --shard images):
  scores_XXXX.csv   image, method, ap50, ap, f1, n_pred, n_gt
  preds_XXXX.npz    every detection (conf >= --conf) for every (image, method),
                    so dataset-level mAP of any selection policy can be
                    computed offline without re-running the detector.

    python scripts/build_utility_labels.py --set ruod_pred_test --limit 50   # smoke test
    python scripts/build_utility_labels.py --set ruod_pred                   # pred_train+val+test
    python scripts/build_utility_labels.py --set duo_clean
    python scripts/build_utility_labels.py --set ruod_pred --methods raw --deep funiegan --tag deep
"""
import argparse
import glob
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root
from uwiqa.detect.metrics import Dets, dataset_map, image_scores, yolo_labels_to_gt
from uwiqa.enhance.registry import CPU_ENHANCERS, ENHANCERS, resize_long_side

SETS = {
    "ruod_pred": ("ruod", "ruod_splits.csv", ["pred_train", "pred_val", "pred_test"]),
    "ruod_pred_test": ("ruod", "ruod_splits.csv", ["pred_test"]),
    "duo_clean": ("duo", "duo_clean.csv", None),
}


def load_set(name):
    ds, csv, splits = SETS[name]
    df = pd.read_csv(RESULTS_ROOT / "splits" / csv)
    if splits:
        df = df[df["split"].isin(splits)]
    root = dataset_root(ds)
    return df.reset_index(drop=True), root


def label_path(root, rel):
    parts = rel.split("/")
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def _keep_numpy(sample):
    """Module-level (picklable) collate that stops DataLoader converting arrays to tensors."""
    return sample


class Variants(Dataset):
    """Worker processes decode + resize + run the CPU enhancers in parallel."""

    def __init__(self, rows, root, methods, work_size):
        self.rows, self.root, self.methods, self.work = rows, root, methods, work_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel = self.rows[i]
        im = Image.open(self.root / rel)
        w, h = im.size
        s = self.work / max(w, h)
        if s < 1:  # fast DCT-domain JPEG downscale to >= target; exact resize below
            im.draft("RGB", (int(w * s) + 1, int(h * s) + 1))
        x = resize_long_side(np.asarray(im.convert("RGB")), self.work)
        return rel, {m: ENHANCERS[m](x) for m in self.methods}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--weights", default=str(RESULTS_ROOT / "detector" / "raw_yolo11s" / "weights" / "best.pt"))
    ap.add_argument("--methods", nargs="*", default=CPU_ENHANCERS)
    ap.add_argument("--work-size", type=int, default=1280)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", nargs="*", type=int, default=None,
                    help="restrict detections, e.g. 0 1 2 3 for DUO")
    ap.add_argument("--deep", nargs="*", default=[], help="GPU enhancers from uwiqa.enhance.deep, e.g. funiegan")
    ap.add_argument("--tag", default="main", help="output subfolder; label method groups in separate runs")
    args = ap.parse_args()
    if "raw" not in args.methods:
        args.methods = ["raw"] + args.methods

    from ultralytics import YOLO
    det = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    from uwiqa.enhance.deep import DEEP
    gpu_enh = {n: DEEP[n]("cuda" if device == 0 else "cpu") for n in args.deep}
    classes = args.classes if args.classes is not None else ([0, 1, 2, 3] if args.set == "duo_clean" else None)

    df, root = load_set(args.set)
    if args.limit:
        df = df.head(args.limit)
    out = RESULTS_ROOT / "utility" / args.set / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.set}: {len(df)} images x {len(args.methods) + len(gpu_enh)} methods "
          f"{args.methods + list(gpu_enh)} -> {out}")

    t0, n_done = time.time(), 0
    for s in range(0, len(df), args.shard):
        tag = f"{s // args.shard:04d}"
        if (out / f"scores_{tag}.csv").exists():
            continue
        rows = df["image"].iloc[s:s + args.shard].tolist()
        dl = DataLoader(Variants(rows, root, args.methods, args.work_size), batch_size=None,
                        collate_fn=_keep_numpy, num_workers=args.workers, prefetch_factor=2 if args.workers else None)
        recs, pred_store = [], {}
        for rel, variants in dl:
            gt = yolo_labels_to_gt(label_path(root, rel).read_text())
            if classes is not None:
                keep = np.isin(gt.cls, classes)
                gt = Dets(gt.boxes[keep], gt.cls[keep])
            for n, f in gpu_enh.items():  # GPU enhancers run here, not in CPU workers
                variants[n] = f(variants["raw"])
            names = list(variants)
            bgr = [np.ascontiguousarray(v[:, :, ::-1]) for v in variants.values()]
            res = det.predict(bgr, imgsz=args.imgsz, conf=args.conf,
                              max_det=300, device=device, verbose=False, classes=classes)
            for m, r in zip(names, res):
                b = r.boxes
                p = Dets(b.xyxyn.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())
                recs.append({"image": rel, "method": m, **image_scores(p, gt)})
                pred_store[f"{rel}|{m}"] = np.c_[p.boxes, p.conf, p.cls].astype(np.float32)
        np.savez_compressed(out / f"preds_{tag}.npz", **pred_store)
        pd.DataFrame(recs).to_csv(out / f"scores_{tag}.csv", index=False)  # written last = shard done
        n_done += len(rows)
        el = time.time() - t0
        print(f"shard {tag}: {s + len(rows)}/{len(df)} images, {el / n_done:.2f}s/img, "
              f"ETA {(len(df) - s - len(rows)) * el / n_done / 60:.0f} min", flush=True)

    sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "scores_*.csv")))])
    sc.to_csv(out / "scores.csv", index=False)
    raw = sc[sc.method == "raw"].set_index("image")
    print(f"\n{sc.image.nunique()} images labelled. Mean per-image score and delta vs raw:")
    rows = []
    for m, g in sc.groupby("method"):
        d = g.set_index("image")[["ap", "ap50", "f1"]] - raw[["ap", "ap50", "f1"]]
        rows.append({"method": m, "ap": g.ap.mean(), "d_ap": d.ap.mean(),
                     "helps(d_ap>0)": (d.ap > 0).mean(), "hurts(d_ap<0)": (d.ap < 0).mean(),
                     "d_ap_std": d.ap.std(), "d_f1": d.f1.mean()})
    print(pd.DataFrame(rows).set_index("method").round(4).to_string())

    # Sanity check: our dataset mAP for raw should be close to Ultralytics' val mAP.
    preds, gts = [], []
    for f in sorted(glob.glob(str(out / "preds_*.npz"))):
        z = np.load(f)
        for k in z.files:
            rel, m = k.rsplit("|", 1)
            if m != "raw":
                continue
            a = z[k]
            preds.append(Dets(a[:, :4], a[:, 5].astype(int), a[:, 4]))
            gt = yolo_labels_to_gt(label_path(root, rel).read_text())
            if classes is not None:
                keep = np.isin(gt.cls, classes)
                gt = Dets(gt.boxes[keep], gt.cls[keep])
            gts.append(gt)
    r = dataset_map(preds, gts)
    print(f"\nraw dataset mAP50={r['map50']:.4f} mAP50-95={r['map']:.4f} "
          f"(compare with the Ultralytics eval at native resolution)")


if __name__ == "__main__":
    main()

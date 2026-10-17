"""O4 inputs: frozen-backbone embeddings of each RAW image, plus NR-IQA scores of
raw and every enhanced variant (for the IQA baselines).

Enhancement is done exactly as in build_utility_labels.py (1280 px), then
variants are downsized to --iqa-size for the IQA metrics.

Outputs in RESULTS_ROOT/features/<set>/ (sharded, resumable):
  emb_XXXX.npz   images + one (n, d) array per backbone
  iqa_XXXX.csv   image, method, metric, score

    python scripts/extract_features.py --set ruod_pred --deep funiegan
    python scripts/extract_features.py --set duo_clean --deep funiegan
"""
import argparse
import glob
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from uwiqa import RESULTS_ROOT
from uwiqa.enhance.registry import CPU_ENHANCERS, ENHANCERS, resize_long_side
from uwiqa.metrics import uciqe, uiqm

import build_utility_labels as bul  # same set definitions and image loading

HANDCRAFTED = {"uiqm": uiqm, "uciqe": uciqe}
MIN_SIDE = 224


def _keep(sample):
    return sample


class Item(Dataset):
    def __init__(self, rows, root, methods, work, iqa_size, hand):
        self.rows, self.root, self.methods = rows, root, methods
        self.work, self.iqa_size, self.hand = work, iqa_size, hand

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel = self.rows[i]
        im = Image.open(self.root / rel)
        w, h = im.size
        s = self.work / max(w, h)
        if s < 1:
            im.draft("RGB", (int(w * s) + 1, int(h * s) + 1))
        x = resize_long_side(np.asarray(im.convert("RGB")), self.work)
        small = {m: resize_long_side(ENHANCERS[m](x), self.iqa_size) for m in self.methods}
        hand = [(m, k, float(f(v))) for m, v in small.items() for k, f in self.hand.items()]
        return rel, x, small, hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(bul.SETS))
    ap.add_argument("--backbones", nargs="*", default=["clip_b32", "dinov2_s14", "resnet18"])
    ap.add_argument("--methods", nargs="*", default=CPU_ENHANCERS)
    ap.add_argument("--deep", nargs="*", default=[])
    ap.add_argument("--iqa", nargs="*", default=["uiqm", "uciqe", "topiq_nr", "liqe", "uranker"])
    ap.add_argument("--work-size", type=int, default=1280)
    ap.add_argument("--iqa-size", type=int, default=512)
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from uwiqa.enhance.deep import DEEP
    from uwiqa.model.features import Backbone
    gpu_enh = {n: DEEP[n](dev) for n in args.deep}
    bbs = {n: Backbone(n, dev) for n in args.backbones}
    hand = {k: HANDCRAFTED[k] for k in args.iqa if k in HANDCRAFTED}
    deep_iqa = {}
    for k in [k for k in args.iqa if k not in HANDCRAFTED]:
        import pyiqa
        m = pyiqa.create_metric(k, device=dev, as_loss=False)
        deep_iqa[k] = m

    df, root = bul.load_set(args.set)
    if args.limit:
        df = df.head(args.limit)
    out = RESULTS_ROOT / "features" / args.set
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.set}: {len(df)} images; backbones {list(bbs)}; IQA {args.iqa} on "
          f"{args.methods + list(gpu_enh)}")

    t0, n_done, n_upsampled = time.time(), 0, 0
    for s in range(0, len(df), args.shard):
        tag = f"{s // args.shard:04d}"
        if (out / f"iqa_{tag}.csv").exists():
            continue
        rows = df["image"].iloc[s:s + args.shard].tolist()
        dl = DataLoader(Item(rows, root, args.methods, args.work_size, args.iqa_size, hand),
                        batch_size=None, collate_fn=_keep, num_workers=args.workers,
                        prefetch_factor=2 if args.workers else None)
        embs = {n: [] for n in bbs}
        recs = []
        for rel, x, small, hand_rows in dl:
            for n, f in gpu_enh.items():
                small[n] = resize_long_side(f(x), args.iqa_size)
                hand_rows += [(n, k, float(fn(small[n]))) for k, fn in hand.items()]
            for n, bb in bbs.items():
                embs[n].append(bb([x])[0])
            recs += [(rel, m, k, v) for m, k, v in hand_rows]
            if deep_iqa:
                names = list(small)
                # all variants share one size (same source, same resize), so score them as one batch;
                # LIQE (and some other pyiqa nets) need a short side >= 224, so upsample tiny/wide images
                arr = np.stack([small[m] for m in names])
                if min(arr.shape[1:3]) < MIN_SIDE:
                    scale = MIN_SIDE / min(arr.shape[1:3])  # not `s`: that is the shard start index
                    size = (round(arr.shape[2] * scale), round(arr.shape[1] * scale))
                    arr = np.stack([cv2.resize(a, size, interpolation=cv2.INTER_CUBIC) for a in arr])
                    n_upsampled += 1
                batch = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255).to(dev)
                with torch.no_grad():
                    for k, metric in deep_iqa.items():
                        try:
                            scores = metric(batch).flatten().float().cpu().numpy()
                        except Exception as e:  # never lose a multi-hour run to one odd image
                            print(f"  ! {k} failed on {rel}: {type(e).__name__}: {e}", flush=True)
                            scores = np.full(len(names), np.nan)
                        recs += [(rel, m, k, float(v)) for m, v in zip(names, scores)]
        np.savez_compressed(out / f"emb_{tag}.npz", images=np.array(rows),
                            **{n: np.stack(v).astype(np.float32) for n, v in embs.items()})
        pd.DataFrame(recs, columns=["image", "method", "metric", "score"]).to_csv(
            out / f"iqa_{tag}.csv", index=False)  # written last = shard done
        n_done += len(rows)
        el = time.time() - t0
        print(f"shard {tag}: {s + len(rows)}/{len(df)}, {el / n_done:.2f}s/img, "
              f"ETA {(len(df) - s - len(rows)) * el / n_done / 60:.0f} min"
              + (f" ({n_upsampled} small images upsampled to {MIN_SIDE}px for IQA)" if n_upsampled else ""),
              flush=True)

    iqa = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(out / "iqa_*.csv")))])
    iqa.to_csv(out / "iqa.csv", index=False)
    print(f"done: {iqa.image.nunique()} images; mean IQA score per method:")
    print(iqa.pivot_table(index="method", columns="metric", values="score").round(3).to_string())


if __name__ == "__main__":
    main()

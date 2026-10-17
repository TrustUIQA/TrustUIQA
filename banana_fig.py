"""Nano Banana chains: a montage figure and a full-reference drift check against the original photo.

The NR drift table (banana_drift.py) only says what metrics think; there are no human labels on these images.
This measures how far each step has moved from the input: PSNR and DISTS of step k vs step 0, with step 0
resized to the generated size (outputs are 1200x896, the input 512 px). A rising DISTS / falling PSNR means
the chain is drifting away from the photo, whatever the NR metrics say.
Writes out_banana/fr_drift_<chain>.csv (mean per step) and out_banana/montage_<chain>.jpg.
Usage: python banana_fig.py enhance|replica [--images 001 021 036]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch

OUT = Path("out_banana")


def load(p, size=None):
    x = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    if size:
        x = cv2.resize(x, size, interpolation=cv2.INTER_CUBIC)
    return x


def tensor(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255)


def montage(d, images, steps, h=200):
    """Rows = images, columns = steps, every tile h px high (step 0 is narrower: 512 px input vs 1200x896)."""
    rows, widths = [], None
    for i in images:
        ims = [load(d / i / f"{s:02d}.png") for s in steps]
        ims = [cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_AREA) for x in ims]
        widths = widths or [x.shape[1] + 4 for x in ims]
        rows.append(np.concatenate([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in ims], 1))
    w = max(r.shape[1] for r in rows)
    head = np.full((28, w, 3), 255, np.uint8)
    for s, x in zip(steps, np.cumsum([0] + widths)):
        cv2.putText(head, f"step {s}", (int(x) + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([head] + [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", choices=["enhance", "replica", "nb2_enhance", "nb2_replica"])
    ap.add_argument("--images", nargs="+", default=["001", "021", "036", "047"])
    a = ap.parse_args()
    d = OUT / (a.chain if a.chain.startswith("nb2_") else f"nbpro_{a.chain}")
    last = max(int(p.stem) for p in (d / "001").glob("*.png"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dists = pyiqa.create_metric("dists", device=dev)
    rows = []
    for img in sorted(p.name for p in d.iterdir()):
        for s in range(1, last + 1):
            y = load(d / img / f"{s:02d}.png")
            x0 = load(d / img / "00.png", (y.shape[1], y.shape[0]))
            with torch.no_grad():
                dd = dists(tensor(y).to(dev), tensor(x0).to(dev)).item()
            rows.append(dict(image=img, step=s, psnr=cv2.PSNR(x0, y), dists=dd))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"fr_drift_{a.chain}_all.csv", index=False)
    m = t.groupby("step")[["psnr", "dists"]].mean()
    m.to_csv(OUT / f"fr_drift_{a.chain}.csv")
    print(m.iloc[[0, 4, 9, 19] + ([49] if last >= 50 else [])].round(3).to_string())

    steps = [0, 1, 5, 10, 20] if last < 50 else [0, 1, 10, 25, 50]
    cv2.imwrite(str(OUT / f"montage_{a.chain}.jpg"), cv2.cvtColor(montage(d, a.images, steps), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {OUT}/montage_{a.chain}.jpg, {OUT}/fr_drift_{a.chain}.csv")

"""Grid of raw vs every enhancer on a few random held-out images (sanity check).

    %run scripts/show_enhancers.py --n 3 --deep funiegan
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root
from uwiqa.enhance.deep import DEEP
from uwiqa.enhance.registry import CPU_ENHANCERS, ENHANCERS, resize_long_side

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--deep", nargs="*", default=[])
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--work-size", type=int, default=1280)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
fns = {m: ENHANCERS[m] for m in CPU_ENHANCERS if m != "null_jpeg95"}
fns.update({n: DEEP[n](dev) for n in args.deep})
sp = pd.read_csv(RESULTS_ROOT / "splits" / "ruod_splits.csv")
rows = sp[sp.split == "pred_test"].sample(args.n, random_state=args.seed).image
root = dataset_root("ruod")

fig, ax = plt.subplots(args.n, len(fns), figsize=(2.6 * len(fns), 1.8 * args.n), squeeze=False)
for r, rel in enumerate(rows):
    x = resize_long_side(np.asarray(Image.open(root / rel).convert("RGB")), args.work_size)
    for c, (name, f) in enumerate(fns.items()):
        ax[r, c].imshow(f(x))
        ax[r, c].set_title(name if r == 0 else "", fontsize=9)
        ax[r, c].axis("off")
plt.tight_layout()
plt.show()

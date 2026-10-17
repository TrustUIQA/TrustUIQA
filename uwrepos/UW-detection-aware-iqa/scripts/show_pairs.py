"""Show duplicate pairs in a Hamming-distance band, to choose the threshold.

    %run scripts/show_pairs.py --min-dist 26 --max-dist 32 --n 8     # in a notebook
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root

ap = argparse.ArgumentParser()
ap.add_argument("--min-dist", type=int, default=0)
ap.add_argument("--max-dist", type=int, default=256)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--leaky-only", action="store_true", help="only pairs crossing a split/dataset")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

pr = pd.read_csv(RESULTS_ROOT / "dedup" / "pairs.csv")
pr = pr[(pr.dist >= args.min_dist) & (pr.dist <= args.max_dist)]
if args.leaky_only:
    pr = pr[(pr.dataset_a != pr.dataset_b) | (pr.split_a != pr.split_b)]
print(f"{len(pr)} pairs with {args.min_dist} <= dist <= {args.max_dist}")
show = pr.sample(min(args.n, len(pr)), random_state=args.seed).sort_values("dist")
if len(show):
    fig, ax = plt.subplots(len(show), 2, figsize=(8, 2.4 * len(show)), squeeze=False)
    for r, (_, row) in enumerate(show.iterrows()):
        for c, side in enumerate("ab"):
            ds = row[f"dataset_{side}"]
            im = Image.open(os.path.join(dataset_root(ds), row[f"image_{side}"]))
            im.draft("RGB", (640, 640))
            ax[r, c].imshow(im.convert("RGB"))
            ax[r, c].set_title(f"{ds}/{row[f'split_{side}']}  d={row.dist}", fontsize=8)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.show()

"""Find near-duplicate images within and across detection datasets.

Outputs (RESULTS_ROOT/dedup/):
  hashes_<ds>.npy        cached hashes (resumable across sessions)
  pairs.csv              every near-duplicate pair with dataset/split of both sides
  groups.csv             image -> group id (use for leakage-safe splits)
and prints how many pairs cross a split or dataset boundary.

    python scripts/find_duplicates.py --datasets ruod duo --max-dist 20
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root, load_manifest
from uwiqa.data.dedup import connected_groups, dhash, hamming_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ruod", "duo"])
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--max-dist", type=int, default=20, help="of hash_size**2 bits (20/256 ~ 8%)")
    args = ap.parse_args()

    out = RESULTS_ROOT / "dedup"
    out.mkdir(parents=True, exist_ok=True)
    frames, hashes = [], []
    for ds in args.datasets:
        m = load_manifest(ds)[["image", "split"]].assign(dataset=ds)
        cache = out / f"hashes_{ds}_{args.hash_size}.npy"
        if cache.exists():
            h = np.load(cache)
        else:
            root = dataset_root(ds)
            h = np.stack([dhash(root / p, args.hash_size) for p in tqdm(m.image, desc=f"hash {ds}")])
            np.save(cache, h)
        frames.append(m)
        hashes.append(h)
    df = pd.concat(frames, ignore_index=True)
    H = np.concatenate(hashes)

    # Near-uniform images hash to almost all-0 and match everything; flag rather than link them.
    ones = np.unpackbits(H, axis=1).sum(1)
    flat = ones < 0.1 * args.hash_size ** 2
    print(f"{flat.sum()} near-uniform images excluded from matching")

    idx = np.nonzero(~flat)[0]
    p = hamming_pairs(H[idx], args.max_dist)
    pairs = np.column_stack([idx[p[:, 0]], idx[p[:, 1]], p[:, 2]]) if len(p) else np.zeros((0, 3), int)

    a, b = df.iloc[pairs[:, 0]].reset_index(drop=True), df.iloc[pairs[:, 1]].reset_index(drop=True)
    pr = pd.concat([a.add_suffix("_a"), b.add_suffix("_b")], axis=1).assign(dist=pairs[:, 2])
    pr.to_csv(out / "pairs.csv", index=False)

    df["group"] = connected_groups(len(df), pairs)
    df.to_csv(out / "groups.csv", index=False)

    print(f"{len(pr)} near-duplicate pairs (dist <= {args.max_dist}/{args.hash_size ** 2})")
    if len(pr):
        same_ds = pr.dataset_a == pr.dataset_b
        print("  across datasets:", int((~same_ds).sum()))
        for ds in args.datasets:
            s = pr[same_ds & (pr.dataset_a == ds)]
            print(f"  within {ds}: {len(s)}  (crossing a split: {int((s.split_a != s.split_b).sum())})")
        print("  distance histogram:", np.bincount(pr.dist, minlength=args.max_dist + 1).tolist())
    sizes = df.group.value_counts()
    print(f"{len(sizes)} groups; {int((sizes > 1).sum())} with >1 image; largest {sizes.max()}")


if __name__ == "__main__":
    main()

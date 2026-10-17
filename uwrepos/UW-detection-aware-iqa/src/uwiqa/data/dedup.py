"""Near-duplicate detection with difference hashes (dHash).

Why: RUOD and DUO come from the same lab and share source footage (identical
720x405 / 1920x1080 / 3840x2160 resolutions), and both contain video frames.
Duplicates across RUOD train/test, or between RUOD and DUO, would leak into
the detector and inflate the "cross-dataset" result. We hash every image,
link pairs within a Hamming threshold, and assign connected-component group
ids that the split code keeps on one side of every split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def dhash(path: str | Path, hash_size: int = 16) -> np.ndarray:
    """hash_size**2-bit difference hash, returned as packed uint8 bytes."""
    im = Image.open(path)
    im.draft("L", (hash_size * 8, hash_size * 8))  # fast JPEG DCT-domain downscale
    g = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR), np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1])


def hamming_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 512) -> np.ndarray:
    """All (i, j, dist) with i < j and Hamming distance <= max_dist. Brute force, chunked."""
    n = len(hashes)
    out = []
    for s in range(0, n, chunk):
        a = hashes[s:s + chunk]
        d = _POP8[a[:, None, :] ^ hashes[None, :, :]].sum(-1, dtype=np.int32)
        i, j = np.nonzero(d <= max_dist)
        keep = j > i + s
        out.append(np.stack([i[keep] + s, j[keep], d[i[keep], j[keep]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def connected_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find over duplicate pairs -> group id per item (singletons keep their own)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])

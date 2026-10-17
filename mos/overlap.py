"""Do the MOS datasets (and UIEB) share source photos? Needed before any train-on-A / test-on-B claim.

One image per scene (the raw photo where the set has one, else its first enhancement). Enhancements change colour,
not structure, so the signature is the z-scored Laplacian of a 48x48 grey thumbnail, computed both squashed and
centre-cropped (sets resize/crop differently). Every cross-set pair with signature correlation >= 0.6 is then
verified with ORB matches + RANSAC homography; >= 25 inliers = same photo.
Writes mos/results/overlap.csv (verified pairs) and prints counts. Usage: python -I mos/overlap.py
"""
import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def reps():
    out = {}
    for ds in ("uid2021", "saud", "saud2", "sota"):
        m = pd.read_csv(REPO / f"mos/manifests/{ds}.csv")
        m["scene"] = m.scene.astype(str)
        if ds == "uid2021":
            m = m.sort_values("method", key=lambda s: s != "Original")
        out[ds] = m.groupby("scene").path.first().to_dict()
    e = pd.read_csv(REPO / "mos/manifests/saud_edits.csv")  # SAUD raws (Raw/ folder) are in the edit manifest
    out["saud"] = e[e.method == "raw"].set_index(e[e.method == "raw"].scene.astype(str)).path.to_dict()
    out["uieb"] = {p.stem: str(p) for p in sorted((REPO / "data/uieb/raw").glob("*"))}
    return out


def gray(p, side=512):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    s = side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g


def sig(g):
    h, w = g.shape
    k = min(h, w)
    out = []
    for x in (g, g[(h - k) // 2:(h + k) // 2, (w - k) // 2:(w + k) // 2]):
        v = cv2.Laplacian(cv2.resize(x, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32), cv2.CV_32F)
        v = np.abs(v).ravel()
        out.append((v - v.mean()) / (v.std() + 1e-6))
    return np.stack(out)  # 2 x 2304


def inliers(ka, da, kb, db):
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 10:
        return 0
    a = np.float32([ka[x.queryIdx].pt for x in m])
    b = np.float32([kb[x.trainIdx].pt for x in m])
    _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    R = reps()
    orb = cv2.ORB_create(1000)
    feats = {}
    for ds, d in R.items():
        for sc, p in d.items():
            g = gray(p)
            # CLAHE so ORB finds the same corners in hazy raws and their enhancements
            kp, de = orb.detectAndCompute(cv2.createCLAHE(2.0, (8, 8)).apply(g), None)
            feats[(ds, sc)] = (sig(g), kp, de)
        print(ds, len(d), flush=True)
    rows = []
    for a, b in itertools.combinations(R, 2):
        ka, kb = [k for k in feats if k[0] == a], [k for k in feats if k[0] == b]
        A = np.stack([feats[k][0] for k in ka])  # na x 2 x d
        B = np.stack([feats[k][0] for k in kb])
        c = np.einsum("iud,jvd->ijuv", A, B).max((2, 3)) / A.shape[-1]
        for i, j in zip(*np.nonzero(c >= 0.6)):
            n = inliers(*feats[ka[i]][1:], *feats[kb[j]][1:])
            if n >= 25:
                rows.append(dict(set_a=a, scene_a=ka[i][1], set_b=b, scene_b=kb[j][1], sig_r=c[i, j], inliers=n))
        print(f"{a}-{b}: {len(ka)}x{len(kb)} candidates {(c >= 0.6).sum()}", flush=True)
    out = pd.DataFrame(rows, columns=["set_a", "scene_a", "set_b", "scene_b", "sig_r", "inliers"])
    out.to_csv(REPO / "mos/results/overlap.csv", index=False)
    print(out.groupby(["set_a", "set_b"]).agg(pairs=("inliers", "size"), scenes_a=("scene_a", "nunique"),
                                               scenes_b=("scene_b", "nunique")).to_string())


if __name__ == "__main__":
    main()

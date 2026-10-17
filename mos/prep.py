"""Build one manifest per public underwater MOS dataset: mos/manifests/<ds>.csv with
columns id,path,scene,method,mos,w,h[,votes,mos_color,mos_visibility]  (id = path relative to the dataset root; scene groups an
original image with its enhanced versions; method = 'Original' for the raw photo).

Datasets live in the gitignored uwdata/<ds>/x (extracted archives, untrusted: only read pixels/tables).
Usage: python -I mos/prep.py uid2021 [saud ...]
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "uwdata"
OUT = Path(__file__).resolve().parent / "manifests"


def uid2021():
    """UID2021 (Hou et al., TOMM 2023): 60 raw + 900 enhanced (15 methods), MOS from mos.xlsx."""
    r = ROOT / "uid2021/x"
    m = pd.read_excel(r / "mos.xlsx")
    paths = {p.name: p for p in r.glob("*/*/*.png")}
    return r, pd.DataFrame(dict(path=[paths[n] for n in m.Name], scene=m.Scene, method=m.Method, mos=m.MOS))


def saud():
    """SAUD (Jiang et al., TCSVT 2022): 100 raw x 10 UIE results, pairwise-comparison study.
    mos = Bradley-Terry score (sub_bt_scores.mat); votes = raw preference counts (sub_votes.mat, the label
    UIQAVSI/NMFC train on). Raw photos carry no score, so only the 1,000 enhanced images are listed."""
    from scipy.io import loadmat
    r = ROOT / "saud/x/Enhanced"
    bt, votes = (loadmat(ROOT / f"saud/{k}.mat")[k] for k in ("sub_bt_scores", "sub_votes"))
    names = [str(n[0]) for n in bt[:, 0]]  # each cell is a 1-element string array
    assert names == [str(n[0]) for n in votes[:, 0]]
    return r, pd.DataFrame(dict(path=[r / n for n in names], scene=["_".join(n.split("/")[:2]) for n in names],
                                method=[n.split("/")[2].split("_", 1)[1].rsplit(".", 1)[0] for n in names],
                                mos=[float(v[0, 0]) for v in bt[:, 1]], votes=[int(v[0, 0]) for v in votes[:, 1]]))


def saud2():
    """SAUD2.0 (Jiang et al., TCSVT 2024): 200 raw x 12 UIE results, overall + colour + visibility MOS.
    The MOS txt files carry no names; their order is the case-insensitive (Windows) sort of the file names:
    that order gives UIQM SRCC 0.44 against 0.16 for Python's case-sensitive sort, and only it ranks Ucolor
    above UWCNN (scratchpad saud2_order.py)."""
    r = ROOT / "saud2/x/SAUD2.0/SAUD2.0_dataset"
    names = sorted((p.name for p in r.glob("*.png")), key=str.casefold)
    m = {k: pd.read_csv(ROOT / f"saud2/x/SAUD2.0/SAUD2.0_MOS/{k}.txt", header=None)[0].to_numpy()
         for k in ("overall", "colormos", "visibility")}
    assert len(names) == 2400 and all(len(v) == 2400 for v in m.values())
    return r, pd.DataFrame(dict(path=[r / n for n in names], scene=[n.split("_")[0] for n in names],
                                method=[n.split("_", 1)[1][:-4] for n in names], mos=m["overall"],
                                mos_color=m["colormos"], mos_visibility=m["visibility"]))


def sota():
    """SOTA (Underwater-Lab-SHU, GLCQE paper): 800 raw + 9 enhancement methods each, MOS in SOTA_dataset.mat
    (MATLAB v7.3: `path` is a cell of '/<method>/<n>.bmp', img_number is the scene)."""
    import h5py
    r = ROOT / "sota/x/SOTA_Dataset"
    with h5py.File(ROOT / "sota/SOTA_dataset.mat", "r") as f:
        rel = ["".join(map(chr, f[ref][()].ravel())).strip("/") for ref in f["path"][()].ravel()]
        scene, mos = f["img_number"][()].ravel().astype(int), f["mos"][()].ravel()
    meth = [p.split("/")[0] for p in rel]
    return r, pd.DataFrame(dict(path=[r / p for p in rel], scene=scene,
                                method=["Original" if d == "raw" else d for d in meth], mos=mos))


DATASETS = {"uid2021": uid2021, "saud": saud, "saud2": saud2, "sota": sota}
EXTRA_COLS = ["votes", "mos_color", "mos_visibility"]  # secondary labels some datasets carry

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for ds in sys.argv[1:]:
        root, df = DATASETS[ds]()
        df["id"] = [str(Path(p).relative_to(root)) for p in df.path]
        hw = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[:2] for p in df.path]
        df["h"], df["w"] = zip(*hw)
        df = df[["id", "path", "scene", "method", "mos", "w", "h"] + [c for c in EXTRA_COLS if c in df]]
        assert df.id.is_unique and df.mos.notna().all(), ds
        df.to_csv(OUT / f"{ds}.csv", index=False)
        print(ds, len(df), "images,", df.scene.nunique(), "scenes,", df.method.nunique(), "methods;",
              "size", df.w.min(), "-", df.w.max(), "x", df.h.min(), "-", df.h.max())

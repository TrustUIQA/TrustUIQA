"""UCIQE from a MATLAB implementation (uwrepos/UCIQE: Z. J. Wang's UCIQE.m + rgb2lab_n.m), run unchanged in GNU
Octave, to check how much our NumPy port (uwmetrics.uciqe) matters. The two differ in scaling: the MATLAB code uses
L in 0-100 and max-min luminance contrast, ours divides L and chroma by 100 and uses the 1-99 percentile range.
Images are decoded here and passed to Octave in .mat batches (same decoding as every other metric).
The MATLAB code returns NaN on any image with a pure-black pixel (30-85% of our images), so these scores are not
used in any analysis: they only verify the NumPy port mos/uciqe_wang.py (which adds the NaN-safe variant).
Writes the score_mos.py format as uciqe_matlab.csv under scratch/octave_scores/ (mos/scores/<ds>/ or
<pool>/scores_ext/ inside it), out of reach of the analysis scripts that glob the score folders.
Usage: python -I mos/uciqe_matlab.py uid2021 saud saud2 | --pool out_ablate out
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import savemat
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from mos.score_mos import save  # noqa: E402

OCT = [str(Path.home() / "miniforge3/bin/conda"), "run", "-n", "octave", "octave-cli"]  # activated env: m-file path
RUN = REPO / "mos/run_uciqe.m"
OUT = REPO / "scratch/octave_scores"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    for ds in a.ds:
        if a.pool:
            paths = sorted((REPO / ds).glob("*/*/*.png"))
            ids = [f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}" for p in paths]
            dst = OUT / ds / "scores_ext/uciqe_matlab.csv"
        else:
            man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
            paths, ids = list(man.path), list(man.id)
            dst = OUT / "mos/scores" / ds / "uciqe_matlab.csv"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        scores = []
        with tempfile.TemporaryDirectory(dir=REPO / "scratch") as tmp:
            for i in tqdm(range(0, len(paths), 100), desc=ds, mininterval=30):
                ims = {f"im{j + 1}": cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                       for j, p in enumerate(paths[i:i + 100])}
                savemat(f"{tmp}/b.mat", ims, do_compression=False)
                subprocess.run([*OCT, "-q", str(RUN), f"{tmp}/b.mat", f"{tmp}/b.txt"], check=True)
                s = np.loadtxt(f"{tmp}/b.txt", ndmin=1)
                assert len(s) == len(ims), (len(s), len(ims))
                scores += s.tolist()
        save(dst, ids, scores, False, a.pool)
        print(f"done {ds} ({len(scores)})", flush=True)

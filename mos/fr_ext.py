"""Every other pyiqa full-reference metric (+ CIEDE2000) against the UIEB reference, on fr_score.py's study images.

fr_score.py already wrote psnr/ssim/lpips/dists/afine_all to <pool>/scores/; these go to <pool>/scores_ext/<m>_gt.csv
in the same format. Run from the repo root: python -I mos/fr_ext.py [metric ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fr_score import TARGETS, gt_like  # noqa: E402
from score import crop32, rss_guard  # noqa: E402

DONE = {"psnr", "ssim", "lpips", "dists", "afine_all", "fgresq_pair"}  # fgresq_pair compares two test images
METRICS = ["ciede2000", "rmse", "ms_ssim", "cw_ssim", "ssimc", "psnry", "fsim", "vif", "gmsd", "vsi", "mad", "nlpd", "msswd",
           "lpips-vgg", "lpips+", "lpips-vgg+", "stlpips", "stlpips-vgg", "pieapp", "ahiq", "ckdn", "deepdc", "dmm",
           "topiq_fr", "topiq_fr-pipal", "wadiqam_fr", "afine_fr", "afine_all_scale"]


def ciede2000(p, ref):
    from skimage.color import deltaE_ciede2000, rgb2lab
    a, b = (rgb2lab(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)) for x in (p, ref))
    return float(np.mean(deltaE_ciede2000(a, b)))


def rmse(p, ref):
    """Pixel RMSE on 0-255 RGB (Rajput et al. supplementary)."""
    a, b = (cv2.imread(x).astype(np.float64) for x in (p, ref))
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    import pyiqa
    rss_guard()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [(d, mode, p) for d, mode, s in TARGETS for p in sorted(Path(d, mode).glob(f"*/{s:02d}.png"))]
    for m in sys.argv[1:] or METRICS:
        dsts = {d: Path(d, "scores_ext", f"{m}_gt.csv") for d in ("out_ablate", "out")}
        if all(p.exists() for p in dsts.values()):
            continue
        try:
            if m in ("ciede2000", "rmse"):
                f, lower = {"ciede2000": ciede2000, "rmse": rmse}[m], True
            else:
                g = pyiqa.create_metric(m, device=dev)
                lower = g.lower_better
                f = (lambda p, r, g=g: g(crop32(p), crop32(r))) if m.startswith("afine") else g
            rows = {d: [] for d in dsts}
            with torch.no_grad():
                for d, mode, p in tqdm(jobs, desc=m, mininterval=30):
                    ref = gt_like(p.parts[-2], cv2.imread(str(p)).shape)
                    rows[d].append(dict(mode=mode, image=p.parts[-2], step=int(p.stem), metric=f"{m}_gt",
                                        score=float(f(str(p), ref)), lower_better=lower))
        except Exception as e:  # unsupported input size / download failure: report, keep going
            print(f"fail {m}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        finally:
            torch.cuda.empty_cache()
        for d, r in rows.items():
            dsts[d].parent.mkdir(exist_ok=True)
            pd.DataFrame(r).to_csv(dsts[d], index=False)
        print(f"done {m} ({len(jobs)}), lower_better={lower}", flush=True)

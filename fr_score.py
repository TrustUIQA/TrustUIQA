"""Full-reference scores against the UIEB reference (gt) image, for the conditions used in the rating study.

The study's "clean" image is the UIEB raw photo, so FR metrics need the UIEB reference-890 image as ground truth.
gt is resized exactly like ablate.py/attack.py resized raw (INTER_AREA to max side 512) so pixels align.
Writes <metric>_gt.csv into out_ablate/scores and out/scores, the format analyze_votes.py already reads.
Usage (from repo root): python fr_score.py [metric ...]   (default: all of METRICS)
"""
import sys
from pathlib import Path

import cv2
import pandas as pd
import pyiqa
import torch
from tqdm import tqdm

from score import crop32, rss_guard

# pixel fidelity, structure, two deep perceptual FR metrics, and A-FINE (CVPR 2025): fidelity + naturalness,
# built for test images that may be better than an imperfect reference
METRICS = ["psnr", "ssim", "lpips", "dists", "afine_all"]
# (score dir, mode, step) for every image a study pair can touch
TARGETS = [("out_ablate", m, s) for m, s in [
    ("clean", 0), ("sharpen@mild", 1), ("clahe@mild", 1), ("saturation@mild", 1), ("saturation@moderate", 1),
    ("contrast@mild", 1), ("noise@mild", 1), ("noise@strong", 1), ("posterize@strong", 1), ("hue@moderate", 1),
    ("red_boost@strong", 1)]] + [("out", "naive", s) for s in (0, 5, 20)]


def gt_like(img_id, shape, cache=Path("data/uieb/gt512")):
    """UIEB reference resized the way the raw image was; cached as PNG so pyiqa can read paths."""
    dst = cache / f"{img_id}.png"
    if not dst.exists():
        cache.mkdir(parents=True, exist_ok=True)
        g = cv2.imread(f"data/uieb/gt/{img_id}.png")
        scale = 512 / max(g.shape[:2])
        if scale < 1:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), g)
    assert cv2.imread(str(dst)).shape == shape, (img_id, shape)
    return str(dst)


if __name__ == "__main__":
    rss_guard()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [(d, m, p) for d, mode, s in TARGETS for p in sorted(Path(d, mode).glob(f"*/{s:02d}.png")) for m in [mode]]
    for metric in sys.argv[1:] or METRICS:
        f = pyiqa.create_metric(metric, device=dev)
        rows = {"out_ablate": [], "out": []}
        with torch.no_grad():
            for d, mode, p in tqdm(jobs, desc=metric, mininterval=30):
                img_id = p.parts[-2]
                ref = gt_like(img_id, cv2.imread(str(p)).shape)
                rows[d].append(dict(mode=mode, image=img_id, step=int(p.stem), metric=f"{metric}_gt",
                                    score=float(f(crop32(p), crop32(ref)) if metric.startswith("afine") else f(str(p), ref)),
                                    lower_better=f.lower_better))
        for d, r in rows.items():
            pd.DataFrame(r).to_csv(Path(d, "scores", f"{metric}_gt.csv"), index=False)
        print(f"done {metric}: {sum(map(len, rows.values()))} images, lower_better={f.lower_better}")
        del f
        torch.cuda.empty_cache()

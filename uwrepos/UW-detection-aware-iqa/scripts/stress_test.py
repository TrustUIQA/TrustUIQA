"""O2: over-enhancement stress test.

Samples N raw images from a manifest, applies each sweep at increasing
strength, scores every variant, and reports per-metric inflation
(Spearman of score vs strength, fraction of increasing steps).

    python scripts/stress_test.py --dataset uid2021 --metrics uiqm uciqe topiq_nr liqe
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root, load_manifest
from uwiqa.enhance import DEFAULT_STRENGTHS, SWEEPS, monotonic_inflation
from uwiqa.metrics.registry import MetricRunner, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uid2021")
    ap.add_argument("--method", default="raw", help="which manifest rows to sample")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--metrics", nargs="*", default=["uiqm", "uciqe", "topiq_nr", "liqe", "musiq"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-images", action="store_true", help="keep variants (for a 2AFC study)")
    args = ap.parse_args()

    man = load_manifest(args.dataset)
    man = man[man["method"] == args.method]
    sample = man.sample(min(args.n, len(man)), random_state=args.seed)["image"].tolist()
    root = dataset_root(args.dataset)
    runner = MetricRunner(args.metrics)

    img_dir = RESULTS_ROOT / "stress" / "images" if args.save_images else Path(tempfile.mkdtemp())
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in tqdm(sample, desc="images"):
        base = load_rgb(root / im)
        for sweep, op in SWEEPS.items():
            for s in DEFAULT_STRENGTHS:
                p = img_dir / f"{Path(im).stem}__{sweep}__{s}.png"
                Image.fromarray(op(base, s)).save(p)
                for m in args.metrics:
                    rows.append({"image": im, "sweep": sweep, "strength": s, "metric": m,
                                 "score": runner.get(m).fn(p)})

    out = RESULTS_ROOT / "stress"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / f"o2_{args.dataset}_raw.csv", index=False)

    summ = []
    for (m, sw, im), g in raw.groupby(["metric", "sweep", "image"]):
        g = g.sort_values("strength")
        summ.append({"metric": m, "sweep": sw, **monotonic_inflation(g["strength"], g["score"])})
    summ = pd.DataFrame(summ).groupby(["metric", "sweep"]).mean(numeric_only=True).reset_index()
    summ.to_csv(out / f"o2_{args.dataset}_summary.csv", index=False)
    print(summ.pivot(index="metric", columns="sweep", values="srcc_vs_strength").round(2))


if __name__ == "__main__":
    main()

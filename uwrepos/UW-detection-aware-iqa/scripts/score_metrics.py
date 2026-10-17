"""Score every image in a manifest with every NR metric. Resumable.

Output is long-format CSV results/scores/<dataset>.csv (image, metric, score),
appended every --flush rows, so a killed Colab session loses at most one
flush. Re-running skips (image, metric) pairs already present.

    python scripts/score_metrics.py --dataset uieb
    python scripts/score_metrics.py --dataset uid2021 --metrics uiqm uciqe topiq_nr
"""
import argparse
import csv
import time

import pandas as pd
from tqdm import tqdm

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root, load_manifest
from uwiqa.metrics.registry import DEFAULT_METRICS, MetricRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale long side (only if a metric OOMs; record it)")
    ap.add_argument("--flush", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    args = ap.parse_args()

    man = load_manifest(args.dataset)
    root = dataset_root(args.dataset)
    images = man["image"].tolist()[: args.limit]

    out = RESULTS_ROOT / "scores" / f"{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["image"], prev["metric"]))
    new_file = not out.exists()

    runner = MetricRunner(args.metrics, device=args.device, max_side=args.max_side)
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["image", "metric", "score", "higher_better"])
        for name in args.metrics:
            todo = [im for im in images if (im, name) not in done]
            if not todo:
                print(f"{name}: already complete")
                continue
            m = runner.get(name)
            t0 = time.time()
            for i, im in enumerate(tqdm(todo, desc=name)):
                try:
                    s = m.fn(root / im)
                except Exception as e:
                    print(f"  ! {name} failed on {im}: {e}")
                    s = float("nan")
                w.writerow([im, name, s, int(m.higher_better)])
                if (i + 1) % args.flush == 0:
                    fh.flush()
            fh.flush()
            print(f"{name}: {len(todo)} imgs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

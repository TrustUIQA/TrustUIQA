"""Train the frozen reference detector on raw RUOD (protocol (a), O3).

Uses the leakage-safe split from make_splits.py: trains on det_train,
early-stops on det_val. pred_* images are never seen. Resumes automatically
from last.pt if a previous run was interrupted (attach the previous version's
output and copy results/detector back first).

    python scripts/train_detector.py                       # yolo11s, 640, 100 epochs
    python scripts/train_detector.py --epochs 3 --fraction 0.1   # smoke test
"""
import argparse
import os

import pandas as pd
import yaml

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root

RUOD_NAMES = ["holothurian", "echinus", "scallop", "starfish", "fish",
              "corals", "diver", "cuttlefish", "turtle", "jellyfish"]


def write_lists(out_dir):
    """Absolute image-path lists per split + an Ultralytics data yaml.

    Ultralytics finds labels by replacing the last /images/ with /labels/,
    which matches both the RUOD (<split>/images) and DUO (images/<split>) layouts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(RESULTS_ROOT / "splits" / "ruod_splits.csv")
    root = dataset_root("ruod")
    for s, g in sp.groupby("split"):
        (out_dir / f"ruod_{s}.txt").write_text("\n".join(str(root / p) for p in g.image) + "\n")
    duo = RESULTS_ROOT / "splits" / "duo_clean.csv"
    if duo.exists():
        droot = dataset_root("duo")
        d = pd.read_csv(duo)
        (out_dir / "duo_clean.txt").write_text("\n".join(str(droot / p) for p in d.image) + "\n")
    data = {"path": str(out_dir), "train": "ruod_det_train.txt", "val": "ruod_det_val.txt",
            "test": "ruod_pred_test.txt", "names": dict(enumerate(RUOD_NAMES))}
    yml = out_dir / "ruod_det.yaml"
    yaml.safe_dump(data, open(yml, "w"), sort_keys=False)
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fraction", type=float, default=1.0, help="<1 for a quick smoke test")
    ap.add_argument("--name", default="raw_yolo11s")
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1; default: all visible GPUs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    yml = write_lists(RESULTS_ROOT / "splits" / "yolo")
    project = RESULTS_ROOT / "detector"
    last = project / args.name / "weights" / "last.pt"
    if not torch.cuda.is_available() and args.device != "cpu":
        raise SystemExit("No GPU visible. On Kaggle set Session options -> Accelerator -> GPU T4 x2 "
                         "(a CPU run would take days). Pass --device cpu to force it anyway.")
    device = args.device or ",".join(map(str, range(torch.cuda.device_count())))

    if last.exists():
        print(f"resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(yml), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                    batch=args.batch, fraction=args.fraction, device=device, seed=0,
                    deterministic=True, workers=min(8, os.cpu_count() or 2),
                    project=str(project), name=args.name, exist_ok=True, plots=not args.no_plots)

    best = project / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    det = YOLO(str(best))
    rows = []
    for split_name, extra in [("ruod_pred_test", {}), ("duo_clean", {"classes": [0, 1, 2, 3]})]:
        lst = yml.parent / f"{split_name}.txt"
        if not lst.exists():
            continue
        tmp = yml.parent / f"_eval_{split_name}.yaml"
        yaml.safe_dump({**yaml.safe_load(open(yml)), "val": lst.name}, open(tmp, "w"))
        m = det.val(data=str(tmp), imgsz=args.imgsz, batch=args.batch, device=device.split(",")[0],
                    project=str(project), name=f"eval_{split_name}", exist_ok=True,
                    plots=not args.no_plots, **extra)
        rows.append({"set": split_name, "mAP50": m.box.map50, "mAP50-95": m.box.map})
    res = pd.DataFrame(rows)
    res.to_csv(project / args.name / "heldout_eval.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

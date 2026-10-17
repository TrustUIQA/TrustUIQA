"""Score every image under out/<mode>/<image>/<step>.png with pyiqa NR-IQA metrics.

One CSV per metric in out/scores/ (rerun skips finished metrics, so it resumes).
Usage: python score.py --out out [--metrics brisque niqe ...]
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import pyiqa
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from uwmetrics import uciqe, uiqm  # noqa: E402

UW = {"uiqm": uiqm, "uciqe": uciqe}  # hand-crafted underwater metrics (higher = better); not in METRICS default


def _invert_attention_mask(self, m):
    """transformers 5 dropped ModuleUtilsMixin.invert_attention_mask; pyiqa's vendored Q-Align still calls it.
    Same math as transformers 4.x: 1 = attend -> 0, 0 = masked -> dtype min."""
    e = (m[:, None, :, :] if m.dim() == 3 else m[:, None, None, :]).to(self.dtype)
    return (1.0 - e) * torch.finfo(self.dtype).min


def rss_guard(limit_gb=float(os.environ.get("RSS_LIMIT_GB", 9))):
    """Exit before a model's host RAM OOMs the WSL VM (a global OOM there restarts the whole distro,
    killing every other job). Polls /proc/self/statm every 0.2 s; exit code 3 = over the limit.
    Counts resident minus file-backed pages: mmapped weights (Q-Align's 15.7 GB safetensors) are reclaimable."""
    page = os.sysconf("SC_PAGE_SIZE")

    def watch():
        while True:
            _, res, shared = map(int, open("/proc/self/statm").read().split()[:3])
            rss = (res - shared) * page
            if rss > limit_gb * 2**30:
                print(f"ABORT: RSS {rss / 2**30:.1f} GB > {limit_gb} GB limit", flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()


def crop32(p):
    """A-FINE asserts H and W divisible by 32: centre-crop to that (< 32 px lost per axis), [1,3,H,W] in [0,1]."""
    x = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = x.shape[0] // 32 * 32, x.shape[1] // 32 * 32
    t, l = (x.shape[0] - h) // 2, (x.shape[1] - w) // 2
    return torch.from_numpy(x[t:t + h, l:l + w].copy()).permute(2, 0, 1)[None].float() / 255


def patch_qalign():
    from pyiqa.archs.q_align.visual_encoder import MplugOwlVisualAbstractorModel as M
    if not hasattr(M, "invert_attention_mask"):
        M.invert_attention_mask = _invert_attention_mask

# Banana100's 21 metrics + URanker (underwater-specific). Q-Align 8-bit: fp16 7B does not fit 16 GB.
METRICS = ["arniqa", "brisque", "clipiqa", "cnniqa", "dbcnn", "hyperiqa", "ilniqe", "liqe", "maniqa",
           "musiq", "nima", "niqe", "nrqm", "paq2piq", "pi", "piqe", "topiq_nr", "tres", "wadiqam_nr",
           "qualiclip", "uranker", "qalign_8bit"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--modes", nargs="+", help="only these mode folders (default: all)")
    ap.add_argument("--steps", nargs="+", type=int, help="only these steps (default: all)")
    a = ap.parse_args()
    rss_guard()
    out = Path(a.out)
    imgs = sorted(p for p in out.glob("*/*/*.png")
                  if (not a.modes or p.parts[-3] in a.modes) and (not a.steps or int(p.stem) in a.steps))
    (out / "scores").mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in a.metrics:
        dst = out / "scores" / f"{m}.csv"
        if dst.exists():
            continue
        if m in UW:
            f, lower_better = (lambda p, fn=UW[m]: fn(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))), False
        else:
            try:
                if m.startswith("qalign"):
                    patch_qalign()
                f = pyiqa.create_metric(m, device=dev)
            except Exception as e:  # unknown name / weights download failure: report, keep going
                print(f"skip {m}: {e}")
                continue
            lower_better = f.lower_better
            if m.startswith("afine"):
                f = (lambda p, g=f: g(crop32(p)))
        rows = []
        with torch.no_grad():
            for p in tqdm(imgs, desc=m, mininterval=30):
                rows.append(dict(mode=p.parts[-3], image=p.parts[-2], step=int(p.stem),
                                 metric=m, score=float(f(str(p))), lower_better=lower_better))
        pd.DataFrame(rows).to_csv(dst, index=False)
        print(f"done {m} ({len(rows)})")
        del f
        torch.cuda.empty_cache()

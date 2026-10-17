"""Iterated cheap-op attack on underwater images.

Modes (each runs --steps iterations from the clean image, step 0 = clean):
  naive         fixed chain every step: saturate -> unsharp -> CLAHE
  greedy_uiqm   each step apply the single candidate op that maximises UIQM
  greedy_uciqe  same, maximising UCIQE
Candidates include Gaussian noise: if greedy picks it, the metric rewards visible noise.

Writes out/<mode>/<image>/<step>.png and out/attack.csv (UIQM, UCIQE, chosen op per step).
Usage: python attack.py data/uieb/raw --n 100 --steps 20
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from uwmetrics import uciqe, uiqm


def saturate(x, s):
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def unsharp(x, a):
    return x + a * (x - cv2.GaussianBlur(x, (0, 0), 2))


def clahe(x, c):
    lab = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(c, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def noise(x, s, rng):
    return x + rng.normal(0, s, x.shape).astype(np.float32)


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


CANDIDATES = [("sat", 1.1), ("sat", 1.3), ("unsharp", 0.5), ("unsharp", 1.5),
              ("clahe", 2.0), ("clahe", 4.0), ("noise", 0.02), ("noise", 0.05)]
TARGETS = {"greedy_uiqm": uiqm, "greedy_uciqe": uciqe}


def apply(op, v, x, rng):
    f = {"sat": saturate, "unsharp": unsharp, "clahe": clahe}.get(op)
    return to_u8(noise(x, v, rng) if op == "noise" else f(x, v))


def run_one(path, out, steps, max_side):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    scale = max_side / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rows = []
    for mode in ["naive", *TARGETS]:
        rng = np.random.default_rng(0)  # same noise draws per image+mode, reproducible
        d = out / mode / path.stem
        d.mkdir(parents=True, exist_ok=True)
        img, op = rgb, "clean"
        for t in range(steps + 1):
            if t:
                x = img.astype(np.float32) / 255
                if mode == "naive":
                    img, op = to_u8(clahe(np.clip(unsharp(saturate(x, 1.15), 0.5), 0, 1), 2.0)), "chain"
                else:
                    cands = [(apply(o, v, x, rng), f"{o}:{v}") for o, v in CANDIDATES]
                    img, op = max(cands, key=lambda c: TARGETS[mode](c[0]))
            cv2.imwrite(str(d / f"{t:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append(dict(mode=mode, image=path.stem, step=t, op=op, uiqm=uiqm(img), uciqe=uciqe(img)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in exts)[: a.n]
    out = Path(a.out)
    with Pool(a.workers) as pool:
        rows = pool.map(partial(run_one, out=out, steps=a.steps, max_side=a.max_side), paths)
    df = pd.DataFrame([r for rs in rows for r in rs])
    df.to_csv(out / "attack.csv", index=False)
    last = df[df.step.isin([0, a.steps])].pivot_table(index="mode", columns="step", values=["uiqm", "uciqe"])
    print(last.round(3))

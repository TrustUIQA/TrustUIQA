"""Single-shot ablation: each op alone at 3 strengths (mild/moderate/strong), plus combos at moderate.

Answers "which condition breaks which metric". Layout matches score.py:
  out_ablate/clean/<image>/00.png      clean reference (scored once)
  out_ablate/<op>@<level>/<image>/01.png   degraded
Usage: python ablate.py data/uieb/raw --n 90
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from attack import clahe, saturate, to_u8, unsharp


def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def jpeg(x, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def gamma(x, g):
    return np.clip(x, 0, 1) ** g


def contrast(x, k):
    return (x - x.mean()) * k + x.mean()


def hue(x, deg):
    hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + deg) % 360
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def posterize(x, bits):
    q = 2 ** (8 - bits)
    return (to_u8(x) // q * q).astype(np.float32) / 255


def red_boost(x, k):  # over-correcting the underwater red loss
    y = x.copy()
    y[..., 0] *= k
    return y


def noise(x, s, seed=0):
    return x + np.random.default_rng(seed).normal(0, s, x.shape).astype(np.float32)


# op -> (fn, mild, moderate, strong)
OPS = {
    "saturation": (saturate, 1.3, 1.6, 2.0),
    "sharpen": (unsharp, 0.5, 1.5, 3.0),
    "clahe": (clahe, 2.0, 4.0, 8.0),
    "noise": (noise, 0.02, 0.05, 0.10),
    "blur": (blur, 1.0, 2.0, 4.0),
    "jpeg": (jpeg, 50, 20, 5),
    "gamma": (gamma, 0.8, 0.6, 0.4),
    "contrast": (contrast, 1.3, 1.6, 2.0),
    "hue": (hue, 15, 45, 90),
    "posterize": (posterize, 5, 4, 3),
    "red_boost": (red_boost, 1.2, 1.5, 2.0),
}
LEVELS = ("mild", "moderate", "strong")
COMBOS = [("saturation", "sharpen"), ("saturation", "noise"), ("sharpen", "noise"),
          ("saturation", "sharpen", "noise"), ("saturation", "sharpen", "clahe")]


def conditions():
    for op, (fn, *vals) in OPS.items():
        for lvl, v in zip(LEVELS, vals):
            yield f"{op}@{lvl}", [(fn, v)]
    for combo in COMBOS:
        yield "+".join(combo) + "@moderate", [(OPS[o][0], OPS[o][2]) for o in combo]


def run_one(path, out, max_side):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    scale = max_side / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def save(cond, step, img):
        d = out / cond / path.stem
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{step:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    save("clean", 0, rgb)
    for cond, chain in conditions():
        x = rgb.astype(np.float32) / 255
        for fn, v in chain:
            x = np.clip(fn(x, v), 0, 1)
        save(cond, 1, to_u8(x))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="out_ablate")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--max-side", type=int, default=512)
    a = ap.parse_args()
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in exts)[: a.n]
    with Pool() as pool:
        pool.map(partial(run_one, out=Path(a.out), max_side=a.max_side), paths)
    print(len(list(conditions())), "conditions x", len(paths), "images ->", a.out)

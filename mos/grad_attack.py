"""Gradient (PGD) attack on the raw photo of each edit-set scene, as in Antsiferova et al. (AAAI 2024): can an
invisible L-inf perturbation of the *unenhanced* photo make a differentiable metric score it above the
enhancement people rated best? Complements the gradient-free cheap edits (mos/edits.py).

Per scene, metric and budget eps: 10 PGD steps of eps/4 on the oriented score (higher = better), then the image is
quantised to 8-bit PNG (uwdata/_gradattack/<ds>/<metric>_e<eps*255>/<scene>.png, research-only data: never pushed)
and rescored from the file, exactly as score_mos.py scores the MOS images. Every attacked image is also scored by
the other metrics of TRANSFER (does the attack carry over?). Rows: mos/results/gradattack_<ds>.csv
  ds, scene, attacked, eps, metric, clean, adv, best, psnr   (scores oriented: higher = better)
Usage: python -I mos/grad_attack.py uid2021 saud sota [--scenes 100] [--metrics ...] [--eps 2 8]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from score import UW, rss_guard  # noqa: E402
from mos.score_mos import min224  # noqa: E402

# differentiable NR models of the paper's metric set: four pyiqa models + RALI (mos/rali_score.py)
ATTACKED = ["musiq", "liqe", "topiq_nr", "clipiqa+", "rali"]
TRANSFER = ATTACKED + ["uiqm", "uciqe"]
OUT = REPO / "uwdata/_gradattack"
STEPS = 10


def load(p):
    """[1,3,H,W] in [0,1]; short side upsampled to 224 like score_mos.min224 (LIQE needs it; harmless elsewhere)."""
    t = min224(p)
    if isinstance(t, torch.Tensor):
        return t
    x = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255


def oriented(ds, metric):
    d = pd.read_csv(REPO / f"mos/scores/{ds}/{metric}.csv")
    return pd.Series(np.where(d.lower_better, -d.score, d.score), index=d.id)


def pgd(f, sign, x, eps):
    d = torch.zeros_like(x, requires_grad=True)
    for _ in range(STEPS):
        s = sign * f((x + d).clamp(0, 1)).sum()
        (g,) = torch.autograd.grad(s, d)
        with torch.no_grad():
            d += eps / 4 * g.sign()
            d.clamp_(-eps, eps)
    return (x + d.detach()).clamp(0, 1)


class RaliLoss(torch.nn.Module):
    """RALI on a [B,3,H,W] tensor in [0,1]: its CLIPProcessor steps (shortest edge to 336 bicubic, centre crop,
    normalise) redone in torch so gradients flow. Attacked images are still rescored through the original PIL path."""
    lower_better = False

    def __init__(self, dev):
        super().__init__()
        from mos.rali_score import model
        self.m = model(dev)
        p = self.m.processor.image_processor
        self.size = p.crop_size["height"]
        self.mean = torch.tensor(p.image_mean, device=dev)[:, None, None]
        self.std = torch.tensor(p.image_std, device=dev)[:, None, None]

    def forward(self, x):
        h, w = x.shape[-2:]
        k = self.size / min(h, w)
        x = F.interpolate(x, size=(max(self.size, round(h * k)), max(self.size, round(w * k))), mode="bicubic",
                          antialias=True, align_corners=False).clamp(0, 1)
        t, l = (x.shape[-2] - self.size) // 2, (x.shape[-1] - self.size) // 2
        x = x[..., t:t + self.size, l:l + self.size]
        return self.m.scorer(self.m.clip.get_image_features(pixel_values=(x - self.mean) / self.std))


def scorer(m, dev):
    """file path -> oriented score."""
    if m == "rali":
        from mos.rali_score import model
        r = model(dev)
        return lambda p: float(r([Image.open(p).convert("RGB")]))
    if m in UW:
        return lambda p: float(UW[m](cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float64)))
    f = pyiqa.create_metric(m, device=dev)
    sign = -1 if f.lower_better else 1
    pre = min224 if m.startswith("liqe") else str
    return lambda p: sign * float(f(pre(str(p))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--scenes", type=int, default=100, help="first N edit-set scenes (sorted) per dataset")
    ap.add_argument("--metrics", nargs="+", default=ATTACKED)
    ap.add_argument("--eps", nargs="+", type=int, default=[2, 8], help="L-inf budgets in 8-bit levels")
    a = ap.parse_args()
    rss_guard()
    dev = "cuda"
    for ds in a.ds:
        dst = REPO / f"mos/results/gradattack_{ds}.csv"
        done = pd.read_csv(dst) if dst.exists() else pd.DataFrame(columns=["attacked", "eps"])
        man = pd.read_csv(REPO / f"mos/manifests/{ds}.csv")
        man = man[man.method != "Original"]
        top = man.loc[man.groupby("scene").mos.idxmax()].set_index("scene").id  # people's best per scene
        top.index = top.index.astype(str)
        em = pd.read_csv(REPO / f"mos/manifests/{ds}_edits.csv")
        raw = em[em.method == "raw"].assign(scene=lambda d: d.scene.astype(str)).sort_values("scene")
        raw = raw.head(a.scenes)
        for m in a.metrics:
            for e in a.eps:
                if ((done.attacked == m) & (done.eps == e)).any():
                    continue
                f = RaliLoss(dev) if m == "rali" else pyiqa.create_metric(m, as_loss=True, device=dev)
                sign = -1 if f.lower_better else 1
                for s, p in tqdm(list(zip(raw.scene, raw.path)), desc=f"{ds} {m} e{e}", mininterval=30):
                    o = OUT / ds / f"{m}_e{e}" / f"{s}.png"
                    if o.exists():
                        continue
                    x = load(p).to(dev)
                    try:
                        y = pgd(f, sign, x, e / 255)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"OOM {ds} {s} {tuple(x.shape)}", flush=True)
                        continue
                    o.parent.mkdir(parents=True, exist_ok=True)
                    u8 = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    cv2.imwrite(str(o), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
                del f
                torch.cuda.empty_cache()
        # score clean raw + every attacked image with every transfer metric
        rows = []
        for t in TRANSFER:
            g, best = scorer(t, dev), oriented(ds, t)
            with torch.no_grad():
                for s, p in tqdm(list(zip(raw.scene, raw.path)), desc=f"{ds} score {t}", mininterval=30):
                    c = g(p)
                    for m in a.metrics:
                        for e in a.eps:
                            o = OUT / ds / f"{m}_e{e}" / f"{s}.png"
                            if not o.exists():
                                continue
                            x0, x1 = load(p)[0], load(o)[0]
                            mse = float(((x0 - x1) ** 2).mean())
                            rows.append(dict(ds=ds, scene=s, attacked=m, eps=e, metric=t, clean=c, adv=g(o),
                                             best=best[top[s]], psnr=10 * np.log10(1 / max(mse, 1e-12))))
            del g
            torch.cuda.empty_cache()
        pd.DataFrame(rows).round(5).to_csv(dst, index=False)
        r = pd.DataFrame(rows)
        r["beat_clean"], r["beat_adv"] = r.clean > r.best, r.adv > r.best
        print(f"== {ds}: share of scenes where the raw photo beats the people's best, clean vs attacked")
        print(r.groupby(["attacked", "eps", "metric"])[["beat_clean", "beat_adv", "psnr"]].mean().round(2).to_string())

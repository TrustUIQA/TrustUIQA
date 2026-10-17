"""RALI (Zhao et al., "Reasoning as Representation", ICLR 2026): CLIP + prototype scorer distilled from an RL-trained
reasoning IQA model. Banana100 found it one of only two NR metrics that track iterative-editing degradation, so we
score it everywhere the other metrics are scored. Higher = better.

Third-party code and weights (uwrepos/RALI, weights from HF ByteDance/Q-Insight/RALI) are untrusted: the torch
checkpoint loads with weights_only=True and the sklearn PCA pickle through a whitelist unpickler.
Output format = score_mos.py (mos/scores/<ds>/rali.csv, or <pool>/scores_ext/rali.csv with --pool).
Usage: python -I mos/rali_score.py uid2021 saud saud2 sota uid2021_edits ... | --pool out_ablate out ...
"""
import argparse
import sys
from pathlib import Path

import joblib.numpy_pickle as jnp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
RALI = REPO / "uwrepos/RALI"
CK = RALI / "checkpoints_hf/RALI"
sys.path[:0] = [str(REPO), str(RALI)]
import demo_rali_score as rali  # noqa: E402
from mos.score_mos import save  # noqa: E402

SAFE = {("sklearn.decomposition._pca", "PCA"), ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar")}


class SafeUnpickler(jnp.NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise RuntimeError(f"blocked pickle global {module}.{name}")
        return super().find_class(module, name)


def safe_load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(str(path), f, ensure_native_byte_order=True, mmap_mode=None).load()


_gif = rali.CLIPModel.get_image_features


def _image_features(self, **kw):
    """RALI was written for transformers 4.51 (returns the projected embedding); v5 wraps it in pooler_output."""
    o = _gif(self, **kw)
    return o if torch.is_tensor(o) else o.pooler_output


def model(dev):
    rali.CLIPModel.get_image_features = _image_features
    rali.joblib.load = safe_load  # PrototypeScorer calls joblib.load(pca_path)
    m = rali.CLIP_IQA_Proto(str(CK / "best"), torch.device(dev), pca_path=str(CK / "pca.pkl"),
                            basis_path=str(CK / "basis.npz"))
    m.load_state_dict(torch.load(CK / "ckpt.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return m.to(dev).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    m = None
    for ds in a.ds:
        if a.pool:
            paths = sorted((REPO / ds).glob("*/*/*.png"))
            ids = [f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}" for p in paths]
            dst = REPO / ds / "scores_ext/rali.csv"
        else:
            man = pd.read_csv(REPO / "mos/manifests" / f"{ds}.csv")
            paths, ids = list(man.path), list(man.id)
            dst = REPO / "mos/scores" / ds / "rali.csv"
        if dst.exists():
            continue
        m = m or model(dev)
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(paths), 16), desc=ds, mininterval=30):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
                scores += m(ims).float().cpu().tolist()
        dst.parent.mkdir(parents=True, exist_ok=True)
        save(dst, ids, scores, False, a.pool)
        print(f"done {ds} ({len(scores)})", flush=True)

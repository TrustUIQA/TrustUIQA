"""Cheap edits of each scene's raw photo in a MOS dataset, to test metrics against the human ranking:
does a metric score a cheap edit of the raw image above the enhancement humans rated best for that scene?

Edits are ablate.py conditions applied at native resolution. Images go to uwdata/_edits/<ds>/<edit>/<scene>.png
(derived from research-only datasets, so kept outside the repo); manifest mos/manifests/<ds>_edits.csv
(id,path,scene,method) is scored by score_mos.py like any dataset.
Usage: python -I mos/edits.py uid2021 saud sota [--scenes 200]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ablate import conditions  # noqa: E402
from attack import to_u8  # noqa: E402

EDITS = ["saturation+sharpen@moderate", "saturation+sharpen+clahe@moderate", "saturation+sharpen+noise@moderate",
         "clahe@moderate", "red_boost@moderate", "noise@moderate"]
OUT = REPO / "uwdata/_edits"


def raws(ds):
    """scene -> raw image path."""
    m = pd.read_csv(REPO / f"mos/manifests/{ds}.csv")
    if ds == "saud":  # scene group_<g>_<n> -> Raw/group_<g>/<n>.png (raw photos carry no score)
        r = REPO / "uwdata/saud/x/Raw"
        return {s: r / s.rsplit("_", 1)[0] / f"{s.rsplit('_', 1)[1]}.png" for s in m.scene.unique()}
    o = m[m.method == "Original"]
    return dict(zip(o.scene, o.path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="+")
    ap.add_argument("--scenes", type=int, default=200, help="random subset of scenes when a dataset has more")
    a = ap.parse_args()
    chains = {k: v for k, v in conditions() if k in EDITS}
    assert len(chains) == len(EDITS)
    for ds in a.ds:
        rw = raws(ds)
        scenes = sorted(rw, key=str)
        if len(scenes) > a.scenes:
            scenes = sorted(np.random.default_rng(0).choice(scenes, a.scenes, replace=False).tolist(), key=str)
        rows = []
        for s in scenes:
            rgb = cv2.cvtColor(cv2.imread(str(rw[s])), cv2.COLOR_BGR2RGB)
            rows.append(dict(id=f"raw/{s}", path=str(rw[s]), scene=s, method="raw"))
            for e, chain in chains.items():
                x = rgb.astype(np.float32) / 255
                for fn, v in chain:
                    x = np.clip(fn(x, v), 0, 1)
                p = OUT / ds / e / f"{s}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2BGR))
                rows.append(dict(id=f"{e}/{s}", path=str(p), scene=s, method=e))
        pd.DataFrame(rows).to_csv(REPO / f"mos/manifests/{ds}_edits.csv", index=False)
        print(ds, len(scenes), "scenes x", len(EDITS), "edits (+ raw)")

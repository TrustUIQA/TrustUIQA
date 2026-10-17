"""Build rating-study stimuli: one vertical JPEG sprite per image + manifest.json.

Tile order per study sprite: clean, blur@strong (attention check), then the 12 test conditions (CONDS order).
Tiles are padded to a multiple of 8 px so JPEG blocks never straddle two conditions.
JPEG q95 with 4:4:4 chroma (no subsampling): colour shifts are part of the study.
Usage (from repo root): python study/build_stimuli.py
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

# code -> (label, source path template); {i} = image id
CONDS = {
    "sharpen_mild": "out_ablate/sharpen@mild/{i}/01.png",
    "clahe_mild": "out_ablate/clahe@mild/{i}/01.png",
    "sat_mild": "out_ablate/saturation@mild/{i}/01.png",
    "sat_mod": "out_ablate/saturation@moderate/{i}/01.png",
    "contrast_mild": "out_ablate/contrast@mild/{i}/01.png",
    "noise_mild": "out_ablate/noise@mild/{i}/01.png",
    "noise_strong": "out_ablate/noise@strong/{i}/01.png",
    "posterize_strong": "out_ablate/posterize@strong/{i}/01.png",
    "hue_mod": "out_ablate/hue@moderate/{i}/01.png",
    "red_strong": "out_ablate/red_boost@strong/{i}/01.png",
    "chain_5": "out/naive/{i}/05.png",
    "chain_20": "out/naive/{i}/20.png",
}
CLEAN = "out_ablate/clean/{i}/00.png"
GOLD = "out_ablate/blur@strong/{i}/01.png"
PRACTICE = ["out_ablate/sharpen@moderate/{i}/01.png", "out_ablate/noise@moderate/{i}/01.png"]

out = Path("study/stimuli")
out.mkdir(parents=True, exist_ok=True)
ids = [f"{k:03d}" for k in random.Random(27).sample(range(90), 34)]
study, practice = ids[:32], ids[32:]


def sprite(img_id, templates):
    tiles = [np.asarray(Image.open(t.format(i=img_id)).convert("RGB")) for t in templates]
    h, w = tiles[0].shape[:2]
    assert all(t.shape == tiles[0].shape for t in tiles), img_id
    hp = -(-h // 8) * 8
    pad = lambda t: np.pad(t, ((0, hp - h), (0, 0), (0, 0)), mode="edge")
    Image.fromarray(np.vstack([pad(t) for t in tiles])).save(out / f"{img_id}.jpg", quality=95, subsampling=0)
    return {"id": img_id, "w": w, "h": h, "hp": hp}


manifest = {
    "tiles": ["clean", "blur_strong", *CONDS],
    "conds": list(CONDS),
    "study": [sprite(i, [CLEAN, GOLD, *CONDS.values()]) for i in study],
    "practice": [sprite(i, [CLEAN, *PRACTICE]) for i in practice],
}
(out / "manifest.json").write_text(json.dumps(manifest))
size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
print(f"{len(study)} study + {len(practice)} practice sprites, {size:.1f} MB -> {out}")

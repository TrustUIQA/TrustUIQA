"""Section-13 step 2: verify every benchmark metric loads and runs on one image.

Downloads pretrained weights on first run (~1-2 GB total, cached under
~/.cache/torch/hub/pyiqa). Run this on Colab/Kaggle, not a CPU laptop.

    python scripts/check_env.py
"""
import time

import numpy as np
import torch
from PIL import Image

from uwiqa.metrics.registry import DEFAULT_METRICS, MetricRunner

print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

rng = np.random.default_rng(0)
img = (rng.random((384, 512, 3)) * 255).astype(np.uint8)
img[..., 0] //= 3  # crude blue-green cast
path = "/tmp/_uwiqa_check.png" if not __import__("os").name == "nt" else "_uwiqa_check.png"
Image.fromarray(img).save(path)

runner = MetricRunner(DEFAULT_METRICS)
failed = []
for name in DEFAULT_METRICS:
    t0 = time.time()
    try:
        m = runner.get(name)
        s = m.fn(path)
        print(f"  ok  {name:14s} {s:9.4f}  higher_better={m.higher_better}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        failed.append(name)
        print(f"  FAIL {name:14s} {type(e).__name__}: {e}")
print("all metrics OK" if not failed else f"FAILED: {failed}")

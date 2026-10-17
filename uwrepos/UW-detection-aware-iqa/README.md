# Detection-Aware NR-IQA for Enhanced Underwater Images

Do quality metrics for enhanced underwater images (UIQM, UCIQE, modern deep NR-IQA) agree with humans, and can a learned no-reference predictor tell, **from the raw image alone**, whether enhancing it will help object detection?

- Human-judgement half (O1, O2): **UID2021** (960 images with MOS)
- Detection half (O3, O4): **RUOD**

Full context: [docs/HANDOFF.md](docs/HANDOFF.md). Plan and status: [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md). Data: [data/README.md](data/README.md).

## Run on Kaggle (recommended)
Import [`notebooks/01_kaggle_benchmark_uid2021.ipynb`](notebooks/01_kaggle_benchmark_uid2021.ipynb) into Kaggle (File → Import Notebook → GitHub). Set **GPU** and **Internet: On**, then run top to bottom. It clones this repo, downloads UID2021, scores all metrics, and prints the O1 table plus the Week-4 gate verdict.

## Run locally (CPU: hand-crafted metrics, tests, analysis)
```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Layout
```
src/uwiqa/metrics   UIQM, UCIQE (2 variants), pyiqa registry (13 NR metrics incl. URanker)
src/uwiqa/eval      SRCC / PLCC (4-param logistic) / KRCC, bootstrap CIs, paired tests, intra-scene SRCC
src/uwiqa/enhance   classical UIE (CLAHE, gray-world, UDCP, Ancuti fusion) + over-enhancement sweeps
src/uwiqa/data      manifest format, dataset builders, inspect_dataset()
src/uwiqa/detect    O3 (next)        src/uwiqa/model   O4
scripts/            check_env, build_manifests, score_metrics (resumable), benchmark, stress_test
notebooks/          Kaggle notebooks
```

Environment variables: `DATA_ROOT`, `RESULTS_ROOT`, `MANIFEST_ROOT`, and per-dataset `<NAME>_ROOT` (e.g. `UID2021_ROOT`).

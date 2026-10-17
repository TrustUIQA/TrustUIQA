"""Regenerate the Kaggle notebooks in notebooks/.

The notebooks are deliberately thin: one setup cell (pulls the latest code)
plus one-line commands. All logic lives in scripts/ and src/, so importing a
notebook into Kaggle once is enough; later fixes arrive by re-running cell 1.

    python scripts/make_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
CLONE = ("!test -d /kaggle/working/repo || git clone -q "
         "https://github.com/Banasri007/UW-detection-aware-iqa.git /kaggle/working/repo")
PERSIST = """
**Kaggle tips**
- Re-running cell 1 always pulls the latest code from GitHub. You never need to re-import this notebook.
- `/kaggle/working` is wiped when a session ends unless you commit (**Save Version → Save & Run All**)
  or enable **Settings → Persistence → Files**.
"""


def setup(extras=""):
    return f"{CLONE}\n%run /kaggle/working/repo/scripts/kaggle_setup.py {extras}".rstrip()


def nb(cells, accelerator):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": accelerator, "isInternetEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}


M, C = "markdown", "code"

NB01 = [
    (M, "# 01 — NR-IQA benchmark on UID2021 (O1 + O2)\n\n"
        "**Settings:** Accelerator **GPU T4 x2** or **P100**, Internet **On**.\n" + PERSIST),
    (C, setup("iqa")),
    (C, "# Download UID2021 (skips if already present or attached as a Kaggle Dataset)\n"
        "!python scripts/get_uid2021.py"),
    (C, "# Show image folders and every score table with its columns\n"
        "from uwiqa.data import dataset_root, inspect_dataset\n"
        "inspect_dataset(dataset_root(\"uid2021\"))"),
    (M, "**Fill in the three values below from the output above.** They are saved to the untracked "
        "`configs/local.yaml`, so code updates never overwrite them."),
    (C, "from uwiqa.data import set_local_config\n"
        "set_local_config(\"uid2021\",\n"
        "    table=\"TODO\",      # MOS file path relative to the dataset root\n"
        "    image_col=\"TODO\",  # column with image names\n"
        "    mos_col=\"TODO\")    # column with MOS\n"
        "!python scripts/build_manifests.py --only uid2021"),
    (C, "# Check every metric loads on the GPU (downloads pretrained weights once)\n"
        "!python scripts/check_env.py"),
    (C, "# Score all 960 images with every metric (resumable), ~30-60 min\n"
        "!python scripts/score_metrics.py --dataset uid2021"),
    (C, "# O1 table + Week-4 gate verdict\n!python scripts/benchmark.py --dataset uid2021"),
    (C, "# O2 over-enhancement stress test on the 60 raw images\n"
        "!python scripts/stress_test.py --dataset uid2021 --metrics uiqm uciqe topiq_nr liqe musiq uranker"),
    (C, "# Week-4 gate debugging: compare UIQM/UCIQE/NIQE with the UID2021 paper (Tables 8-9), check the\n"
        "# MOS workbook and image<->MOS alignment. Needs results/scores/uid2021.csv from the scoring cell.\n"
        "!python scripts/diagnose_uid2021.py"),
    (C, "# Does the original (2022, 52-observer) MOS reproduce the paper's numbers?\n"
        "!python scripts/compare_uid2021_mos.py"),
]

NB03 = [
    (M, "# 03 — Leakage-safe splits + reference detector (O3, part 1)\n\n"
        "**Settings:** Accelerator **GPU T4 x2** (DDP is used automatically), Internet **On**.\n"
        "**Input:** *Underwater Domain in ODverse33* (skycol).\n" + PERSIST),
    (C, setup("detect")),
    (C, "!python scripts/build_manifests.py --only ruod duo"),
    (M, "### 1. Duplicate threshold\n"
        "Notebook 02 showed that pairs at d=20 are still the same video scene, so we search wider (32/256) "
        "and look at the band just above 20. Hashes are cached, so this takes ~3 min if notebook 02's "
        "`results/dedup` is present, ~10 min otherwise."),
    (C, "!python scripts/find_duplicates.py --datasets ruod duo --max-dist 32"),
    (C, "%run scripts/show_pairs.py --min-dist 21 --max-dist 26 --leaky-only --n 6"),
    (C, "%run scripts/show_pairs.py --min-dist 27 --max-dist 32 --leaky-only --n 6"),
    (M, "If the 27–32 pairs are still the same scene, keep 32. If they are different scenes, "
        "re-run the duplicate cell with the largest distance that still looked like a duplicate."),
    (M, "### 2. Leakage-safe splits"),
    (C, "!python scripts/make_splits.py"),
    (M, "### 3. Reference detector (YOLO11-s on raw RUOD `det_train`)\n"
        "Run the smoke test first (~5 min). For the full run, use **Save Version → Save & Run All**: "
        "roughly 3–5 h on T4 x2. If it times out, attach this notebook's output as input, copy "
        "`results/detector` back to `/kaggle/working/results/`, and re-run: training resumes from `last.pt`."),
    (C, "# Smoke test: 5% of det_train, 2 epochs\n"
        "!python scripts/train_detector.py --epochs 2 --fraction 0.05 --name smoke"),
    (C, "# Full training + held-out evaluation on RUOD pred_test and duplicate-free DUO\n"
        "!python scripts/train_detector.py"),
]

NB04 = [
    (M, "# 04 — Per-image detection-utility labels (O3, part 2)\n\n"
        "**Settings:** Accelerator **GPU T4 x2**, Internet **On**.\n"
        "**Inputs (Add Input):**\n"
        "1. *Underwater Domain in ODverse33* (skycol)\n"
        "2. **Your Work → notebook 03** (the committed version with the trained detector). Cell 1 copies its "
        "`results/` (splits + `best.pt`) into `/kaggle/working/results`.\n\n"
        "For every held-out image: resize to 1280 px (long side), apply each enhancer in memory, run the frozen "
        "detector, and score per-image AP / AP50 / F1. `raw` is the baseline; `null_jpeg95` (an invisible JPEG "
        "re-encode) measures label noise. All detections are saved so selective-enhancement policies can be "
        "evaluated later without re-running the detector.\n" + PERSIST),
    (C, setup("detect")),
    (C, "# Official FUnIE-GAN code + PyTorch weights (28 MB, ship inside its GitHub repo)\n"
        "!python scripts/get_enhancers.py"),
    (C, "# Visual sanity check: raw vs every enhancer on 3 held-out images\n"
        "%run scripts/show_enhancers.py --n 3 --deep funiegan"),
    (M, "### Full labelling (~1.6 h on T4) — run with **Save Version → Save & Run All (Commit)**\n"
        "Measured speed: 0.5 s/image. Shards are resumable: if a commit is interrupted, attach that version's "
        "output as an extra input and commit again; finished shards are restored by cell 1 and skipped."),
    (C, "# RUOD pred_train + pred_val + pred_test (7,000 images)\n"
        "!python scripts/build_utility_labels.py --set ruod_pred --deep funiegan"),
    (C, "# Duplicate-free DUO (4,539 images), cross-dataset labels for O5\n"
        "!python scripts/build_utility_labels.py --set duo_clean --deep funiegan"),
]

NB05 = [
    (M, "# 05 — Detection-aware predictor + selective enhancement (O4, O5, headline result)\n\n"
        "**Settings:** Accelerator **GPU T4 x2**, Internet **On**.\n"
        "**Inputs (Add Input):**\n"
        "1. *Underwater Domain in ODverse33* (skycol)\n"
        "2. **Your Work → notebook 04** (committed version with the utility labels). It already contains "
        "notebook 03's splits and detector, so cell 1 restores everything.\n\n"
        "Run everything with **Save Version → Save & Run All (Commit)** (~2 h). Long steps are resumable.\n"
        + PERSIST),
    (C, setup("iqa")),
    (C, "!python scripts/get_enhancers.py"),
    (M, "### 1. Features (~1 h)\nFrozen CLIP / DINOv2 / ResNet-18 embeddings of each **raw** image, plus "
        "UIQM, UCIQE, TOPIQ-NR, LIQE and URanker scores of raw and every enhanced variant (for the baselines)."),
    (C, "!python scripts/extract_features.py --set ruod_pred --deep funiegan"),
    (C, "!python scripts/extract_features.py --set duo_clean --deep funiegan"),
    (M, "### 2. Predictor vs NR-IQA baselines (~15 min)\n"
        "Train on RUOD pred_train, tune on pred_val, report on pred_test and DUO (nothing tuned on DUO)."),
    (C, "!python scripts/train_predictor.py"),
    (M, "### 3. Selective enhancement: dataset mAP by policy (~30 min)"),
    (C, "!python scripts/evaluate_policies.py"),
]

NB06 = [
    (M, "# 06 — Policy re-tuning on dataset mAP + paper figures (CPU, ~30 min)\n\n"
        "**Settings:** Accelerator **None** (CPU), Internet **On**.\n"
        "**Inputs (Add Input):**\n"
        "1. *Underwater Domain in ODverse33* (skycol): ground-truth boxes\n"
        "2. **Your Work → notebook 05** (fcv-project-5, committed version): labels, features, predictor\n"
        "3. **Your Work → notebook 01** (fcv-project-1): UID2021 benchmark + stress test\n\n"
        "Run with **Save Version → Save & Run All (Commit)**, then download `paper_results.zip` from the "
        "version's **Output** tab.\n" + PERSIST),
    (C, setup()),
    (C, "# Selective enhancement with the margin tuned on validation *dataset mAP* (the metric we report)\n"
        "!python scripts/evaluate_policies.py --objective map"),
    (C, "# All paper figures (PDF) and LaTeX table rows\n!python scripts/make_figures.py"),
    (C, "!cd /kaggle/working/results && zip -qr /kaggle/working/paper_results.zip paper tables policies "
        "predictor/summary_test.csv predictor/summary_duo.csv predictor/per_method_test.csv "
        "predictor/per_method_duo.csv utility/ruod_pred/main/scores.csv utility/duo_clean/main/scores.csv "
        "stress/o2_uid2021_raw.csv stress/o2_uid2021_summary.csv && ls -lh /kaggle/working/paper_results.zip"),
]

if __name__ == "__main__":
    for name, cells, acc in [("01_kaggle_benchmark_uid2021.ipynb", NB01, "gpu"),
                             ("03_kaggle_splits_and_detector.ipynb", NB03, "gpu"),
                             ("04_kaggle_utility_labels.ipynb", NB04, "gpu"),
                             ("05_kaggle_predictor_and_policies.ipynb", NB05, "gpu"),
                             ("06_kaggle_policies_and_figures.ipynb", NB06, "none")]:
        p = NB_DIR / name
        json.dump(nb(cells, acc), open(p, "w", encoding="utf-8"), indent=1)
        print("wrote", p)

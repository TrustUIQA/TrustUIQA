# TrustUIQA

Code for **Do Underwater Image Quality Metrics Agree with Human Perception? A Benchmark with Human Choices, Simple Edits and Generative Drift** (WVAQ 2027).

Rishii Kumar Singh, Abhinav Rajput, Dinesh Kumar Vishwakarma
Artificial Intelligence and Machine Learning Society (AIMS)

Data (metric scores, result tables, study choices): https://huggingface.co/datasets/TrustUIQA/TrustUIQA

## Layout

| Path | What |
|---|---|
| `score.py`, `uwmetrics.py`, `vqr1_score.py`, `fr_score.py` | Score images with 73 NR-IQA metrics (pyiqa plus underwater metrics) |
| `attack.py`, `ablate.py`, `ablate_report.py`, `plot.py` | Iterated simple-edit attacks and the single-edit ablation |
| `banana_*.py` | Repeated Nano Banana edits (Vertex AI) and the drift analysis |
| `mos/` | Opinion-score sets: manifests (`prep.py`), scoring, within-scene kappa, simple edits, PGD attack, gMAD, flips and resizing, UCIQE implementations |
| `study/` | Paired-comparison study page (`rating.html`), stimuli lists, vote analysis |
| `paper/iteration5/scripts/` | Builds every table and number in the paper from the CSVs |
| `uwrepos/` | Code from other authors, unchanged, with their licenses |

## Data

Download the Hugging Face dataset into the repository root. Its folders (`mos/`, `out*/`, `study/`) match the paths the scripts read. The raw study sessions and the Nano Banana token log are not released, so `study/choices.csv` (one row per choice, random rater ids) replaces the sessions.

The images are not redistributed. Get UIEB, SAUD, SAUD2.0, UID2021 and SOTA from their authors, then run `mos/prep.py` to build the manifests and the scripts above to rebuild our edits, attacks and generated images.

## Third-party code

| Folder | Source license |
|---|---|
| `uwrepos/PSNR-SSIM-UCIQE-UIQM-Python` | MIT |
| `uwrepos/UnderwaterRanker` | CC BY-NC 4.0 (images and examples removed) |
| `uwrepos/RALI` | Apache 2.0 (code only, no checkpoints) |

Each folder keeps its original LICENSE. The other metric code we ran (UCIQE MATLAB, PyFADE, CCF, FDUM, NMFC, UIQAVSI, QUIQ, Twice-Mixing and others) has no license or does not allow redistribution, so it is not included. Clone it from its authors into `uwrepos/` under the same folder names that `mos/uw_extra.py` uses.

## Contact

trustuiqa@gmail.com

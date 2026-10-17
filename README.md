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

The dataset also holds the images:

- `datasets/`: the four opinion-score sets as released by their authors (extract each into `uwdata/<set>/x/`, then run `mos/prep.py`), and the 90 UIEB photos we use (copy to `data/uieb/`)
- `images/`: our simple edits, gradient attacks, iterated attacks, ablation, study images and every Nano Banana step

Cite the original papers of UIEB, SAUD, SAUD2.0, UID2021 and SOTA when you use their images.

## Third-party code

`uwrepos/` holds the metric code from other authors that we ran, unchanged, with each repository's LICENSE where its authors gave one. Files over 50 MB and PyTorch checkpoints (`.pth`, `.pt`, `.ckpt`, `.safetensors`, `.bin`) are not included. Get them from the original repositories.

## License

Our code (everything outside `uwrepos/`) is released under the MIT License, see `LICENSE`. Each folder in `uwrepos/` keeps the terms of its original authors. The data on Hugging Face has its own license, see its dataset card.

## Contact

trustuiqa@gmail.com

# Data

Nothing in this folder is committed. Two Kaggle-ready sources cover the whole project:

| Dataset | Role | Source | Notes |
|---|---|---|---|
| **UID2021** | O1 benchmark + O2 stress test (human MOS) | https://github.com/Hou-Guojia/UID2021 → Google Drive | 60 raw + 900 enhanced (15 methods), MOS from 77 observers. Non-commercial; cite Hou et al., ACM TOMM 2023. Notebook 01 downloads it. |
| **RUOD** | O3 utility labels + O4 predictor | Kaggle: [skycol/underwater-domain-in-odverse33](https://www.kaggle.com/datasets/skycol/underwater-domain-in-odverse33) → `Underwater/RUOD` | YOLO format, 10 classes. **Re-split by ODverse33**: test = 1,400 images (original RUOD test = 4,200). |
| **DUO** | O5 cross-dataset check | same Kaggle dataset → `Underwater/DUO` | YOLO format, 4 classes, 7,782 images (6,225 / 778 / 779 train/valid/test), matching the paper's total. |

The Kaggle dataset is ~7.0 GB, licensed CC BY-NC-SA 4.0 by the uploader; cite the original RUOD/DUO papers **and** state that you used the ODverse33 re-split. Run `notebooks/02_kaggle_check_detection_data.ipynb` once to confirm counts, classes and image resolution before training.

Optional: SAUD (https://github.com/yia-yuese/SAUD-Dataset) as a second MOS dataset.

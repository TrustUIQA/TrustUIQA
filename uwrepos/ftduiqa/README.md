# FTDUIQA — Fusion of Traditional and Deep Learning Models for Underwater Image Quality Assessment

> **Published paper:** *Fusion of Traditional and Deep Learning Models for Underwater Image Quality Assessment*
> **Authors:** Dhivya R M, V Nagasai, V. Masilamani
> **Affiliation:** Indian Institute of Information Technology Design and Manufacturing (IIITDM), Kancheepuram, Chennai, India

---

## Overview

Underwater images suffer from unique degradation challenges — light attenuation, color distortion, and backscatter — that make standard Image Quality Assessment (IQA) metrics unreliable. **FTDUIQA** is a hybrid ensemble model that fuses the quality scores of four deep learning models and one traditional machine learning model through **Support Vector Regression (SVR)**, achieving significantly higher correlation with human Mean Opinion Scores (MOS) than any individual constituent model.

```
[CNNIQA] ─┐
[DBCNN]  ─┤
[HyperIQA]─┤──► 1×5 feature vector ──► SVR ──► Final Quality Score
[MAUIQA]  ─┤
[MSAEQA]  ─┘
```

---

## Key Results

Evaluated on the **UID2021** dataset (960 underwater images) using PLCC, SROCC, and RMSE:

| Method | PLCC | SROCC | RMSE |
|---|---|---|---|
| MSAEQA (Hand-crafted) | 0.7258 | 0.7144 | 1.67 |
| MAUIQA (Modified DL) | 0.7558 | 0.7399 | 0.22 |
| HyperIQA | 0.7593 | 0.7446 | 1.49 |
| DBCNN | 0.7695 | 0.7630 | 1.38 |
| CNNIQA | 0.7997 | 0.7769 | 1.29 |
| **FTDUIQA (Proposed)** | **0.8654** | **0.8581** | **1.06** |

FTDUIQA outperforms all individual models in PLCC and SROCC, demonstrating the benefit of fusing complementary model representations.

---

## Repository Structure

```
uw-iqa/
├── config.py              # Centralized paths for datasets and models
├── SAUDDataset.py         # PyTorch Dataset loader for SAUD dataset
├── UIDDataset.py          # PyTorch Dataset loader for UID2021 dataset
│
├── CNNIQA/                # CNN-based IQA model (max-min pooling)
│   ├── model.py
│   ├── train_test_uid.py
│   ├── train_test_saud.py
│   ├── CNNIQA_UID_model   # Pre-trained weights (UID2021)
│   └── CNNIQA_SAUD_model  # Pre-trained weights (SAUD)
│
├── DBCNN/                 # Deep Bilinear CNN (VGG-16 + SCNN)
│   ├── model.py
│   ├── train_test_uid.py
│   ├── train_test_saud.py
│   ├── Pretrained SCNN.pkl
│   ├── DBCNN_UID_model
│   └── DBCNN_SAUD_model
│
├── HyperIQA/              # HyperNet-TargetNet architecture
│   ├── model.py
│   ├── train_test_uid.py
│   ├── train_test_saud.py
│   ├── HyperIQA_UID_model
│   └── HyperIQA_SAUD_model
│
├── MAUIQA/                # Multi-scale Attention for Underwater IQA (novel)
│   ├── model.py           # ResNet-50 backbone + channel attention
│   ├── train_test_uid.py
│   ├── train_test_saud.py
│   ├── MAUIQA_UID_model
│   └── MAUIQA_SAUD_model
│
├── MSAEQA/                # Hand-crafted feature model (SVR-based)
│   ├── feature_extractor.py   # Statistical, texture, color features
│   ├── feature_extraction_uid.py
│   ├── feature_extraction_saud.py
│   ├── train_test_uid.py
│   ├── train_test_saud.py
│   ├── features_uid.xlsx      # Pre-extracted features for UID2021
│   ├── features_saud.csv      # Pre-extracted features for SAUD
│   ├── MSAEQA_UID_model
│   └── MSAEQA_SAUD_model
│
└── FusionIQA/             # FTDUIQA ensemble fusion model
    ├── model.py            # Loads all 5 sub-models, extracts a 1×5 score vector
    ├── train_test_uid.py   # Trains LR + SVR fusion heads on UID2021
    ├── train_test_saud.py  # Trains LR + SVR fusion heads on SAUD
    ├── LRFusionIQA_UID_model
    └── SVRFusionIQA_UID_model
```

---

## Components

### Datasets

| Dataset | Description | Path (configured in `config.py`) |
|---|---|---|
| **UID2021** | 960 underwater images (60 raw + 900 enhanced via 15 algorithms), MOS annotations | `UID 2021/` |
| **SAUD** | Underwater image quality dataset with enhanced images, MOS annotations | `SAUD/Enhanced/` |

Both datasets require a `mos.xlsx` file in their respective directories containing `Name` and `MOS` columns.

### Models

#### CNNIQA
A compact CNN with a single convolutional layer followed by max-min pooling to jointly capture sharpness, contrast, and noise. Outputs a scalar quality score.

#### DBCNN
A Deep Bilinear CNN that fuses features from a pre-trained **VGG-16** and a custom **Scattering CNN (SCNN)** via bilinear pooling. Requires `Pretrained SCNN.pkl`.

#### HyperIQA
A two-network architecture: a **HyperNet** generates the weights for a **TargetNet** that processes image patches, enabling content-aware, local-distortion-sensitive quality prediction.

#### MAUIQA *(Novel — introduced in this paper)*
A multi-scale attention network based on a **ResNet-50** backbone. Features are extracted from three residual stages and passed through channel-wise **attention modules** before pooling and concatenation. Specially adapted for underwater-specific distortions.

Architecture summary:
- Layer 1 → AttentionModule(256 channels)
- Layer 2 → AttentionModule(512 channels)
- Layer 3 → AttentionModule(1024 channels)
- Concatenated → FC(512) → Dropout → FC(128) → FC(1)

#### MSAEQA *(Novel — introduced in this paper)*
A hand-crafted feature pipeline computing **198 features** across three categories:
- **Statistical (54 features):** Multi-scale chrominance analyzed via MSCN transforms, GGD and AGGD fitting over a 3-level Gaussian/Laplacian pyramid.
- **Texture & Structural (138 features):** Weighted LBP histograms and HOG descriptors on gamma-transformed Laplacian pyramid levels.
- **Statistical Distribution (6 features):** Weibull distribution fitting on SVD singular values in CIELab color space.

An **SVR (RBF kernel)** is trained on these features to predict MOS values.

#### FusionIQA (FTDUIQA)
Concatenates the scalar quality scores from all five models into a **1×5 feature vector**, then trains an SVR (or Linear Regression) fusion head to predict the final quality score.

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install scikit-learn scikit-image scipy
pip install opencv-python pillow pandas openpyxl
pip install joblib numpy
```

### Dataset Preparation

1. Place the UID2021 dataset under `UID 2021/` with `mos.xlsx` inside.
2. Place the SAUD dataset under `SAUD/Enhanced/` with `mos.xlsx` inside.
3. Paths are configured centrally in `config.py` — update if your directory layout differs.

---

## Training & Evaluation

Each sub-model can be trained and evaluated independently. Run scripts from the **project root** directory.

### Step 1 — Train individual deep learning models

```bash
# CNNIQA
python -m CNNIQA.train_test_uid        # UID2021
python -m CNNIQA.train_test_saud       # SAUD

# DBCNN
python -m DBCNN.train_test_uid
python -m DBCNN.train_test_saud

# HyperIQA
python -m HyperIQA.train_test_uid
python -m HyperIQA.train_test_saud

# MAUIQA (proposed)
python -m MAUIQA.train_test_uid
python -m MAUIQA.train_test_saud
```

Training configuration (MAUIQA/CNNIQA/DBCNN/HyperIQA):
- Image resize: 224×224
- Normalization: ImageNet mean/std
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: L1Loss
- Batch size: 32, Epochs: 50
- Split: 70% train / 10% val / 20% test

### Step 2 — Extract MSAEQA hand-crafted features

```bash
python -m MSAEQA.feature_extraction_uid    # saves features to MSAEQA/features_uid.xlsx
python -m MSAEQA.feature_extraction_saud   # saves features to MSAEQA/features_saud.csv
```

### Step 3 — Train MSAEQA SVR

```bash
python -m MSAEQA.train_test_uid
python -m MSAEQA.train_test_saud
```

### Step 4 — Train FusionIQA (FTDUIQA)

Requires all five sub-models to be trained first.

```bash
python -m FusionIQA.train_test_uid     # trains LR + SVR fusion heads on UID2021
python -m FusionIQA.train_test_saud    # trains LR + SVR fusion heads on SAUD
```

Outputs:
- `FusionIQA/LRFusionIQA_UID_model` — Linear Regression fusion
- `FusionIQA/SVRFusionIQA_UID_model` — **SVR fusion (proposed FTDUIQA)**

### Device Support

`config.py` auto-detects the best available device:
- Apple Silicon (MPS) → `mps`
- NVIDIA GPU → `cuda`
- Fallback → `cpu`

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **PLCC** | Pearson Linear Correlation Coefficient — measures linear correlation with MOS |
| **SROCC** | Spearman Rank-Order Correlation Coefficient — measures monotonic correlation |
| **RMSE** | Root Mean Square Error — measures absolute prediction error |

Higher PLCC/SROCC and lower RMSE indicate better alignment with human perception.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{dhivya2024fusion,
  title={Fusion of Traditional and Deep Learning Models for Underwater Image Quality Assessment},
  author={Dhivya, RM and Nagasai, V and Masilamani, V},
  booktitle={2024 IEEE 8th International Conference on Information and Communication Technology (CICT)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

---

## License

This repository contains code used for the experimentation described in the paper above. Please contact the authors for licensing or reuse inquiries.

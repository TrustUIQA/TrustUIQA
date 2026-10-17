# HANDOFF — Detection-Aware No-Reference Quality Assessment for Enhanced Underwater Images

**Audience:** Claude Code (or any agent/engineer picking up implementation).
**Status at handoff:** Topic selected, literature verified, proposal written and submitted as a .docx. **No code written yet.** This document is the complete context needed to start implementation from zero.
**Date:** 2026-10-01

---

## 0. TL;DR for the implementer

Build and evaluate a **no-reference (NR) image quality metric for underwater images that predicts downstream object-detection utility**, not just perceptual prettiness. Two halves:

1. **Benchmark half (low risk, must succeed):** Run ~10 NR-IQA metrics against ≥3 underwater subjective (MOS) datasets under one consistent protocol. Show UIQM/UCIQE correlate poorly with humans and that modern deep NR-IQA (TOPIQ, LIQE) does better.
2. **Novel half (the contribution):** Build per-image "did enhancement help detection?" labels on a detection dataset, then train a lightweight NR predictor (frozen CLIP/DINOv2 + MLP head) that predicts this from the **raw image alone**. Final payoff: per-image *selective enhancement* driven by that predictor beats both all-raw and all-enhanced on dataset mAP.

Hard constraints: **single GPU** (Colab Pro / Kaggle T4, P100 or L4), **one student**, **~14 weeks**. Freeze backbones and cache features — this is non-negotiable, not an optimization.

---

## 1. How we got here (decision trail)

Understanding *why* this topic was chosen matters, because it determines what counts as success.

### 1.1 Original brief
A computer vision course project in one of two domains: **aquatic/marine life** OR **cybersecurity** (biometrics — fingerprint, retina/iris, face anti-spoofing, deepfake detection, etc.).

### 1.2 Constraints established
- **Purpose:** course project **AND** research-relevant — the mentor suggested a research-flavoured topic so there's a path to continuing under him (paper / research assistantship). So: must be semester-sized *and* have a credible open problem.
- **Compute:** Colab Pro / Kaggle GPU only. Single T4/P100/L4, limited session hours. No multi-GPU, no pretraining runs.
- **Team:** solo student, ~3–4 months.
- **Student profile:** 3rd-year CSE (AI/ML specialization), MIT Manipal.

### 1.3 Domain comparison (why marine won)

| Criterion | Marine | Cybersecurity/Biometrics |
|---|---|---|
| Dataset access for solo student | Nearly all one-click free (Kaggle/GitHub/Zenodo) | **Many require signed institutional licences** (CASIA-Iris, IIT-Delhi Iris, OULU-NPU, SiW, most LivDet partitions) |
| Single-GPU feasibility | Excellent | Good, but DFDC is ~470 GB — video deepfake work is out |
| Field saturation | Medium (enhancement crowded; evaluation/metrics open) | High on the open datasets; open only on cross-domain protocols |
| Ease of a small novel contribution | High — documented open problems, few strong baselines | Medium — must beat strong cross-domain SOTA |
| First-paper realism for an undergrad | High (Marine Vision workshops, MaCVi, LifeCLEF working notes with DOI/DBLP indexing) | Lower (IJCB, TIFS, T-BIOM are competitive) |

**Decision:** Marine. The dataset licensing wall in biometrics was the deciding factor — a solo undergraduate frequently cannot legally obtain the key PAD datasets.

### 1.4 Shortlist that was generated (7 topics), and which was picked

1. "When does underwater enhancement help detection?" — selective enhancement policy
2. Open-set marine individual re-ID (whale/turtle/shark)
3. **→ Reliable no-reference quality assessment for enhanced underwater images (SELECTED)**
4. Long-tail plankton / fine-grained fish classification
5. Diffusion-era deepfake detection generalization (best biometrics option)
6. Cross-sensor fingerprint PAD / altered-fingerprint recognition
7. Morphing attack detection trained on synthetic data only

**Why #3 was chosen:** it is the most defensible (an evaluation/metrics contribution is publishable even with a modest result), and it **composes with #1** — the detection-aware metric *is* the gating mechanism that a selective-enhancement policy needs. If the mentor wants to extend into a thesis, #1 is the natural follow-on.

### 1.5 Topics explicitly rejected as bad first choices
- "Beat SOTA on UIEB PSNR/SSIM" — saturated (~27 dB), and the metric itself is contested.
- Full deepfake *video* detection on DFDC — infeasible on Colab.
- Iris/face PAD without a mentor's signed licence — data legally unobtainable.
- Copy-move/splice forgery on CASIA v2.0 — already reported at 99%+.
- Sonar sim-to-real — label-scarce, hardware-flavoured.
- Diffusion-model *training* for UIE — exceeds session limits. Fine-tuning only.
- Coral 3D semantic mapping / real-time AUV deployment — needs hardware we don't have.

---

## 2. The problem, precisely

### 2.1 Physical background (for context in code comments / writeup)
- **Wavelength-dependent attenuation:** red largely gone in upper ~10 m, orange by ~40 m, yellow before ~100 m → blue-green cast.
- **Light falloff:** ~45% of surface solar energy at 1 m, ~16% at 10 m, ~1% at 100 m.
- **Scattering:** forward scattering blurs edges; **backscatter** adds veiling haze killing contrast.
- Net: low contrast, colour cast, reduced visibility, non-uniform illumination — worsening with depth, turbidity, camera-to-object distance.

### 2.2 Why evaluation is the hard part
There is **no true ground-truth clean reference** underwater. You cannot synthetically degrade-and-recover like in denoising/super-resolution. UIEB's "references" are *quasi*-references picked by human pairwise voting among 12 candidate algorithm outputs — not physically clean images. So full-reference IQA (PSNR/SSIM/LPIPS) is mostly inapplicable, and the field defaults to two hand-crafted NR metrics.

### 2.3 The two metrics under attack

**UIQM** — Panetta, Gao & Agaian, *IEEE J. Oceanic Eng.* 41(3):541–551, 2016.
`UIQM = c1·UICM + c2·UISM + c3·UIConM` (colourfulness, sharpness, contrast). Commonly `c1=0.0282, c2=0.2953, c3=3.5753`.

**UCIQE** — Yang & Sowmya, *IEEE TIP* 24(12):6062–6071, 2015.
`UCIQE = 0.4680·σ_c + 0.2745·con_l + 0.2575·μ_s` in CIELab (chroma std, luminance contrast, mean saturation).

**Failure mode:** both are purely low-level and semantically blind. Over-saturated / over-contrasted output scores *higher*. The UIEB paper itself notes methods with severe reddish shift from excessive enhancement get higher UIQM, and that UCIQE favours unnaturally high contrast.

### 2.4 Problem statement (as written in the proposal)
> The de facto standard NR quality metrics for enhanced underwater images — UIQM and UCIQE — are hand-crafted linear combinations of low-level colourfulness, sharpness, contrast and saturation statistics. They correlate weakly with human perceptual judgement and can be inflated, or effectively gamed, by over-enhancement. Furthermore, no existing NR metric predicts whether enhancing a given underwater image will actually improve the performance of a downstream task such as object detection, even though downstream utility is what most operational underwater vision pipelines ultimately care about.

Two coupled questions:
1. How well do classical and modern learning-based NR-IQA models track human judgement of enhanced underwater images, measured consistently across multiple subjective datasets?
2. Can an NR metric be trained to predict, **per image**, whether enhancement will help or hurt object detection?

---

## 3. Research gap (four gaps, with the evidence behind each)

**Gap 1 — UIQM/UCIQE are unreliable and reward over-enhancement.**
Evidence: UIEB paper's own observations; **Li & Cavallaro, ICIP 2022** (arXiv:2207.05470) concluded that none of the evaluated no-reference measures satisfactorily rates the quality of enhanced underwater images.
⚠️ **Citation correction:** this paper is **ICIP 2022, not ICASSP**. Several secondary sources get this wrong. Verify page numbers on IEEE Xplore before submission.

**Gap 2 — Transfer of general-purpose deep NR-IQA to underwater is unproven at scale.**
MUSIQ, CLIP-IQA, TOPIQ, LIQE, MANIQA, TReS are trained on in-air distortions (compression, blur, noise) from LIVE/KonIQ-10k/FLIVE/KADID. Underwater degradation is physically different.
Evidence it *partially* works: **Dumic et al., Electronics 15(11):2412, 2026** (doi:10.3390/electronics15112412) ran an ITU-R ACR study on **132 UIEBD + 120 EUVP images** and found **TOPIQ_NR Spearman ≈ 0.80 on UIEBD** and **LIQE ≈ 0.87 on EUVP**, both beating UIQM/UCIQE. But: only 2 datasets, 4 UIE methods. Thin.
⚠️ MDPI metadata for this paper is internally inconsistent (listed *Electronics* 2026, vol 15, no 11, art 2412, online ~mid-2026). Cite the DOI and re-verify year/volume at submission time.

**Gap 3 — No metric predicts downstream task utility. ← this is the contribution's target**
- **Wang et al., *IEEE J. Oceanic Eng.* 2024** (arXiv:2311.18814), "Is Underwater Image Enhancement All Object Detectors Need?": 18 enhancement algorithms × 7 detectors = 126 enhanced models + 7 raw-trained = **133 total**, concluding enhancement generally does **not** universally help detection.
  ⚠️ An older project-page draft cites "13 algorithms / 98 models" — cite the published IEEE JOE figures (18/126/133).
- **Saleem, Awad, Paheding, Lucas, Havens & Esselman, *Remote Sensing* 17(2):185, 2025**: per-image analysis — most enhanced images perform equal or better individually; only a small percentage cause overall negative impact; **over-enhancement specifically degrades detection**.
- **Awad et al., *Journal of Imaging* 12(1):18, 2026** (doi:10.3390/jimaging12010018) — **the closest prior work.** Proposes a composite "Q-index", a per-image COCO-mAP protocol, and a **"mixed-set upper bound"** showing selective enhancement beats both all-raw and all-enhanced. Explicitly concludes traditional image quality metrics do not reliably predict detection performance, and calls for metrics considering image quality and detection simultaneously.

**Gap 4 — Underwater subjective datasets are fragmented.**
UWIQA, UID2021, SAUD, SAUD 2.0, LUIQD use different scales, observer counts, protocols. Published correlations are not comparable across studies. No unified cross-dataset NR-IQA benchmark exists.

### 3.1 ⚠️ HONEST NOVELTY POSITIONING — read this before writing any claim
"Quality that predicts downstream utility" is **not brand new**. It exists in Awad et al. (underwater) and more broadly in the **Video/Image Coding for Machines (VCM/FCM)** literature (Satisfied Machine Ratio; DT-JRD just-recognizable-difference; ML-CLIPSim; "IQA for Machines" databases), which repeatedly shows PSNR/SSIM/LPIPS fail to predict machine task performance.

**The defensible delta is narrow and specific:**
> No published work delivers a **standalone, learnable NR predictor whose explicit training target is the per-image detection-utility change caused by enhancement**, benchmarked jointly against human MOS across multiple underwater subjective datasets, with cross-dataset generalisation reported.

Differentiators vs. Awad et al.: (i) *learned predictor* rather than post-hoc oracle analysis; (ii) runs on the **raw image alone** (deployable — you don't have to enhance first to know whether to enhance); (iii) unified MOS + utility evaluation; (iv) cross-dataset generalisation.

**Do NOT claim "first task-aware quality metric."** Overclaiming will draw reviewer fire. Frame as above.

---

## 4. Objectives (as submitted)

| ID | Objective | Measurable criterion |
|---|---|---|
| **O1** | Unified NR-IQA benchmark | ≥10 metrics × ≥3 MOS datasets, one protocol, SRCC/PLCC/KRCC + bootstrap CIs |
| **O2** | Over-enhancement stress test | Controlled saturation/contrast/colour sweeps; quantify monotonic score inflation per metric |
| **O3** | Per-image detection-utility dataset | RUOD + ≥5 UIE methods; per-image utility delta under a fixed detector; labels released |
| **O4** | Detection-aware NR metric | SRCC/PLCC vs. true delta; balanced accuracy / F1 / AUROC on binary "will enhancement help?" |
| **O5** | Cross-dataset generalisation | Evaluate on held-out DUO + Brackish; report degradation honestly |
| **O6** *(optional)* | In-house MOS study | ITU-R BT.500 style, ~20 raters, ~100–150 images |

O1–O5 are core. **O6 is droppable** and should be dropped if behind schedule.

---

## 5. Datasets — exact specs and roles

### 5.1 Enhancement benchmarks
| Dataset | Content | Notes |
|---|---|---|
| **UIEB** | 890 paired + 60 "challenging" (no reference) | Quasi-reference chosen by **50-volunteer pairwise vote over 12 algorithms**. Li et al., *IEEE TIP* 29:4376–4389, 2020; arXiv:1901.05495 |
| **EUVP** | **~11,345 image pairs**; **515-image paired test set at 256×256**; plus unpaired subsets | Islam et al., *IEEE RA-L* 5(2):3227–3234, 2020 |
| **LSUI** | 4,279 paired images | Released with U-shape Transformer, *IEEE TIP* 32:3066–3079, 2023 |
| **U45** | 45 images | NR qualitative test set |
| **RUIE** | multi-subset | Liu et al., *IEEE TCSVT* 30(12):4861–4875, 2020 |
| **SQUID** | 57 stereo pairs | Berman et al., *IEEE TPAMI* 43(8):2822–2837, 2021 |

### 5.2 Subjective / MOS datasets (critical for O1)
| Dataset | Content | Protocol |
|---|---|---|
| **UID2021** | 60 raw + 900 enhanced (15 algorithms), ~1,060 total, 6 degradation scenes | Pair-comparison sorting, **52 observers**. Hou et al., *ACM TOMM* 2023 |
| **SAUD** | 100 raw + 1,000 enhanced (10 algorithms) | Subjective ranking/pairwise. Jiang et al., *IEEE TCSVT* 2022 (proposed NUIQ) |
| **SAUD 2.0** | 200 raw + 2,400 enhanced (2,600 total) | Multi-dimensional MOS (SS-ACR), 2024 |
| **UWIQA** | 890 raw (UIEB-derived) | MOS on ~10 discrete levels — **coarse scale, handle carefully** |
| **LUIQD** | 6,418 raw + 57,762 enhanced (64,180) | Scores 0–100. Large, recent |
| **UIEB voting** | 890 pairs | Pairwise voting → quasi-reference. **Not per-image MOS** |

⚠️ UIQD / USRD / UEIQA appear in the literature but were **only verified secondhand** — re-check before relying on or citing them.

### 5.3 Detection datasets (for O3/O5)
| Dataset | Content | Role |
|---|---|---|
| **RUOD** | **14,000 images, 74,903 labelled objects, 10 aquatic classes** (train 9,800 / val 4,200). Fu et al., *Neurocomputing* 517:243–256, 2023; github.com/dlut-dimt/RUOD | **Primary** — utility labels built here |
| **DUO** | **7,782 images** (6,671 train / 1,111 test), 74,515 instances, 4 classes: echinus, holothurian, starfish, scallop. Liu et al., ICMEW 2021; arXiv:2106.05681 | Cross-dataset generalisation |
| **Brackish** | **14,518 frames, 25,613 annotations, 6 classes** (big fish, crab, jellyfish, shrimp, small fish, starfish); captured 9 m deep in Limfjorden, Denmark; 89 video sequences. Pedersen et al., CVPRW 2019 | Cross-dataset generalisation (turbid) |
| **TrashCan 1.0** | marine debris | Optional robustness check |

⚠️ **RUOD count conflict:** the source paper says 14,000 images / 74,903 objects; a widely re-hosted Ultralytics copy lists 13,112 / 71,935. **Cite the source paper's figures.** Check which one you actually downloaded and note it.

⚠️ Some datasets (e.g. UVEB) host downloads on Baidu/Terabox — slow/awkward from India. Budget time; find mirrors.

### 5.4 MOS normalization policy
Normalize heterogeneous MOS to 0–1 **only for cross-dataset visualisation**. **Per-dataset, rank-based (Spearman/Kendall) analysis is the primary reporting mode** — pooling across incompatible scales (UWIQA's ~10 levels vs. continuous MOS) will mislead.

---

## 6. Methodology — implementation spec

### 6.1 UIE algorithms to generate enhanced variants (≥5, mixed families)
Use **pretrained public weights wherever possible** — do not train enhancers.

*Classical / physics-based:*
- Multi-scale fusion — Ancuti et al., *IEEE TIP* 27(1):379–393, 2018
- UDCP (Underwater Dark Channel Prior) — Drews Jr. et al., ICCVW 2013
- CLAHE / Retinex baseline
- (optional) Peng & Cosman blurriness + light absorption, *IEEE TIP* 26(4):1579–1594, 2017

*Deep:*
- **Water-Net** (released with UIEB)
- **FUnIE-GAN** (released with EUVP)
- **Ucolor** — Li et al., *IEEE TIP* 30:4985–5000, 2021
- One recent 2024–2026 method with public weights — e.g. **U-shape Transformer** or PixMamba

**Rationale for ≥5 spanning both families:** prevents the learned metric from keying on a single enhancer's artefact signature.

### 6.2 Metrics to benchmark
Use **`pyiqa` / IQA-PyTorch** (github.com/chaofengc/IQA-PyTorch) which ships pretrained: `niqe`, `brisque`, `musiq`, `clipiqa` / `clipiqa+`, `topiq_nr`, `liqe` / `liqe_mix`, `maniqa`, `tres`.

**UIQM and UCIQE are NOT core pyiqa NR entries** — implement from the original formulas (Section 2.3) and cross-check against a verified public reference implementation. This is a known source of silent bugs; validate that your values land in the published ranges before trusting anything downstream.

FR metrics (PSNR/SSIM/LPIPS) reported **only** on the UIEB quasi-reference subset, for context.

### 6.3 Correlation protocol (O1)
- **SRCC** (Spearman), **PLCC** (Pearson, computed **after the standard 4-parameter logistic fit** — don't skip the fit), **KRCC** (Kendall).
- Per-dataset primary; pooled secondary.
- **Bootstrap confidence intervals** on every correlation.

### 6.4 Over-enhancement stress test (O2)
Apply controlled sweeps (saturation ↑, contrast ↑, red-channel shift ↑) at increasing strength to a fixed image set. Plot each metric's score vs. distortion strength. A metric that rises monotonically while human judgement falls is "gameable" — quantify that.

### 6.5 Detection-utility labelling (O3) — the crux
1. Train **one** detector (YOLOv8-s or YOLOv11-s via Ultralytics) on **raw RUOD**. Freeze it. This is the fixed reference detector.
2. For each test image: run detection on raw and on each enhanced variant.
3. Per-image utility = **COCO-style per-image AP** against that image's GT boxes, OR a confidence-/IoU-weighted matched-detection score.
4. **Δ = score(enhanced) − score(raw)**
5. Labels: **Δ** (regression) and **sign(Δ)** (binary "does enhancement help?").

Report **two protocols:**
- (a) Frozen detector trained on raw only → isolates the pure preprocessing effect. **This is the primary.**
- (b) Sensitivity check with the detector retrained on enhanced data (note the extra compute).

Optional spot-check with a two-stage detector (Faster R-CNN).
⚠️ **Utility labels are detector-dependent.** State this limitation explicitly in the writeup.

### 6.6 The proposed metric (O4)
- **Input:** a single **raw** underwater image. (Critical design point: it predicts whether to enhance *before* enhancing.)
- **Backbone:** frozen **CLIP** image encoder or **DINOv2**. Cheap, cacheable, resists overfitting on a modest label set. Trainable **ResNet-18** as comparison baseline.
- **Heads:** lightweight MLP → predicted Δ (regression); parallel head → P(enhancement helps) (classification). Optionally condition on a UIE-method embedding.
- **Losses:** L1 or MSE (regression) + BCE (classification) + auxiliary **ranking / differentiable Spearman surrogate** to align orderings.
- **Optimizer:** AdamW, cosine schedule, early stopping. Standard augmentation.
- **Splits:** image-level with **scene-level separation to prevent leakage**, plus a fully separate cross-dataset test partition (DUO, Brackish).

### 6.7 Evaluation + ablations
- Metric vs. MOS across all subjective datasets (O1).
- Over-enhancement inflation curves (O2).
- Predictor: SRCC/PLCC of predicted vs. true Δ; balanced accuracy / F1 / AUROC on the binary decision.
- **Baselines:** UIQM, UCIQE, TOPIQ_NR, LIQE used *directly* as Δ predictors. The learned head must beat these or it adds nothing.
- Ablations: backbone (CLIP vs. DINOv2 vs. ResNet-18); head type; ranking loss on/off; number of UIE methods used to build labels; regression vs. classification.
- **Practical validation (the headline result):** per-image "enhance vs. keep raw" selection driven by the predictor, compared against all-raw and all-enhanced on dataset-level mAP. This is the NR-driven version of Awad et al.'s oracle mixed-set upper bound.

### 6.8 Stack
PyTorch · `pyiqa` (IQA-PyTorch) · Ultralytics (YOLO) · OpenCV + scikit-image (classical UIE, sweeps) · SciPy / statsmodels (correlations, logistic fit) · public UIE repos for pretrained weights.

### 6.9 Compute budget
Everything fits a single T4/P100/L4 on Colab Pro or Kaggle.
- Running pretrained NR metrics over a few thousand images: **inference only**, minutes–hours.
- Generating enhanced variants with pretrained UIE: **inference only**.
- Substantive costs: (a) training one YOLOv8/11-s on RUOD — a **few GPU-hours**; (b) training the lightweight head on **cached frozen features** — minutes to low hours.
- **Caching backbone features to disk is a hard requirement**, not an optimization. Session time limits will otherwise kill you.

---

## 7. Timeline with go/no-go gates

| Week | Milestone | Gate |
|---|---|---|
| 1 | Lit lock, env setup, dataset downloads (UIEB/EUVP/UID2021/SAUD/RUOD) | Data acquired; pyiqa operational |
| 2–3 | Implement + verify UIQM/UCIQE; run all NR metrics on MOS datasets | Metrics reproduce published value ranges |
| 4 | Correlation benchmark SRCC/PLCC/KRCC (O1) | **TOPIQ_NR SRCC ≳ 0.7 on ≥1 dataset.** If not, debug data/normalization before proceeding — everything rests on this |
| 5 | Over-enhancement stress test (O2) | Clear inflation curves |
| 6 | Generate enhanced RUOD with ≥5 UIE methods | Enhanced sets cached |
| 7 | Train reference detector; build per-image Δ labels (O3) | Δ distribution non-degenerate |
| 8 | **CHECKPOINT 1** — benchmark + labels done, metric design frozen | If behind, drop O6 |
| 9–10 | Train detection-aware predictor (O4) | Beats UIQM/UCIQE/TOPIQ_NR as a Δ predictor |
| 11 | Cross-dataset generalisation DUO/Brackish (O5) | Degradation quantified honestly |
| 12 | Ablations + selective-enhancement validation | Predictor-driven selection ≥ all-raw |
| 13 | **CHECKPOINT 2** — results frozen; optional MOS study if time | Result tables final |
| 14 | Write-up, figures, code + label release | Draft paper + public repo |

### 7.1 Pivot rules (important)
- **Week 4 gate fails** → debug data/normalization. Do not build the novel half on a broken benchmark.
- **Week 8, Δ distribution degenerate** (enhancement changes detection on <~10% of images, or Δ is pure noise) → **reframe**: make the benchmark + over-enhancement stress test the primary contribution, demote the detection-aware head to exploratory. This is still publishable.
- **Week 10, binary head no better than majority-class baseline** → stop escalating to regression/ranking; report the negative result rigorously. Negative results are publishable in *this specific debate* (Wang et al. and Awad et al. both published essentially cautionary findings).
- **Always drop O6 before dropping anything else.**

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Deep NR-IQA transfers poorly to underwater | Med | Med | Report as a finding; fine-tune a light head on underwater MOS |
| Detection-utility signal too weak/noisy | Med | **High** | Reduce to binary help/hurt; aggregate across UIE methods; report negative result rigorously |
| Awad et al. too close for a novelty claim | Med | Med | Lean on the four differentiators in §3.1; emphasise *learned, raw-image-only, cross-dataset* |
| Compute/session limits exceeded | Med | Med | Freeze backbones, cache features, use `-s` detector variants |
| MOS dataset access/licensing | Low | Med | UID2021 + SAUD are public; in-house study as fallback |
| Heterogeneous MOS scales confound pooling | Med | Low | Per-dataset rank-based analysis as primary |
| Utility labels detector-dependent | Med | Med | State explicitly; spot-check with a second detector |

---

## 9. Deliverables

1. Unified, reproducible cross-dataset NR-IQA benchmark (code + result tables).
2. Quantified evidence on UIQM/UCIQE unreliability and on deep NR-IQA transfer to underwater.
3. **Publicly released per-image detection-utility label set derived from RUOD** — this artifact alone has standalone value.
4. Trained detection-aware NR predictor + full evaluation, including the selective-enhancement demonstration.
5. Workshop-length manuscript + public repo.

---

## 10. Publication targets

**First paper (realistic):**
- **MaCVi** (Maritime Computer Vision) — WACV 2023–2025, **4th edition at CVPR 2026**. MaCVi 2025 ran an underwater image-restoration challenge; MaCVi 2026 tracks skewed surface/thermal/segmentation. ⚠️ **Re-check track fit each year before committing.**
- CVPR / ICCV / ECCV **Marine Vision** workshops (joint CVAUI + AAMVEM).

**Extended version:** *IET Image Processing*, *Journal of Imaging* (MDPI), *Journal of Marine Science and Engineering*.

---

## 11. Reference list (36, IEEE style, verification status noted)

1. K. Panetta, C. Gao and S. Agaian, "Human-visual-system-inspired underwater image quality measures," *IEEE J. Oceanic Eng.*, vol. 41, no. 3, pp. 541–551, 2016. doi:10.1109/JOE.2015.2469915
2. M. Yang and A. Sowmya, "An underwater color image quality evaluation metric," *IEEE TIP*, vol. 24, no. 12, pp. 6062–6071, 2015. doi:10.1109/TIP.2015.2491020
3. C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong and D. Tao, "An underwater image enhancement benchmark dataset and beyond," *IEEE TIP*, vol. 29, pp. 4376–4389, 2020. arXiv:1901.05495
4. M. J. Islam, Y. Xia and J. Sattar, "Fast underwater image enhancement for improved visual perception," *IEEE RA-L*, vol. 5, no. 2, pp. 3227–3234, 2020.
5. C. Li, S. Anwar, J. Hou, R. Cong, C. Guo and W. Ren, "Underwater image enhancement via medium transmission-guided multi-color space embedding," *IEEE TIP*, vol. 30, pp. 4985–5000, 2021.
6. L. Peng, C. Zhu and L. Bian, "U-shape transformer for underwater image enhancement," *IEEE TIP*, vol. 32, pp. 3066–3079, 2023. ⚠️ *verified secondhand — re-check volume/pages*
7. C. O. Ancuti, C. Ancuti, C. De Vleeschouwer and P. Bekaert, "Color balance and fusion for underwater image enhancement," *IEEE TIP*, vol. 27, no. 1, pp. 379–393, 2018.
8. P. Drews Jr., E. do Nascimento, F. Moraes, S. Botelho and M. Campos, "Transmission estimation in underwater single images," in *Proc. ICCVW*, 2013, pp. 825–830.
9. Y.-T. Peng and P. C. Cosman, "Underwater image restoration based on image blurriness and light absorption," *IEEE TIP*, vol. 26, no. 4, pp. 1579–1594, 2017.
10. C. Li and A. Cavallaro, "On the limits of perceptual quality measures for enhanced underwater images," in *Proc. IEEE ICIP*, 2022. arXiv:2207.05470 ⚠️ *ICIP, not ICASSP — verify pages*
11. E. Dumic et al., "Towards reliable evaluation of underwater image enhancement using subjective and objective analysis," *Electronics*, vol. 15, no. 11, art. 2412, 2026. doi:10.3390/electronics15112412 ⚠️ *metadata inconsistent — re-verify year/volume*
12. H. Wang et al., "Is underwater image enhancement all object detectors need?," *IEEE J. Oceanic Eng.*, 2024. arXiv:2311.18814
13. A. Saleem, A. Awad, S. Paheding, A. Lucas, T. Havens and P. Esselman, "Understanding the influence of image enhancement on underwater object detection: a quantitative and qualitative study," *Remote Sensing*, vol. 17, no. 2, art. 185, 2025. doi:10.3390/rs17020185
14. A. Awad et al., "Revisiting underwater image enhancement for object detection: a unified quality-detection evaluation framework," *Journal of Imaging*, vol. 12, no. 1, art. 18, 2026. doi:10.3390/jimaging12010018
15. A. Mittal, A. K. Moorthy and A. C. Bovik, "No-reference image quality assessment in the spatial domain," *IEEE TIP*, vol. 21, no. 12, pp. 4695–4708, 2012.
16. A. Mittal, R. Soundararajan and A. C. Bovik, "Making a completely blind image quality analyzer," *IEEE SPL*, vol. 20, no. 3, pp. 209–212, 2013.
17. J. Ke, Q. Wang, Y. Wang, P. Milanfar and F. Yang, "MUSIQ: multi-scale image quality transformer," in *Proc. ICCV*, 2021, pp. 5148–5157.
18. J. Wang, K. C. K. Chan and C. C. Loy, "Exploring CLIP for assessing the look and feel of images," in *Proc. AAAI*, 2023.
19. C. Chen, J. Mo, J. Hou, H. Wu, L. Liao, W. Sun, Q. Yan and W. Lin, "TOPIQ: a top-down approach from semantics to distortions for image quality assessment," *IEEE TIP*, 2024. arXiv:2308.03060
20. W. Zhang, G. Zhai, Y. Wei, X. Yang and K. Ma, "Blind image quality assessment via vision-language correspondence: a multitask learning perspective," in *Proc. CVPR*, 2023.
21. S. Yang, T. Wu, S. Shi, S. Lao, Y. Gong, M. Cao, J. Wang and Y. Yang, "MANIQA: multi-dimension attention network for no-reference image quality assessment," in *Proc. CVPRW*, 2022.
22. S. A. Golestaneh, S. Dadsetan and K. M. Kitani, "No-reference image quality assessment via transformers, relative ranking, and self-consistency," in *Proc. WACV*, 2022.
23. G. Hou, Y. Li, H. Yang, K. Li and Z. Pan, "UID2021: an underwater image dataset for evaluation of no-reference quality assessment metrics," *ACM TOMM*, 2023.
24. Q. Jiang, Y. Gu, C. Li, R. Cong and F. Shao, "Underwater image enhancement quality evaluation: benchmark dataset and objective metric," *IEEE TCSVT*, 2022. ⚠️ *verified secondhand*
25. C. Fu, R. Liu, X. Fan, P. Chen, H. Fu, W. Yuan, M. Zhu and Z. Luo, "Rethinking general underwater object detection: datasets, challenges and solutions," *Neurocomputing*, vol. 517, pp. 243–256, 2023. ⚠️ *verified secondhand*
26. C. Liu, H. Li, S. Wang, M. Zhu, D. Wang, X. Fan and Z. Wang, "A dataset and benchmark of underwater object detection for robot picking," in *Proc. IEEE ICMEW*, 2021. arXiv:2106.05681
27. M. Pedersen, J. B. Haurum, R. Gade, T. B. Moeslund and N. Madsen, "Detection of marine animals in a new underwater dataset with varying visibility," in *Proc. CVPRW*, 2019.
28. J. Hong, M. Fulton and J. Sattar, "TrashCan: a semantically-segmented dataset towards visual detection of marine debris," arXiv:2007.08097, 2020.
29. R. Liu, X. Fan, M. Zhu, M. Hou and Z. Luo, "Real-world underwater enhancement: challenges, benchmarks and solutions under natural light," *IEEE TCSVT*, vol. 30, no. 12, pp. 4861–4875, 2020. ⚠️ *verified secondhand*
30. D. Berman, D. Levy, S. Avidan and T. Treibitz, "Underwater single image color restoration using haze-lines and a new quantitative dataset," *IEEE TPAMI*, vol. 43, no. 8, pp. 2822–2837, 2021.
31. C. Chen and J. Mo, "IQA-PyTorch: PyTorch toolbox for image quality assessment," github.com/chaofengc/IQA-PyTorch
32. G. Jocher, A. Chaurasia and J. Qiu, "Ultralytics YOLO," github.com/ultralytics/ultralytics
33. A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021.
34. M. Oquab et al., "DINOv2: learning robust visual features without supervision," *TMLR*, 2024. arXiv:2304.07193
35. ITU-R, "Methodology for the subjective assessment of the quality of television pictures," Rec. ITU-R BT.500-14, 2019.
36. B. Kiefer et al., "3rd workshop on maritime computer vision (MaCVi) 2025: challenge results," in *Proc. WACVW*, 2025. arXiv:2501.10343

⚠️ Also to verify if used: the Satisfied Machine Ratio / VCM references (verified secondhand only).

---

## 12. Artifacts already produced

- **`UIQA_Project_Proposal.docx`** — formal submission-ready proposal with exactly six sections: Introduction, Problem Statement, Research Gap, Objectives, Methodology, References. Formatted to match the department's PCAP synopsis template: **Times New Roman, 12 pt body, bold 16 pt centred title, bold 14 pt section headings, bold 13 pt subheadings, 1.15 line spacing (`w:line=276`), A4, 1″ margins**, no coloured text.
- No code, no notebooks, no downloaded data yet.

---

## 13. Suggested first actions for Claude Code

1. Scaffold the repo: `data/`, `src/{enhance,metrics,detect,model}/`, `configs/`, `notebooks/`, `results/`.
2. Set up the environment and verify `pyiqa` loads `topiq_nr`, `liqe`, `musiq`, `clipiqa`, `maniqa`, `tres`, `niqe`, `brisque` with pretrained weights on a single GPU.
3. Implement UIQM and UCIQE from the formulas in §2.3; write unit tests that check values land in published ranges on a handful of UIEB images.
4. Write dataset loaders for UIEB and EUVP first (easiest), then UID2021 and SAUD.
5. Build the correlation harness (SRCC / PLCC-with-logistic-fit / KRCC + bootstrap CI) as a reusable module — it's used in O1, O4 and the ablations.
6. **Do not start the detection half until the Week-4 gate passes.**

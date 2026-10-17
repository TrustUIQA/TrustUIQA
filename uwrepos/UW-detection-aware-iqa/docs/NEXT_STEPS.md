# Next Steps & Proposed Improvements

Status date: 2026-10-01. Week 1 of 14.

## Done (handoff §13, steps 1–5)
- [x] Repo scaffold, installable package `uwiqa`, git initialised
- [x] `pyiqa` 0.1.16 installs; registry wraps 13 NR metrics + UIQM/UCIQE (weights not yet downloaded — run `scripts/check_env.py` on Kaggle)
- [x] UIQM: vectorised; tested against a loop port of the reference implementation (exact match)
- [x] UCIQE: `paper` and `legacy_cv2` variants (ports disagree — this is itself reportable)
- [x] Manifest format + UIEB/EUVP builders + generic MOS-table builder
- [x] Correlation harness: SRCC / PLCC (4-param logistic) / KRCC, bootstrap CIs, paired bootstrap test, intra-scene SRCC
- [x] Classical UIE (CLAHE, gray-world, UDCP, fusion) and O2 sweeps (saturation, contrast, red shift, unsharp)
- [x] 41 unit tests passing
- [x] Kaggle notebook `notebooks/01_kaggle_benchmark_uid2021.ipynb`

## Scope decision (2026-10-01): two datasets, Kaggle only
No single dataset has both human quality scores and detection boxes, so the minimum is one of each:
- **UID2021** for O1 + O2 (its 60 raw images also serve the stress test, so UIEB/EUVP are dropped)
- **RUOD** for O3 + O4

Consequences, stated honestly in the write-up:
- O1 becomes a single-dataset benchmark. Gap 4 ("fragmented subjective datasets") is no longer addressed; drop it from the claims or add SAUD later (~1 h of Kaggle inference).
- ~~O5 becomes cross-split only~~ → **restored**: the Kaggle ODverse33 copy ships RUOD *and* DUO in YOLO format, so the cross-dataset check costs no extra download. DUO's 4 classes (echinus, holothurian, starfish, scallop) correspond to RUOD's seaurchin / seacucumber / starfish / scallop.
- RUOD is the **ODverse33 re-split** (test = 1,400, not 4,200). Fine for our purpose; state it in the paper and don't compare mAP against numbers on the original split.

## Immediate next steps (you)
1. Import the notebook into Kaggle (GPU, Internet on), run cells 1–4.
2. From the `inspect_dataset` output, fill the three values in cell 5.
3. Run the rest: `check_env`, scoring, benchmark (prints the **Week-4 gate**: TOPIQ_NR SRCC ≳ 0.7), stress test.
4. Attach Kaggle dataset `skycol/underwater-domain-in-odverse33` and run notebook 02 (CPU) to confirm RUOD/DUO counts, classes, resolution and JPEG quality.

## Detection data — verified on Kaggle (notebook 02, 2026-10-01)
| | images | objects | split (train/val/test) | notes |
|---|---|---|---|---|
| RUOD | 14,000 | 74,904 (paper 74,903) | 11,200 / 1,400 / 1,400 | complete dataset, ODverse33 80/10/10 re-split; 10 classes, 0 unlabeled images |
| DUO | 7,782 | 74,515 (= paper) | 6,225 / 778 / 779 | 65 unlabeled images; classes holothurian, echinus, scallop, starfish |

- **DUO class ids 0–3 = RUOD class ids 0–3** (same names, same order), so a RUOD-trained detector evaluates on DUO directly by restricting to classes 0–3.
- Images are at native resolution (720×405 up to 3840×2160), not resized. JPEG quality: RUOD ≈ q85 (mean luma quant 16.1), DUO ≈ q95 (5.8).
- The uploader's DUO `data.yaml` has a stray line and Windows paths; `train_detector.py` writes its own.

### Near-duplicate check (dHash 256-bit, threshold 20/256)
- 5,905 near-duplicate pairs: **3,158 RUOD↔DUO**, 2,580 within RUOD (**965 cross the official ODverse33 splits**), 167 within DUO.
- Visual check: d=0 pairs are identical frames; even d=20 pairs are adjacent frames of the same video. The threshold is conservative, so notebook 03 widens it to 32 and inspects the 21–32 band.
- 16,843 groups, 4,240 with more than one image, largest 66.
- **Consequence: the official splits leak and DUO is not independent of RUOD.** We re-split RUOD by duplicate group and evaluate DUO only on images sharing no group with RUOD (`make_splits.py`). Say this in the paper; it is a useful finding in its own right.

## O3 protocol
1. **Leakage-safe RUOD re-split by duplicate group** (`make_splits.py`, fractions in `uwiqa/data/splits.py`):
   `det_train` 45% / `det_val` 5% train the frozen reference detector (YOLO11-s); `pred_train` 30% / `pred_val` 10% / `pred_test` 10% are never seen by the detector.
2. **Utility labels on pred_\*** (≈ 7,000 images). The predictor trains on pred_train, is tuned on pred_val, and is reported on pred_test. DUO-clean is the cross-dataset test (O5).
3. **Enhance on the fly, never store enhanced images.** The predictor only sees the raw image, so we only need Δ: enhance → detect → score → discard. Cache per-image detections instead (a few MB). This keeps us under Kaggle's 20 GB `/kaggle/working` limit (storing 7,000 × 5 enhanced 4K PNGs would not fit).
4. Enhanced outputs stay in memory, so no extra JPEG generation. Enhancers will amplify RUOD's q≈85 block artefacts; that is part of the real effect, so leave the inputs as they are.
5. Null-enhancement noise floor: also score a JPEG-q95 re-encode of each raw image to estimate label noise (improvement #4 below).

## Reference detector — trained (notebook 03 commit, 2026-10-01)
YOLO11-s, 640 px, 100 epochs on `det_train` (6,300 images), 1.97 h on 2×T4; converged (val flat over the last epochs).

| set | images | mAP50 | mAP50-95 |
|---|---|---|---|
| det_val | 700 | 0.811 | 0.569 |
| RUOD pred_test (held out) | 1,400 | 0.798 | 0.556 |
| DUO clean (cross-dataset, classes 0–3) | 4,539 | 0.639 | 0.387 |

- Held-out ≈ val, so no overfitting through the split. Weakest RUOD classes: corals (0.56 mAP50) and jellyfish (0.59).
- DUO drop is mostly echinus recall (0.54; ~9 small urchins per image). Precision stays high (0.96).

## O3 labelling — implemented (notebook 04)
- `scripts/build_utility_labels.py`: per image, enhancers applied in memory at 1280 px, then the frozen detector; per-image AP / AP50 / F1 for `raw`, `null_jpeg95`, clahe, gray_world, udcp, fusion and FUnIE-GAN. All detections are saved for offline policy evaluation.
- `uwiqa.detect.metrics`: COCO-style per-image AP and dataset mAP. Verified against Ultralytics on identical predictions (Δ ≤ 0.005 mAP; the residual is the interpolation scheme).
- Deep enhancers: FUnIE-GAN uses the official repo code and weights. U-shape Transformer and PUIE-Net (PyTorch, Google Drive weights) are possible additions. Water-Net and Ucolor are TF1-only, so they were dropped as infeasible.

### Smoke test (40 RUOD pred_test images, 2026-10-02)
- 0.50 s/image on T4 → ~1 h for ruod_pred (7,000) + ~40 min for duo_clean (4,539).
- Raw dataset mAP at 1280 px work size: 0.794 / 0.563 (native-resolution Ultralytics eval: 0.798 / 0.556), so the resizing protocol does not change the detector's behaviour.
- **Noise floor is small:** null_jpeg95 Δap = −0.001 ± 0.033.
- **Every enhancer hurts on average** (mean Δap: clahe −0.05, udcp −0.07, funiegan −0.10, fusion −0.13, gray_world −0.14), **but each helps on 12–22% of images.** The per-image signal is non-degenerate (the Week-8 pivot rule is not triggered), and blanket enhancement is worse than none. This matches Wang et al. 2024 and Saleem et al. 2025.
- Caveat for the write-up: the detector was trained on raw images, so enhanced inputs are slightly out-of-distribution for it (protocol (a) in the handoff). Protocol (b), a detector trained with enhancement augmentation, is the sensitivity check.

### Full labels (notebook 04 commit, 2026-10-02): 0.24 s/image on T4

Mean per-image AP50-95 and Δ vs raw. "helps"/"hurts" here means Δ>0 / Δ<0, **before** the noise threshold.

| method | RUOD (7,000) ap | Δap | helps | hurts | DUO-clean (4,539) ap | Δap | helps | hurts |
|---|---|---|---|---|---|---|---|---|
| raw | 0.673 | — | — | — | 0.531 | — | — | — |
| null_jpeg95 | 0.672 | −0.001 | 21% | 22% | 0.530 | −0.001 | 24% | 29% |
| clahe | 0.651 | −0.022 | 27% | 50% | 0.486 | −0.046 | 33% | 64% |
| udcp | 0.631 | −0.041 | 24% | 54% | 0.453 | −0.078 | 26% | 71% |
| funiegan | 0.607 | −0.066 | 21% | 61% | 0.424 | −0.107 | 23% | 75% |
| gray_world | 0.597 | −0.075 | 20% | 60% | 0.437 | −0.094 | 26% | 72% |
| fusion | 0.589 | −0.083 | 20% | 62% | 0.332 | −0.199 | 14% | 84% |

- The raw dataset mAP computed from saved detections (RUOD 0.799/0.561, DUO 0.644/0.402) matches the Ultralytics held-out evals, so the labels are trustworthy.
- **Every enhancer lowers mean detection AP; the drop is larger on DUO** (out of distribution for the detector).
- **Key caveat:** the invisible null re-encode "helps" 21–24% of images, the same order as the enhancers. Sign(Δ) alone is mostly noise, so the predictor uses `helps = Δ > τ`, with τ the 95th percentile of |Δ_null| on pred_train. Policy evaluation includes a `noise_oracle` control (best of raw vs null per image) to show how much per-image "oracle" gain is pure noise.

## O4/O5: predictor + selective enhancement (notebook 05)
- `extract_features.py`: frozen CLIP-B/32, DINOv2-S/14 and ResNet-18 embeddings of the raw image; UIQM, UCIQE, TOPIQ-NR, LIQE and URanker on raw and on every enhanced variant.
- `train_predictor.py`: ridge, logistic and multi-task MLP (Δ regression + help classification + ranking loss; no-rank ablation; 3-seed ensemble) per backbone. Baselines: `iqa_raw-k` (−Q(raw)) and `iqa_delta-k` (Q(enh) − Q(raw)). Trained on pred_train, tuned on pred_val, reported on pred_test and DUO.
- `evaluate_policies.py`: dataset mAP for all_raw, all_m, oracle, noise_oracle, learned and IQA-driven policies, with margins tuned on pred_val. The best learned policy is chosen on val, then paired-bootstrapped against all_raw and the best fixed enhancer.

### Results (notebook 05 commit, 2026-10-02)
**O4: predicting the utility delta from the raw image** (mean over 5 enhancers; τ = 0.050; helps-rate ≈ 11% on RUOD, 9–19% on DUO)

| predictor | RUOD test SRCC | AUROC | DUO SRCC | AUROC |
|---|---|---|---|---|
| mlp@clip_b32 (best SRCC) | **0.319** | 0.597 | 0.174 | 0.597 |
| mlp@clip_b32+dinov2_s14 | 0.314 | 0.598 | 0.172 | **0.602** |
| ridge@clip_b32 | 0.315 | 0.519 | 0.154 | 0.483 |
| best IQA, quality gain (iqa_delta-uciqe) | 0.133 | 0.532 | −0.015 | 0.442 |
| IQA raw quality (iqa_raw-liqe / topiq_nr) | −0.235 / −0.212 | 0.513 / 0.528 | −0.026 / −0.020 | 0.598 / 0.593 |

- Learned raw-only predictors capture real signal (SRCC ≈ 0.31 in-domain, ≈ 0.17 cross-dataset), well above every NR-IQA baseline.
- **NR-IQA "quality gain" does not predict detection gain** (SRCC −0.18 … 0.13; AUROC ≈ 0.44–0.53). This holds for UIQM/UCIQE and for the deep metrics, including the underwater-specific URanker.
- **Raw-image quality correlates the "wrong" way** (SRCC ≈ −0.22 for TOPIQ/LIQE/URanker): enhancement hurts *more* on images that already look poor.
- Backbone and ranking-loss ablations barely matter (SRCC 0.30–0.32). Ridge ranks as well as the MLP but classifies worse.

**O5: selective enhancement (dataset mAP50-95)**

| policy | RUOD test | Δ vs raw | DUO clean | Δ vs raw |
|---|---|---|---|---|
| all_raw | 0.5548 | — | 0.4020 | — |
| oracle (upper bound) | 0.5801 | +0.025 | 0.4233 | +0.021 |
| noise_oracle (control) | 0.5572 | +0.002 | 0.4036 | +0.002 |
| best learned, chosen on val (ridge@dinov2_s14) | 0.5551 | +0.0003 [−0.0002, 0.0007] | 0.4020 | 0.000 |
| best fixed enhancer | 0.5299 (clahe) | −0.025 | 0.3420 (udcp) | −0.060 |
| IQA-driven, naive (best quality gain) | 0.465–0.509 | −0.046 … −0.089 | 0.228–0.336 | −0.066 … −0.174 |

- **Headroom is real** (oracle ≈ 10× the noise control), **but the learned policy does not convert it into mAP.** With margins tuned on val, it enhances 0–3% of images and ties all_raw. Selective enhancement beats every *always-enhance* policy by 0.025–0.06 mAP, because it learns to almost never enhance.
- **Choosing the enhancer by NR-IQA, as common practice does, costs 5–17 mAP points.** This is the strongest practical result.
- This is the handoff's "negative result, report rigorously" branch for the policy, while O4 is a positive ranking result over the IQA baselines.
- **Re-tuned on validation dataset mAP (notebook 06, 2026-10-03):** the val-selected policy (logreg@clip_b32|p) enhances 3.9% of images and *loses* 0.0017 mAP on RUOD (CI [−0.0037, −0.0004]) and 0.0010 on DUO (CI [−0.0018, −0.0003]). The best NR-IQA policy (iqa_delta-liqe) ties raw (−0.0002, CI includes 0). The val-optimal margin does not transfer, so the negative policy result is robust to the tuning objective. Paper Table V reports both objectives.

## O1/O2 on UID2021 (notebook 01, 2026-10-02): 960 images, MOS from 77 observers
| metric | SRCC [95% CI] | intra-scene SRCC |
|---|---|---|
| uranker (underwater-trained) | **0.623** [0.581, 0.665] | **0.578** |
| liqe_mix / liqe | 0.484 / 0.482 | 0.381 / 0.401 |
| musiq / arniqa | 0.403 / 0.388 | 0.303 / 0.289 |
| uciqe / uiqm | 0.334 / 0.325 | 0.351 / 0.293 |
| topiq_nr | 0.316 | 0.167 |
| clipiqa+ / tres / qualiclip / maniqa / clipiqa | 0.30 / 0.29 / 0.25 / 0.21 / 0.19 | 0.07 / 0.07 / 0.02 / −0.09 / 0.02 |
| niqe / brisque | 0.158 / 0.034 | 0.07 / −0.06 |

Every metric is significantly worse than URanker (paired bootstrap p ≈ 0).

- **Week-4 gate (TOPIQ ≥ 0.7): FAIL (0.316). Under investigation, not yet interpretable.** Our UIQM/UCIQE/NIQE are about 0.2–0.27 below the UID2021 paper's own Table 9 (0.540 / 0.603 / 0.330). Candidate causes: the 77-observer MOS file vs the paper's 52; the workbook content; image↔MOS alignment; implementation differences. URanker's 0.62 argues against wholesale misalignment. `scripts/diagnose_uid2021.py` (last cell of notebook 01) checks these.
- **Diagnostic result (diagnose_uid2021.py):** no data bug.
  - All 960 image↔MOS pairs align, and all 60 scenes match the table's Scene column. The 60 "method mismatches" are just the raw images' label.
  - The workbook has one sheet and one MOS column (range 0–8.9).
  - Per-method mean MOS is sensible: UWCNN 1.56 < raw 1.88 < … < UWB-VCSE 6.78.
  - Images are 512×384; about 7% are RGBA (alpha is dropped on load).
  - The shortfall vs the paper is uniform across all six subsets (UCIQE 0.23–0.46 vs 0.54–0.67). Remaining explanations: (a) the MOS revision (77 vs 52 observers; the README switched Drive files in Dec 2022), or (b) the paper's MATLAB implementations.
  - `compare_uid2021_mos.py` was meant to test (a), but the pre-Dec-2022 Drive file is no longer public (2026-10-02), so (a) vs (b) cannot be resolved.
  - **Decision: the benchmark stands.** Report it on the released 77-observer MOS with open Python implementations, and state the gap to Table 9 and its two possible causes explicitly. The Week-4 gate's purpose ("the benchmark is not broken") is met: alignment is verified, per-method MOS is sensible, and URanker reaches 0.62. Its numeric threshold (TOPIQ ≥ 0.7, taken from Dumic et al.'s 4-method study) does not carry over to UID2021's 15 methods. That is itself evidence for Gap 2 (in-air NR-IQA transfers poorly to underwater images).
- Intra-scene SRCC (ranking enhancements of the same image): in-air deep metrics collapse to ≈ 0 (CLIP-IQA, QualiCLIP, MANIQA, TReS). URanker keeps 0.58. This holds whatever the gate outcome is.

**O2 stress test** (Spearman of score vs distortion strength 0 → 3, mean over 60 raw images; +1 = score rises steadily as the image is over-processed):

| metric | contrast | red shift | saturation | unsharp |
|---|---|---|---|---|
| uciqe | **0.91** | −0.01 | **0.76** | **1.00** |
| uranker | **0.89** | −0.51 | −0.86 | **1.00** |
| musiq / topiq_nr | 0.08 / −0.34 | −0.10 / −0.01 | −0.63 / −0.52 | **0.68 / 0.60** |
| liqe | 0.28 | −0.18 | −0.47 | −0.10 |
| uiqm | −0.77 | −0.30 | −0.73 | −0.09 |

- **UCIQE is gameable** by contrast, saturation and sharpening. **URanker** (the best metric on MOS) is gameable by contrast and sharpening. MUSIQ and TOPIQ reward over-sharpening. UIQM and LIQE do not inflate monotonically over this range.
- Caveat: the range spans 0 → 3, which is heavy. Small-strength inflation (e.g. UIQM under a mild red shift, seen in the unit tests) can hide inside a negative overall Spearman. A per-strength curve plot should go in the write-up. Claiming "humans prefer less" needs the small 2AFC study (improvement #8).

## Kaggle workflow
Imported notebooks are frozen copies, so all logic lives in the repo. The first cell of every notebook is
`%run /kaggle/working/repo/scripts/kaggle_setup.py`, which resets the clone to `origin/main`, reinstalls, and
sets every `*_ROOT`. Re-running it picks up new code; the notebook never needs re-importing. Session-specific
values go in the untracked `configs/local.yaml` via `set_local_config()`. Notebooks are generated by
`scripts/make_notebooks.py`.

## Then (Weeks 5–8)
- `enhance/deep.py`: wrappers for pretrained Water-Net, FUnIE-GAN, Ucolor, U-shape Transformer (inference only).
- `detect/per_image_utility.py`: YOLO11-s on raw RUOD → per-image AP on raw vs. each enhanced variant → Δ labels.
- `model/features.py`: cache frozen CLIP + DINOv2 embeddings to disk; `model/head.py`: MLP heads.

## Proposed improvements over the handoff plan (for discussion)

| # | Improvement | Why | Cost |
|---|---|---|---|
| 1 | **Intra-scene SRCC** next to global SRCC (implemented) | UID2021/SAUD contain K enhanced versions of each raw image. A metric is used to *rank enhancements of the same image*; global SRCC can be inflated by content differences between scenes. Likely a key finding. | none |
| 2 | **Paired significance tests** between metrics (implemented) | "TOPIQ beats UIQM" needs a p-value, not overlapping CIs | none |
| 3 | **Add URanker** (underwater-specific, in pyiqa) + ARNIQA, QualiCLIP | Without an underwater-trained baseline, reviewers will ask why it's missing | minutes |
| 4 | **Null-enhancement noise floor for Δ** | Run the detector on raw vs. a near-identical copy (JPEG q95 / 1-px shift). The spread of that Δ is label noise. Use it to define a 3-class label (help / neutral / hurt) instead of sign(Δ), which otherwise flips on noise. Directly addresses the biggest risk (§8). | ~1 extra detection pass |
| 5 | **Smoother per-image utility**: AP@[.5:.95] *and* a confidence-weighted matched-recall score; report both | Per-image AP is very jumpy on images with 1–2 objects | none |
| 6 | **Predict the best of K options, not just yes/no** | Condition on the enhancer → choose argmax over {raw, method₁..ₖ}. Stronger headline: "NR-selected enhancer vs. oracle selection vs. best single enhancer" | small |
| 7 | **Leakage-safe splits for RUOD** via DINOv2 near-duplicate clustering | RUOD has no scene IDs and contains near-duplicate frames; random splits leak | minutes (features are cached anyway) |
| 8 | **Replace O6 (MOS study) with a small 2AFC study on the O2 sweep images** | O2 claims "metric rises while human judgement falls" — that needs *some* human data. ~10 raters × 60 pairs is far cheaper than BT.500 and directly supports O2 | ~1 day |
| 9 | Report **both UCIQE variants** | Shows published UCIQE numbers are not comparable across papers | none |
| 10 | Detector-retrained protocol (b): use **enhancement-augmented** training rather than per-enhancer retraining | One extra training run instead of K | 1 run |

## Caveats found so far
- UCIQE (`paper` variant) *drops* after colour correction on a synthetic test image, because a strong cast gives high, uniform chroma. In other words, UCIQE can reward colour casts. Check this on real data.
- UIQM rises with red over-compensation (synthetic test: 2.04 → 2.34, above the clean image at 2.32). This is the O2 hypothesis in miniature.
- Classical fusion port is untuned (gamma = 2 output is somewhat dark). Acceptable as a baseline; deep enhancers matter more.
- `pip install pyiqa` upgraded protobuf to 7.x, which conflicts with streamlit/proto-plus in the global env. Use a venv locally if you need those.

## Citation checks still pending (from handoff)
Li & Cavallaro = ICIP 2022 (pages); Dumic et al. 2026 metadata; U-shape Transformer vol/pages; SAUD/RUIE/RUOD details.

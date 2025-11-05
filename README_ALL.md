# Surgical Phase Recognition — Repository Overview

This repository contains the full pipeline used in my Bachelor’s Thesis projects to recognize surgical phases in **endoscopic pituitary surgery**. It includes:

- **Spatial study** (frame-wise ResNet‑50 fine‑tuning)
- **Temporal study** (Multi‑Stage TCN over sliding windows of ResNet features)
- **Per‑patient visualization** (color‑coded ribbons)
- **5‑Fold cross‑validation** for the temporal study + **K‑fold ribbons**

> **Data splits (CSVs) are publicly available at:** https://doi.org/10.21950/YDGPZM  
> **Images** must be stored as PNG files named `{id}.png` in the `images/` folder.  
> Expected ID format: `PATIENT_SEG_FRAME` (anonymized).

---

## How to run (recommended order)

1) **Prepare data**
   - Download CSVs from the DOI above and place them at repo root:
     - combined_ResNet_DB.csv`, `train_ResNet_DB.csv`, `val_ResNet_DB.csv`, `test_ResNet_DB.csv`
   - Put PNG frames in `./images/` as `{id}.png` (IDs must match the CSVs).

2) **Train the spatial model (ResNet‑50) and get weights**
   ```bash
   python spatial_study.py
   ```
   - Produces: `best_spatial_study_weights.pth` (used for feature extraction).

3) **Train the temporal model (MS‑TCN), export predictions & metrics**
   ```bash
   python temporal_study.py
   ```
   - Extracts features (if missing), trains MS‑TCN with early stopping,
   - Exports: `predictions_tcn.csv` and `predictions_tcn_framewise.csv`,
   - Saves model: `model_mstcn.pth`,
   - Prints patient‑level accuracy and macro Recall/Precision/F1,
   - Shows a confusion matrix + training curves.

4) **Visualize per‑patient ribbons (GT vs. ResNet vs. TCN)**
   ```bash
   python surgical_phase_ribbon_comparison.py
   ```
   - Set `patient_id` at the top of the script.
   - Produces a 3‑row figure with color‑coded ribbons aligned in time.

5) **(Optional) 5‑Fold cross‑validation for the temporal study**
   ```bash
   python study_temporal_kfold.py
   ```
   - Performs 5 folds, saving per‑fold weights and frame‑ordered CSVs:
     - `best_model_fold{1..5}.pth`
     - `groundtruth_fold{1..5}.csv`
     - `predictions_tcn_fold{1..5}.csv`
   - Prints per‑fold metrics + **95% confidence intervals** across folds.

6) **(Optional) K‑fold ribbons (compare all folds vs. full GT)**
   ```bash
   python surgical_phase_ribbon_comparison_kfold.py
   ```
   - Renders a multi‑row figure: full GT (top), then GT(test)/Pred per fold.

---

## Repository layout (expected)

```
.
├── images/                                  # PNG frames {id}.png
├── combined_ResNet_DB.csv                    # all frames (GT)
├── train_ResNet_DB.csv, val_ResNet_DB.csv, test_ResNet_DB.csv
│
├── spatial_study.py                          # ResNet‑50 fine‑tuning (spatial baseline)
├── temporal_study.py                         # MS‑TCN training & test (single split)
├── surgical_phase_ribbon_comparison.py       # 3‑ribbon plot per patient
│
├── study_temporal_kfold.py                   # MS‑TCN with 5‑fold CV
├── surgical_phase_ribbon_comparison_kfold.py # ribbons across folds vs. full GT
│
├── best_spatial_study_weights.pth            # produced by spatial_study.py
├── model_mstcn.pth                           # produced by temporal_study.py
├── best_model_fold{1..5}.pth                 # produced by study_temporal_kfold.py
└── predictions_resnet.csv / predictions_tcn*.csv  # produced by scripts
```

---

## What each script does (summary)

| Script | Purpose | Key Inputs | Main Outputs |
|---|---|---|---|
| `spatial_study.py` | **Fine‑tunes ResNet‑50** on frame images to learn strong spatial features and provide a transparent baseline. | `images/`, `train_ResNet_DB.csv`, `val_ResNet_DB.csv` (+ optional `test_ResNet_DB.csv`) | `best_spatial_study_weights.pth`, training curves, (optional) `predictions_resnet.csv`, test metrics & confusion matrix |
| `temporal_study.py` | **Temporal modeling with MS‑TCN** on sliding windows of **pre‑extracted ResNet features**. Trains/validates on windows; stitches frame‑wise predictions at test time; exports CSVs; prints patient‑level metrics. | `images/`, CSV splits, **`best_spatial_study_weights.pth`** | `features_*` dirs (if missing), `model_mstcn.pth`, `predictions_tcn.csv`, `predictions_tcn_framewise.csv`, confusion matrix + curves |
| `surgical_phase_ribbon_comparison.py` | **Visualization**: 3 color‑coded ribbons for a given **patient** — GT, ResNet (upsampled), TCN (original length). | `combined_ResNet_DB.csv`, `predictions_resnet.csv`, `predictions_tcn.csv` (or `predictions_tcn_framewise.csv`) | 3‑row ribbon figure |
| `study_temporal_kfold.py` | **5‑Fold CV** for the MS‑TCN. Keeps exact frame order, writes per‑fold GT/pred CSVs, saves best fold weights, prints metrics and **95% CIs**. | `combined_ResNet_DB.csv`, `images/`, **`best_spatial_study_weights.pth`** | `best_model_fold{1..5}.pth`, `groundtruth_fold{1..5}.csv`, `predictions_tcn_fold{1..5}.csv`, confusion matrices, CI summary |
| `surgical_phase_ribbon_comparison_kfold.py` | **Visualization**: ribbons across folds aligned to a **global timeline** — full GT (top), then GT(test)/Pred for each fold. | `groundtruth_fold{1..5}.csv`, `predictions_tcn_fold{1..5}.csv`, `combined_ResNet_DB.csv` | multi‑row ribbon figure (GT full + per‑fold GT/Pred) |

> **Why spatial first?** Temporal models benefit from stable per‑frame features. A strong spatial backbone (ResNet‑50) improves TCN stability and reduces label jitter at transitions.

---

## Color palette (phase → color)

| Phase | Hex |
|---|---|
| P0 | `#D8BFD8` |
| P1 | `#386cb0` |
| P2 | `#ffff99` |
| P3 | `#f0027f` |
| P4 | `#FFA07A` |
| P5 | `#7fc97f` |
| P6 | `#B0E0E6` |
| Not in test (K‑fold ribbons) | `#FFFFFF` |

---

## Requirements

- Python ≥ 3.9 (tested with 3.10)
- **PyTorch** + **torchvision**
- pandas, numpy, matplotlib, Pillow, tqdm, scikit‑learn, scipy

Install (example):
```bash
pip install torch torchvision pandas numpy matplotlib pillow tqdm scikit-learn scipy
```

**GPU:** The code prefers CUDA if available (default device is set inside the scripts). If you see CUDA driver/runtime warnings, the scripts will fall back to CPU.

---

## Conventions & Notes

- **ID format:** `PATIENT_SEG_FRAME` (anonymized); must be consistent across CSVs and image filenames.
- **Features caching:** Temporal scripts will **extract features once** and reuse them on subsequent runs.
- **Metrics:** We report macro Recall/Precision/F1 and **mean accuracy per patient** (to account for variable video lengths).
- **Reproducibility:** We keep the code straightforward and avoid heavy, opaque abstractions. Early stopping and LR scheduling are enabled.
- **Ethics & privacy:** Ensure all IDs are anonymized before sharing artifacts or figures.

---

## Data citation

If you use the provided splits, please cite the dataset DOI:
- **DOI:** https://doi.org/10.21950/YDGPZM

---

## Questions?

Open an issue or reach out. Thanks for checking out the project!

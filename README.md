
# MS‑TCN Temporal Study (5‑Fold CV) & Color‑Ribbon Visualization

This folder contains the **K‑Fold** version of the temporal study (Multi‑Stage TCN over sliding windows) and a companion script to visualize per‑fold results as **color‑coded ribbons** aligned to the full ground‑truth timeline.

> ✅ The CSVs used by these scripts are available at the DOI: **https://doi.org/10.21950/YDGPZM**  
> (Download `combined_ResNet_DB.csv` and the split CSVs if you wish to reproduce the same evaluation setup.)

---

## Contents

- `study_temporal_kfold.py` — Trains & evaluates a **Multi‑Stage TCN (MS‑TCN)** with **5‑fold cross‑validation** on pre‑extracted ResNet‑50 features.  
  Exports per‑fold frame‑ordered **ground truth** and **predictions** to CSV, prints metrics (macro **Recall/Precision/F1**, **mean patient‑level accuracy**) and computes **95% confidence intervals** across folds.

- `surgical_phase_ribbon_comparison_kfold.py` — Plots color‑strip timelines that compare:  
  1) **Full** ground truth (all frames),  
  2) **Per‑fold** ground truth (test subset),  
  3) **Per‑fold** TCN predictions.  
  Frames outside a fold’s test subset are shown **white** (“not in test”).

- *(Optional, included in this repo screenshot)*:  
  `best_model_fold1.pth` … `best_model_fold5.pth` — saved MS‑TCN weights per fold (by validation accuracy). You can reuse them to regenerate visualizations without retraining.

---

## Data & Expected File Structure

```
repo_root/
├─ images/                                  # PNG frames named {id}.png
├─ combined_ResNet_DB.csv                   # Full dataset (columns: id,label)
├─ best_spatial_study_weights.pth           # Fine‑tuned ResNet‑50 (from spatial study)
├─ study_temporal_kfold.py
├─ surgical_phase_ribbon_comparison_kfold.py
└─ (optional) best_model_fold{1..5}.pth
```

- **IDs** must be anonymized and follow the pattern used in the project (e.g., `PATIENT_SEG_FRAME`).  
- If the `features_all/` directory does not exist, it will be generated the first time you run the temporal study (one `.npy` per frame).

> **Source of CSVs:** DOI **https://doi.org/10.21950/YDGPZM**.

---

## Quick Start

1) **Install requirements** (Python ≥ 3.9 recommended):

```bash
pip install torch torchvision pandas numpy matplotlib pillow tqdm scikit-learn scipy
```

2) **Place assets** as shown in the structure above. Ensure:
   - `images/` contains `{id}.png` frames.
   - `combined_ResNet_DB.csv` exists with columns `id,label`.
   - `best_spatial_study_weights.pth` is present (from your spatial fine‑tuning).

3) **Run 5‑Fold CV training/evaluation:**

```bash
python study_temporal_kfold.py
```

This will:
- Extract **ResNet‑50** features once to `features_all/` (if missing).
- Perform **5 folds** (StratifiedKFold) over `combined_ResNet_DB.csv`.
- For each fold *i*:
  - Save best TCN weights to `best_model_fold{i}.pth`.
  - Export CSVs in original temporal order:  
    `groundtruth_fold{i}.csv` and `predictions_tcn_fold{i}.csv`.
  - Show a normalized **confusion matrix** figure.
- Print per‑fold metrics and **95% CIs** across folds for Accuracy, Recall, Precision, and F1.

4) **Render color‑ribbon timelines:**

```bash
python surgical_phase_ribbon_comparison_kfold.py
```

This will plot:
- Row 1: **Full ground truth** (all frames).
- For each fold *i*:  
  **GT (test)** and **TCN prediction** ribbons, aligned to the same global timeline.  
  Frames **not in test** for that fold appear **white**.

---

## Color Palette (Phases 0–6)

The same discrete palette is used consistently across scripts:

| Phase | Label | Hex |
|---|---|---|
| 0 | P0 | `#D8BFD8` |
| 1 | P1 | `#386cb0` |
| 2 | P2 | `#ffff99` |
| 3 | P3 | `#f0027f` |
| 4 | P4 | `#FFA07A` |
| 5 | P5 | `#7fc97f` |
| 6 | P6 | `#B0E0E6` |

- **Not in test** (for a given fold): `#FFFFFF` (white).

---

## Key Implementation Notes

- **Feature extraction** uses a fine‑tuned **ResNet‑50** (classification head removed).  
  The temporal model consumes `[T, 2048]` per‑frame vectors via **sliding windows** (`window_size=300`, `step=100` by default).

- **MS‑TCN** architecture: one stage over features + refinement stages over softmax outputs. A **truncated MSE** on log‑prob temporal differences helps reduce over‑segmentation.

- **Validation** within each fold uses a 90/10 split of the training windows (last 10% as validation).

- **Frame‑ordered predictions** are reconstructed to match the exact timeline of `combined_ResNet_DB.csv` test indices for each fold; gaps are filled with the last valid prediction to keep continuity.

- **Device selection:** by default, scripts try `cuda:1` if available; otherwise **CPU**. Change the device line to suit your system (e.g., `torch.device("cuda")`).

---

## Outputs Summary

After running `study_temporal_kfold.py`, you should see (per fold `i`):

- `best_model_fold{i}.pth` — best MS‑TCN weights.  
- `groundtruth_fold{i}.csv` — columns: `id,label` (frame‑ordered).  
- `predictions_tcn_fold{i}.csv` — columns: `id,pred` (frame‑ordered).  
- Confusion matrix figure titled **“Confusion Matrix – fold i”**.  
- Console metrics: **mean patient‑level accuracy (%)**, macro **Recall / Precision / F1**.  
- Final **95% CI** across folds for the above metrics.

Then, `surgical_phase_ribbon_comparison_kfold.py` produces the multi‑row ribbon plot aligning: **full GT**, **per‑fold GT(test)**, and **per‑fold TCN predictions**.

---

## Reproducibility Tips

- Keep the same **CSV IDs** and **image naming** (`{id}.png`).  
- If you already have `features_all/`, the script will reuse them (saves time).  
- Set a fixed random seed if you want deterministic window ordering / loaders.

---

## Citation & Data

If you use these scripts, please reference the dataset/CSV source:  
**DOI:** https://doi.org/10.21950/YDGPZM

---

**Author:** Ángel Pérez Navas  
**License:** Research/educational use. Verify clinical data is anonymized before sharing outputs.

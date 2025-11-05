# Surgical Phase Recognition — Temporal Study (MS‑TCN) & Color Ribbons

This folder contains two scripts to **train/evaluate a Multi‑Stage Temporal Convolutional Network (MS‑TCN)** on surgical videos (using pre‑extracted ResNet‑50 features) and to **visualize results** as color‑coded timelines per patient.

> **CSV splits** (train/val/test and combined) used by these scripts are publicly available here: **https://doi.org/10.21950/YDGPZM**.  
> Make sure your IDs remain **anonymized** before publishing any artifacts.

---

## Files

- `temporal_study.py` — End‑to‑end pipeline:
  1) load a fine‑tuned ResNet‑50 as a frozen **feature extractor** (removing the FC head),
  2) **extract** per‑frame features into `.npy` files,
  3) build **sliding‑window** datasets (train/val/test),
  4) **train** a Multi‑Stage TCN with early stopping,
  5) **test** on windows and **stitch** frame‑wise predictions to compute **per‑patient accuracy**,
  6) export predictions to CSV and plot a normalized confusion matrix.

- `surgical_phase_ribbon_comparison.py` — **Visualization** of **Ground Truth vs. ResNet vs. TCN** for a single patient as **color ribbons** aligned along the surgery timeline.
- *(optional)* `model_mstcn.pth` — a saved MS‑TCN checkpoint (produced by `temporal_study.py`).

---

## Data layout

- **Images**: folder `images/` with PNG frames named `{id}.png`.
- **CSV splits**: `train_ResNet_DB.csv`, `val_ResNet_DB.csv`, `test_ResNet_DB.csv` (UTF‑8, header; columns: `id,label`).  
  A combined CSV `combined_ResNet_DB.csv` is used by the visualization script.

> Download the CSVs from **https://doi.org/10.21950/YDGPZM**.

---

## Environment

- Python ≥ 3.9
- PyTorch, TorchVision
- NumPy, pandas, scikit‑learn, matplotlib, Pillow, tqdm

Quick install:

```bash
pip install torch torchvision torchaudio             numpy pandas scikit-learn matplotlib pillow tqdm
```

> **GPU:** the script defaults to `cuda:1` when available. Adjust the device line in `main()` if your setup differs (e.g., `cuda`/`cuda:0`).

---

## 1) Training & Testing — `temporal_study.py`

### What it does
- Uses your **fine‑tuned ResNet‑50** weights (from the spatial study) as a **frozen backbone** to extract a 2048‑D vector per frame.
- Builds **sliding windows** (e.g., `window_size=300`, `step=100`).
- Trains an **MS‑TCN** (multi‑stage causal 1D convs) with **truncated MSE** temporal smoothing and optional **class‑weighted CE**.
- During **test**, stitches overlapping windows to compute **per‑patient accuracy**. Also writes two CSVs:
  - `predictions_tcn.csv` — window‑level (each row is a *frame in a specific window*),
  - `predictions_tcn_framewise.csv` — frame‑level (majority vote across overlapping windows).

### Inputs
- `images/` with `{id}.png`
- CSVs: `train_ResNet_DB.csv`, `val_ResNet_DB.csv`, `test_ResNet_DB.csv`
- **Pretrained spatial weights**: path in code (default: `best_spatial_study_weights.pth`).

### Outputs
- Feature folders: `features_train/`, `features_val/`, `features_test/` containing `{id}.npy`
- Checkpoint: `model_mstcn.pth`
- CSVs: `predictions_tcn.csv`, `predictions_tcn_framewise.csv`
- Plots: Train/Val loss curves, Validation accuracy curve, Test confusion matrix
- Printed metrics: patient‑level mean accuracy (%), macro Recall/Precision/F1, per‑class recall/precision

### How to run
1. Open `temporal_study.py` and **adjust paths/hyperparameters** in `main()`:
   - CSV/Images paths
   - `resnet_in` (your fine‑tuned spatial weights)
   - windowing params, number of stages/layers, LR, epochs
2. Run:
   ```bash
   python temporal_study.py
   ```

---

## 2) Visualization — `surgical_phase_ribbon_comparison.py`

### What it does
Draws three **color‑coded ribbons** (height=1) aligned by time for a **single patient**:
1. **Ground Truth** (upsampled to TCN length),
2. **ResNet** frame‑wise predictions (upsampled),
3. **TCN** predictions (original length).

Since lengths may differ, the script **upsamples by nearest neighbor** to match the TCN sequence length.

### Inputs
- `combined_ResNet_DB.csv` (`id,label`) — ground truth
- `predictions_resnet.csv` (`img_id,true_label,predicted_label`)
- `predictions_tcn.csv` (`frame_id,true_label,predicted_label`)  
  *(or switch to `predictions_tcn_framewise.csv` inside the script)*

### Usage
1. Set `patient_id` and CSV paths in the configuration section.
2. Run:
   ```bash
   python surgical_phase_ribbon_comparison.py
   ```

### Phase palette
Mapping **phase → color** (used consistently across scripts):
- `0: P0` → `#D8BFD8`
- `1: P1` → `#386cb0`
- `2: P2` → `#ffff99`
- `3: P3` → `#f0027f`
- `4: P4` → `#FFA07A`
- `5: P5` → `#7fc97f`
- `6: P6` → `#B0E0E6`

---

## Notes & Good Practices

- **Anonymization:** ensure patient and frame IDs are anonymized. Never include PHI/PII in filenames or CSVs.
- **Device:** change `torch.device("cuda:1" if torch.cuda.is_available() else "cpu")` if your GPU index differs.
- **Class imbalance:** enable weighted CE by keeping `class_counts` derived from the training CSV.
- **Determinism:** for exact reproducibility, you may set random seeds and `torch.backends.cudnn.deterministic = True` (trade‑off with speed).

---

## Data
CSV splits used by these scripts are available at: **https://doi.org/10.21950/YDGPZM**.

---

## Acknowledgments
- MS‑TCN architecture inspired by the original action segmentation work.
- ResNet‑50 used as spatial backbone whose fine‑tuned weights come from the spatial study.


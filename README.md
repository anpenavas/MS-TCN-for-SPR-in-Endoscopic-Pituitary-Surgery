# Spatial Study — ResNet-50 Fine-Tuning for Surgical Phase Recognition

This repository contains the **spatial baseline** for surgical phase recognition: a ResNet-50 fine-tuned on frame images to learn robust **per-frame** visual features. The resulting backbone serves as the foundation for subsequent **temporal** models (e.g., MS-TCN).

> **Why a spatial study first?**  
> Temporal models benefit from stable frame-wise features. A strong spatial backbone provides those features and a transparent, reproducible baseline before adding temporal context.

---

## 📦 Repository Contents

- `spatial_study.py` — training/evaluation script (fine-tune ResNet-50 on frames).
- `best_spatial_study_weights.pth` — example trained weights produced by the script.

---

## 📊 Data

The CSV splits referenced by the script can be obtained here:

**CSV datasets (train/val/test)** → https://doi.org/10.21950/YDGPZM

Expected **CSV schema** (comma-separated):

```csv
id,label
1050_1_000001,3
1050_1_000002,3
...
```

- `id` → base filename (without `.png`)
- `label` → integer in `[0..num_classes-1]`

Place the CSVs at the project root (or update paths in `main()`).

---

## 🗂️ Directory Structure

Minimal structure to run the script:

```
.
├── images/                          # PNG frames named {id}.png
│   ├── 1050_1_000001.png
│   └── ...
├── train_ResNet_DB.csv
├── val_ResNet_DB.csv
├── test_ResNet_DB.csv               # optional (enables test metrics & exports)
├── spatial_study.py
└── best_spatial_study_weights.pth   # will be created if you train from scratch
```

---

## 🔧 Requirements

- Python 3.9+
- PyTorch & TorchVision (CUDA optional)
- NumPy, Pandas, Pillow, scikit-learn, Matplotlib, tqdm

Install (example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install numpy pandas pillow scikit-learn matplotlib tqdm
```

> Use the appropriate PyTorch/TorchVision wheels for your CUDA/CPU environment.

---

## ▶️ Quick Start

1) **Adjust paths/hyperparameters** in `main()` of `spatial_study.py` if needed:

```python
train_csv  = "train_ResNet_DB.csv"
val_csv    = "val_ResNet_DB.csv"
test_csv   = "test_ResNet_DB.csv"   # leave as-is; if missing, test is skipped
img_dir    = "images"
out_model  = "best_spatial_study_weights.pth"

num_classes = 7
batch_size  = 32
lr          = 1e-4
epochs      = 50
freeze_until= 6
```

2) **Run**:

```bash
python spatial_study.py
```

The script will:
- Train with early stopping (patience=10) and LR-on-Plateau (by val accuracy),
- Save the best weights to `best_spatial_study_weights.pth`,
- Plot train/val **loss** and **accuracy** curves,
- If `test_ResNet_DB.csv` exists:
  - Print test **accuracy** and **macro-F1**,
  - Show a **normalized confusion matrix**,
  - Export test predictions to `predictions_resnet.csv`.

---

## 🧠 Model & Augmentations

- **Backbone**: ResNet-50 (`IMAGENET1K_V1` weights), final FC replaced with `num_classes`.
- **Freezing**: early blocks frozen up to `freeze_until` (0..7 approx).
- **Train transforms**: RandomResizedCrop(224), HFlip, Rotation(±10°), ColorJitter, Normalize.
- **Val/Test transforms**: Resize(256) → CenterCrop(224) → Normalize.

---

## 🧾 Outputs

- `best_spatial_study_weights.pth` — best validation checkpoint (state dict).
- `predictions_resnet.csv` (if test CSV is present):

```csv
img_id,true_label,predicted_label
1050_1_000001,3,3
1050_1_000002,3,2
...
```

- Matplotlib figures:
  - **Loss Curves** (Train/Val)
  - **Accuracy Curves** (Train/Val)
  - **Confusion Matrix** (Test, normalized)

---

## ⚙️ Compute Notes

- The script auto-selects a device via `get_device()`:
  - Uses the first visible CUDA device if available, otherwise **CPU**.
  - It **does not hardcode** `cuda:1` (respects `CUDA_VISIBLE_DEVICES`).
  - Silences the common CUDA 804 init warning to avoid noisy logs.

---

## 🧯 Troubleshooting

- **TorchVision IO warning** (`Failed to load image Python extension`):  
  If you **don’t** use `torchvision.io`, you can ignore it. Otherwise, ensure
  compatible `torch/torchvision` wheels and system libs (`libjpeg`, `libpng`).

- **CUDA init warnings (e.g., Error 804)**:  
  Usually driver/runtime mismatch. The script already falls back to CPU.  
  If you have GPUs, verify:
  - `nvidia-smi` works (driver OK),
  - Your `torch/torchvision` match the installed CUDA runtime.

- **DataLoader workers**:  
  If you see stuck runs on Windows/WSL, set `num_workers=0` in the data loaders.

---

## 🔒 Privacy

Ensure all image filenames/IDs in CSVs are **anonymized** before sharing or publishing artifacts.

---

## 📄 Citation

If you use the CSVs, please reference the dataset DOI:  
**https://doi.org/10.21950/YDGPZM**

---

## 🙌 Acknowledgments

- ResNet-50 pretraining: ImageNet (TorchVision).
- This spatial baseline is part of a broader project that includes temporal modeling (MS-TCN) on top of these features.

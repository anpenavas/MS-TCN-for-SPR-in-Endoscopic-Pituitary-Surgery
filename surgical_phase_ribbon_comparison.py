# ÁNGEL PÉREZ NAVAS
"""
surgical_phase_ribbon_comparison.py
=========================

Visualization — Ground Truth vs. ResNet vs. TCN (per patient)

Purpose
-------
This script plots three color-strip timelines (1 pixel tall each) for a given
patient to visually compare:
1) Ground-truth labels,
2) ResNet frame-wise predictions,
3) TCN predictions.

Since the three sequences may have different lengths, ground truth and ResNet
predictions are **upsampled (nearest)** to match the TCN sequence length so
that all timelines align for visual inspection.

Inputs
------
- `combined_ResNet_DB.csv`  : combined dataset with columns ['id','label'] (ground truth).
- `predictions_resnet.csv`  : ResNet predictions with columns ['img_id','true_label','predicted_label'].
- `predictions_tcn.csv`     : TCN predictions with columns ['frame_id','true_label','predicted_label'].
  (Alternatively, you can switch to `predictions_tcn_framewise.csv`.)

Notes on filtering:
- The patient is selected by string prefix match on the 'id' or 'frame_id'
  column. Ensure your patient identifiers are **anonymized**.

Outputs
-------
- A Matplotlib figure with three aligned color strips:
  * Ground Truth (upsampled)
  * ResNet Predictions (upsampled)
  * TCN Predictions (original length)
- A legend mapping phase indices (0..6) to short labels ("P0".."P6").

Usage
-----
1) Adjust `patient_id` and CSV paths in the configuration section.
2) Run:
   python surgical_phase_ribbon_comparison.py

Dependencies
------------
pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================
# 1) CONFIGURATION
# ============================================

patient_id = "1050"

csv_combined = "combined_ResNet_DB.csv"
csv_resnet   = "predictions_resnet.csv"
csv_tcn      = "predictions_tcn.csv"
# csv_tcn      = "predictions_tcn_framewise.csv"  # <- uncomment to use frame-wise voted TCN

phase2label_dict = {
    0: "P0", 1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5", 6: "P6"
}

# Color palette per phase (0..6)
colors = [
    "#D8BFD8",  # P0
    "#386cb0",  # P1
    "#ffff99",  # P2
    "#f0027f",  # P3
    "#FFA07A",  # P4
    "#7fc97f",  # P5
    "#B0E0E6",  # P6
]

# ============================================
# Utility — nearest-neighbor upsampling
# ============================================

def upsample_labels(labels: np.ndarray, new_len: int) -> np.ndarray:
    """
    Nearest-neighbor expand 'labels' to 'new_len' without losing information.
    Example: if old_len=4 and new_len=8, each element is repeated ~2x.

    Parameters
    ----------
    labels : np.ndarray
        1D array of integer labels.
    new_len : int
        Target length after upsampling.

    Returns
    -------
    np.ndarray
        1D array of length 'new_len' with the same dtype as 'labels'.
    """
    old_len = len(labels)
    if old_len == 0:
        return np.array([], dtype=labels.dtype)
    factor = new_len / float(old_len)
    idxs_float = np.arange(new_len) / factor            # in [0, old_len)
    idxs_round = np.round(idxs_float).astype(int)
    idxs_round = np.clip(idxs_round, 0, old_len - 1)    # guard bounds
    return labels[idxs_round]

# ============================================
# 2) DATA LOADING
# ============================================

print(f"\nReading combined dataset (ground truth) from: {csv_combined}")
df_all = pd.read_csv(csv_combined)
filtered_df = df_all[df_all["id"].astype(str).str.startswith(patient_id)]
ground_truth_labels = filtered_df["label"].values
print("Length ground_truth_labels =", len(ground_truth_labels))

print(f"\nReading ResNet predictions from: {csv_resnet}")
df_pred_resnet = pd.read_csv(csv_resnet)
filtered_df_resnet = df_pred_resnet[df_pred_resnet["img_id"].astype(str).str.startswith(patient_id)]
resnet_labels = filtered_df_resnet["predicted_label"].values
print("Length resnet_labels =", len(resnet_labels))

print(f"\nReading TCN predictions from: {csv_tcn}")
df_pred_tcn = pd.read_csv(csv_tcn)
df_pred_tcn["frame_id"] = df_pred_tcn["frame_id"].astype(str)
filtered_df_tcn = df_pred_tcn[df_pred_tcn["frame_id"].str.startswith(patient_id)]
tcn_labels = filtered_df_tcn["predicted_label"].values
print("Length tcn_labels =", len(tcn_labels))

# ============================================
# 3) ALIGN (UPSAMPLE) TO TCN LENGTH
# ============================================

desired_len = len(tcn_labels)
print(f"\nUpsampling ground truth and ResNet to {desired_len}...")

gtruth_upsampled = upsample_labels(ground_truth_labels, desired_len)
resnet_upsampled = upsample_labels(resnet_labels, desired_len)

print("ground_truth =>", len(gtruth_upsampled))
print("resnet       =>", len(resnet_upsampled))
print("tcn          =>", len(tcn_labels))

# ============================================
# 4) PLOT CONFIGURATION
# ============================================

cmap = ListedColormap(colors)
bounds = np.arange(0, 8)  # 0..7 to create 7 discrete bins
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(3, 1, figsize=(20, 5), sharex=True)

def plot_sequence(ax_, data, title):
    """
    Render a 1-row color strip representing a label sequence.
    """
    data = np.array(data)
    reshaped = data.reshape(1, -1)  # [1, T]
    ax_.imshow(
        reshaped, aspect="auto", cmap=cmap, norm=norm,
        extent=[0, len(data), 0, 1]
    )
    ax_.set_title(title)
    ax_.set_yticks([])
    ax_.set_xticks([])

plot_sequence(ax[0], gtruth_upsampled, "Ground Truth (upsampled)")
plot_sequence(ax[1], resnet_upsampled, "ResNet Predictions (upsampled)")
plot_sequence(ax[2], tcn_labels,       "TCN Predictions (original)")

# ============================================
# 5) LEGEND
# ============================================

handles = [
    mpatches.Patch(color=colors[i], label=phase2label_dict[i])
    for i in range(7)
]
# Position legend outside the plot on the right
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(1.15, 0.3), prop={"size": 12})

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()

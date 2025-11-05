#ÁNGEL PEREZ NAVAS
"""
surgical_phase_ribbon_comparison_kfold.py
==============================

Visualization — Color-Coded Ribbons for 5-Fold Results vs. Full Ground Truth

Purpose
-------
This script renders color-strip timelines to compare, across K folds:
  • The full ground-truth ribbon (all frames, in temporal order),
  • Per-fold ground-truth ribbons (test split only),
  • Per-fold TCN prediction ribbons (aligned to the same global timeline).

Ribbons are 1-row color images where the horizontal axis represents time (frame
index in the surgery). Colors map phases 0..6; frames not belonging to a fold’s
test set are shown as white (“not in test”).

Inputs
------
- 'groundtruth_fold{i}.csv'     (for i in 1..K): columns include an ID column and 'label'
- 'predictions_tcn_fold{i}.csv' (for i in 1..K): columns include an ID column and 'pred'
- 'combined_ResNet_DB.csv'      : columns ['id','label'] for the full sequence

Notes
-----
- The script automatically detects the ID and label/pred columns by name
  (looks for substrings like 'id', 'frame', 'img' and 'label'/'pred').
- The global timeline is built from the fold ground-truth CSVs, preserving
  their order of appearance.
- Ensure IDs are anonymized before sharing.

Outputs
-------
- A Matplotlib figure with:
  • Row 1: Full Ground Truth ribbon,
  • Rows 2..(1+2K): For each fold, the test Ground Truth ribbon and the TCN Prediction ribbon.
- A legend mapping phases (P0..P6) and the “not in test” color.

Usage
-----
1) Place the per-fold CSVs (ground truth and predictions) next to this script.
2) Adjust 'n_folds' if needed.
3) Run:
   python surgical_phase_ribbon_comparison_kfold.py

Dependencies
------------
pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------
n_folds        = 5
csv_gt_folds   = [f"groundtruth_fold{i}.csv"     for i in range(1, n_folds + 1)]
csv_pred_folds = [f"predictions_tcn_fold{i}.csv" for i in range(1, n_folds + 1)]

phase2label  = dict(enumerate(["P0", "P1", "P2", "P3", "P4", "P5", "P6"]))
phase_colors = ['#D8BFD8', '#386cb0', '#ffff99', '#f0027f',
                '#FFA07A', '#7fc97f', '#B0E0E6']
missing_color = "#FFFFFF"  # frames not in the test split for that fold

# ------------------------------------------------------------
# 2) Helpers — detect ID / label columns by name
# ------------------------------------------------------------
def detect_id_col(df):
    for c in df.columns:
        if any(t in c.lower() for t in ('id', 'frame', 'img')):
            return c
    raise KeyError("No ID-like column found (expected something like 'id', 'frame', or 'img').")

def detect_lab_col(df, idc):
    for c in df.columns:
        if c != idc and any(t in c.lower() for t in ('label', 'pred')):
            return c
    raise KeyError("No label/pred column found (expected something like 'label' or 'pred').")

# ------------------------------------------------------------
# 3) Read all fold CSVs and build the global ID universe
# ------------------------------------------------------------
fold_pairs = []   # [(df_gt, df_pred), ...]
all_ids    = []   # accumulate in temporal order

for f_gt, f_pr in zip(csv_gt_folds, csv_pred_folds):
    dgt, dpr = pd.read_csv(f_gt), pd.read_csv(f_pr)
    idg, idp = detect_id_col(dgt), detect_id_col(dpr)
    dgt[idg] = dgt[idg].astype(str)
    dpr[idp] = dpr[idp].astype(str)

    fold_pairs.append((dgt, dpr))
    # GT and predictions should carry the same IDs for the fold
    all_ids.extend(dgt[idg].tolist())

# Global universe (unique IDs, preserving order)
id_univ = list(dict.fromkeys(all_ids))
N = len(id_univ)
pos_map = {v: i for i, v in enumerate(id_univ)}

print(f"Total frames in global timeline: {N}")
for i, (dgt, dpr) in enumerate(fold_pairs, 1):
    idg = detect_id_col(dgt)
    print(f"tmp_te_fold{i}.csv: {dgt.shape[0]} test frames")
    # predictions_tcn_fold{i}.csv should match this count for seen frames

# ------------------------------------------------------------
# 4) Build aligned arrays per fold (global length N)
# ------------------------------------------------------------
fold_gt, fold_pred = [], []
for dgt, dpr in fold_pairs:
    idg, idp   = detect_id_col(dgt), detect_id_col(dpr)
    labg, labp = detect_lab_col(dgt, idg), detect_lab_col(dpr, idp)

    # arrays filled with "missing" (-1)
    a_gt = np.full(N, -1, int)
    a_pr = np.full(N, -1, int)

    # fill GT
    idx = dgt[idg].map(pos_map)
    msk = idx.notna()
    a_gt[idx[msk].astype(int).values] = dgt.loc[msk, labg].values

    # fill Predictions
    idx = dpr[idp].map(pos_map)
    msk = idx.notna()
    a_pr[idx[msk].astype(int).values] = dpr.loc[msk, labp].values

    fold_gt.append(a_gt)
    fold_pred.append(a_pr)

# ------------------------------------------------------------
# 5) Read full Ground Truth (not only test sets)
# ------------------------------------------------------------
df_comb = pd.read_csv("combined_ResNet_DB.csv")
id_comb = detect_id_col(df_comb)
df_comb[id_comb] = df_comb[id_comb].astype(str)

gt_full = np.full(N, -1, dtype=int)
idx_comb = df_comb[id_comb].map(pos_map)
valid_comb = idx_comb.notna()
gt_full[idx_comb[valid_comb].astype(int).values] = df_comb.loc[valid_comb, 'label'].values

# ------------------------------------------------------------
# 6) Colormap
# ------------------------------------------------------------
# The data arrays contain -1 (missing) and 0..6 (phases).
# We put "missing" first in the colormap, then the 7 phase colors.
cmap = ListedColormap([missing_color] + phase_colors)
norm = BoundaryNorm(np.arange(-1, len(phase_colors) + 1), cmap.N)

# ------------------------------------------------------------
# 7) Plot
# ------------------------------------------------------------
rows = 1 + 2 * n_folds
fig, axs = plt.subplots(rows, 1, figsize=(22, 2.2 * rows), sharex=True)

def paint(ax, data, title):
    """
    Render a 1-row color strip for a label sequence (global length N).
    """
    ax.imshow([data], aspect='auto', cmap=cmap, norm=norm, extent=[0, N, 0, 1])
    ax.set_title(title, loc='left', fontsize=11)
    ax.set_yticks([]); ax.set_xticks([])

# Row 0: full ground truth across the entire timeline
paint(axs[0], gt_full, "Ground Truth (full)")

# Rows per fold: GT(test) and TCN prediction
for i in range(n_folds):
    paint(axs[2*i + 1], fold_gt[i],   f"Fold {i+1} • Ground Truth (test)")
    paint(axs[2*i + 2], fold_pred[i], f"Fold {i+1} • TCN Prediction")

# Legend (phases + "not in test")
patches = [mp.Patch(color=phase_colors[i], label=phase2label[i])
           for i in range(len(phase_colors))]
patches.append(mp.Patch(color=missing_color, label="not in test"))
fig.legend(handles=patches, loc='lower center',
           bbox_to_anchor=(0.5, -0.05), ncol=4, prop={'size': 12})

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()

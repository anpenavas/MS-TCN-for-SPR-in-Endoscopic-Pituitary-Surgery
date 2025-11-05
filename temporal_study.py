# ÁNGEL PÉREZ NAVAS
"""
temporal_study.py
================================

Temporal Study — Multi-Stage TCN on Sliding Windows with Per-Patient Stitching

Purpose
-------
This script trains a Multi-Stage Temporal Convolutional Network (MS-TCN) over
sliding windows of pre-extracted ResNet-50 features. It validates on windows,
and during testing it stitches frame-wise predictions to compute per-patient
accuracy. It also exports window-level and frame-level predictions to CSV.

Pipeline
--------
1) Load a fine-tuned ResNet-50 and use it as a frozen feature extractor
   (removing the classification head). Extract features to .npy files.
2) Build sliding-window datasets for train/val/test from the features.
3) Train a Multi-Stage TCN (with early stopping).
4) Validate on windows and track losses/accuracy per epoch.
5) Test on windows and stitch frame-wise predictions to compute patient-level metrics.
6) Export predictions to CSV files and generate a confusion matrix + metrics.

Inputs
------
1) Frame images:
   - Folder 'images/' with PNG files named '{id}.png'.

2) CSV splits (UTF-8 with header):
   - 'train_ResNet_DB.csv', 'val_ResNet_DB.csv', 'test_ResNet_DB.csv'
   - Columns: 'id,label'
     * 'id'    → base filename (without '.png'), assumed anonymized
     * 'label' → integer in '[0 .. num_classes-1]'

3) Pretrained ResNet-50 weights (fine-tuned):
   - Path: 'best_spatial_study_weights.pth' (adjust in 'main()' if needed)

Outputs
-------
- Feature directories (if not already present):
  * 'features_train/', 'features_val/', 'features_test/' with '{id}.npy' vectors
- Trained TCN weights: 'modelo_tcn.pth'
- Plots:
  * Train/Val loss curves
  * Validation accuracy curve
  * (Test) Normalized confusion matrix
- CSV exports (during test):
  * 'predictions_tcn.csv'            → window-level per-frame rows
  * 'predictions_tcn_framewise.csv'  → frame-level rows after majority voting
- Printed metrics:
  * Patient-level mean accuracy (%)
  * Global macro Recall/Precision/F1
  * Per-class recall and precision

Usage
-----
1) Adjust paths/hyperparameters in 'main()' and run:
   python temporal_study.py

Notes
-----
- By default, the script uses 'cuda:1' if available; otherwise CPU.
  Change the device line if your setup differs.
- Early stopping uses patience=10; LR scheduler is ReduceLROnPlateau (by val accuracy).
- Ensure your CSV IDs are **anonymized** before publishing artifacts.
"""

import os
import csv
import time
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, precision_score, f1_score
)

# =====================================================
# 1) Dataset for ResNet feature extraction
# =====================================================

class SimpleImageDataset(Dataset):
    """
    Dataset that reads (image, label, id) tuples from a CSV.
    CSV columns required: ['id', 'label'].
    Images must exist at: {img_dir}/{id}.png
    """
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = row['id']
        label  = row['label']
        path_img = os.path.join(self.img_dir, img_id + ".png")
        im = Image.open(path_img).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return im, label, img_id


def extract_features(loader, backbone, out_dir):
    """
    Pass images through the ResNet backbone and save per-frame features to .npy.
    Output vector shape per frame: [2048]
    """
    os.makedirs(out_dir, exist_ok=True)
    device = next(backbone.parameters()).device
    backbone.eval()
    with torch.no_grad():
        for imgs, labs, ids in tqdm(loader, desc=f"Extract feats => {out_dir}"):
            imgs  = imgs.to(device)
            feats = backbone(imgs)                  # [B, 2048, 1, 1]
            feats = feats.view(feats.size(0), -1)   # [B, 2048]
            for i, f_id in enumerate(ids):
                np.save(os.path.join(out_dir, f_id + ".npy"),
                        feats[i].cpu().numpy())


# =====================================================
# 2) TCN Datasets: Sliding windows of features
# =====================================================

class TemporalFeaturesByPatientWindowDataset(Dataset):
    """
    Builds sliding windows per patient from pre-extracted features.

    CSV columns:
      id="patient_seg_frame", label
    Windowing:
      window_size, step (stride)
    Returns:
      (X, Y, patient_id, start_idx)
      X: [window_size, 2048] features
      Y: [window_size] integer labels
    """
    def __init__(self, csv_path, features_dir, window_size=300, step=100):
        self.features_dir = features_dir
        df = pd.read_csv(csv_path)

        def parse_id(img_id):
            parts = img_id.split('_')
            pat   = parts[0]
            seg   = int(parts[1])
            frm   = int(parts[2])
            return pat, seg, frm

        df['patient'], df['seg'], df['frm'] = zip(*df['id'].map(parse_id))
        self.data_windows = []

        for pat, grp in df.groupby('patient'):
            grp   = grp.sort_values(['seg', 'frm']).reset_index(drop=True)
            feats = []
            labs  = []
            for _, row in grp.iterrows():
                fid   = row['id']
                label = row['label']
                pathf = os.path.join(features_dir, fid + ".npy")
                vec   = np.load(pathf)   # shape [2048]
                feats.append(vec)
                labs.append(label)
            feats = np.array(feats, dtype='float32')  # [T, 2048]
            labs  = np.array(labs,  dtype='int64')    # [T]
            T     = feats.shape[0]

            # sliding windows
            idx = 0
            while idx < T:
                end  = idx + window_size
                subF = feats[idx:end]
                subL = labs[idx:end]
                if subF.shape[0] == window_size:
                    self.data_windows.append((pat, idx, subF, subL))
                idx += step

            # add final window if leftover frames exist and T > window_size
            if T > 0 and (T - window_size) % step != 0 and T > window_size:
                last_start = T - window_size
                subF = feats[last_start:]
                subL = labs[last_start:]
                self.data_windows.append((pat, last_start, subF, subL))

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        pat_id, start_idx, subF, subL = self.data_windows[idx]
        X = torch.from_numpy(subF).float()  # [win_size, 2048]
        Y = torch.from_numpy(subL).long()   # [win_size]
        return X, Y, pat_id, start_idx


# =====================================================
# 3) TCN Model (Multi-Stage)
# =====================================================

class DilatedResidualLayer(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size,
                               padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample:
            res = self.downsample(res)
        out += res
        return self.relu(out)


class SingleStageHead(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers, num_classes):
        super().__init__()
        layers = [nn.Conv1d(in_ch, hidden_ch, kernel_size=1)]
        dilation = 1
        for _ in range(num_layers):
            layers.append(DilatedResidualLayer(hidden_ch, hidden_ch, dilation=dilation))
            dilation *= 2
            if dilation > 16:
                dilation = 16
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(hidden_ch, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.network(x)         # [B, hidden_ch, T]
        logits = self.classifier(out) # [B, num_classes, T]
        return logits


class MultiStageTCN(nn.Module):
    def __init__(self, num_stages=3, in_channels=2048,
                 hidden=64, num_layers=8, num_classes=7):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        # stage 1 (from features)
        self.stages.append(SingleStageHead(in_channels, hidden, num_layers, num_classes))
        # stages 2..N (from previous softmax)
        for _ in range(1, num_stages):
            self.stages.append(SingleStageHead(num_classes, hidden, num_layers, num_classes))

    def forward(self, x):
        # x: [B, 2048, T]
        outputs = []
        out1 = self.stages[0](x)
        outputs.append(out1)
        prev = out1
        for s in range(1, self.num_stages):
            probs = F.softmax(prev, dim=1)
            out_s = self.stages[s](probs)
            outputs.append(out_s)
            prev = out_s
        return outputs


def truncated_mse_loss(logits, tau=4.0):
    """
    Temporal smoothing loss (truncated MSE on log-probs differences).
    Reduces over-segmentation by penalizing abrupt changes.
    """
    B, C, T = logits.shape
    probs = F.softmax(logits, dim=1)
    logp = torch.log(probs + 1e-8)
    diffs = []
    for t in range(1, T):
        diff_t = (logp[:, :, t] - logp[:, :, t-1]).abs()
        clamped = torch.clamp(diff_t, max=tau)
        diffs.append(clamped ** 2)
    if len(diffs) == 0:
        return torch.tensor(0.0, device=logits.device)
    diffs_all = torch.stack(diffs, dim=2)
    return diffs_all.mean()


# =====================================================
# 4) Optional temporal post-filter (mode filter)
# =====================================================

def mode_filter(preds, window=15):
    """
    Apply local mode filter over a 1D integer sequence.
    """
    T_ = len(preds)
    filtered = preds.copy()
    half = window // 2
    for i in range(T_):
        left = max(0, i - half)
        right = min(T_, i + half + 1)
        sub = preds[left:right]
        vals, counts = np.unique(sub, return_counts=True)
        maj = vals[counts.argmax()]
        filtered[i] = maj
    return filtered


# =====================================================
# 5) Training (with Early Stopping and logging)
# =====================================================

def train_tcn_sliding(
    train_ds, val_ds, model,
    epochs=20, lr=1e-4,
    lambda_smooth=0.15, tau=4.0,
    class_counts=None,
    early_stop_patience=10
):
    """
    Train the MS-TCN on sliding windows (train_ds, val_ds).
    Returns the model loaded with best validation accuracy weights.
    """
    device = next(model.parameters()).device
    dl_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    dl_val   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    # Weighted CrossEntropy (optional)
    weights_tensor = None
    if class_counts is not None:
        inv = [1.0 / c for c in class_counts]
        w_np = np.array(inv, dtype='float32')
        w_np = w_np / w_np.sum() * len(w_np)
        weights_tensor = torch.FloatTensor(w_np).to(device)
        print("[INFO] Using Weighted CrossEntropy with weights:", weights_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_acc = 0.0
    best_wts = None

    # For plotting
    train_losses_epochs = []
    val_losses_epochs   = []
    val_accs_epochs     = []

    # Early stopping
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for X, Y, pat, stidx in dl_train:
            X = X.to(device)   # [1, win_size, 2048]
            Y = Y.to(device)
            X = X.permute(0, 2, 1)  # [1, 2048, win_size]

            optimizer.zero_grad()
            logits_list = model(X)
            loss_total = 0.0
            for logit in logits_list:
                B_, C_, T_ = logit.shape
                logits2d = logit.view(B_ * T_, C_)
                labs1d   = Y.view(B_ * T_)

                ce_loss = F.cross_entropy(logits2d, labs1d, weight=weights_tensor)
                sm_loss = truncated_mse_loss(logit, tau)
                loss_stage = ce_loss + lambda_smooth * sm_loss
                loss_total += loss_stage

            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            train_count += 1

        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0

        # Validation
        model.eval()
        correct = 0
        totalf  = 0
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for Xv, Yv, patv, stv in dl_val:
                Xv = Xv.to(device)
                Yv = Yv.to(device)
                Xv = Xv.permute(0, 2, 1)
                out_list = model(Xv)
                final_out = out_list[-1]
                Bv, Cv, Tv = final_out.shape
                logits2d = final_out.view(Bv * Tv, Cv)
                labs1d   = Yv.view(Bv * Tv)

                loss_ = F.cross_entropy(logits2d, labs1d, weight=weights_tensor)
                val_loss += loss_.item()
                val_count += 1

                pred = logits2d.argmax(dim=1)
                correct += (pred == labs1d).sum().item()
                totalf  += labs1d.numel()

        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        val_acc = 100.0 * correct / totalf if totalf > 0 else 0.0

        # log for plots
        train_losses_epochs.append(avg_train_loss)
        val_losses_epochs.append(avg_val_loss)
        val_accs_epochs.append(val_acc)

        print(f"[Epoch {ep+1}/{epochs}] train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.2f}%")

        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc  = val_acc
            best_wts  = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"[EARLY STOPPING] No improvement in val_acc for {early_stop_patience} epochs.")
            break

    print("[Training complete] Best val_acc =", best_acc)
    if best_wts:
        model.load_state_dict(best_wts)

    # Loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses_epochs) + 1), train_losses_epochs, label='Train Loss')
    plt.plot(range(1, len(val_losses_epochs) + 1),   val_losses_epochs,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Train/Val)')
    plt.legend()
    plt.show()

    # Validation accuracy curve
    plt.figure()
    plt.plot(range(1, len(val_accs_epochs) + 1), val_accs_epochs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.show()

    return model


# =====================================================
# 6) Testing on windows + stitching per patient
# =====================================================

def test_tcn_sliding_per_patient(test_ds, model, window_postproc=15, num_classes=7):
    """
    Iterate over test windows, get frame-wise predictions, and aggregate
    correctness per patient: acc_patient = correct_patient / total_patient.
    Also export window-level and frame-level (after voting) predictions to CSV.
    """
    device = next(model.parameters()).device
    dl_test = DataLoader(test_ds, batch_size=1, shuffle=False)

    patient_stats = {}   # patient_id -> [correct_frames, total_frames]
    model.eval()

    all_preds_list  = []
    all_labels_list = []

    all_ids_list = []    # for window-level CSV (per-frame rows)
    frame_votes  = {}    # frame_id -> {'label': gt, 'preds': [votes...]}

    with torch.no_grad():
        for X, Y, pat_id, st_idx in dl_test:
            X = X.to(device)  # [1, W, 2048]
            Y = Y.to(device)
            X = X.permute(0, 2, 1)  # [1, 2048, W]
            out_list  = model(X)
            final_out = out_list[-1]
            B_, C_, T_ = final_out.shape

            logits2d = final_out.view(B_ * T_, C_)
            labs1d   = Y.view(B_ * T_)

            preds   = logits2d.argmax(dim=1).cpu().numpy()
            labs_np = labs1d.cpu().numpy()

            # optional smoothing inside each window
            if window_postproc > 1:
                preds = mode_filter(preds, window=window_postproc)

            # update per-patient stats
            correct_local = (preds == labs_np).sum()
            total_local   = labs_np.size
            if pat_id[0] not in patient_stats:
                patient_stats[pat_id[0]] = [0, 0]
            patient_stats[pat_id[0]][0] += correct_local
            patient_stats[pat_id[0]][1] += total_local

            # global metrics lists
            all_preds_list.extend(preds)
            all_labels_list.extend(labs_np)

            # window-level rows for CSV + frame-level vote reservoir
            for i in range(len(labs_np)):
                frame_id = f"{pat_id[0]}_idx{st_idx.item() + i}"
                all_ids_list.append((frame_id, labs_np[i], preds[i]))

                if frame_id not in frame_votes:
                    frame_votes[frame_id] = {'label': labs_np[i], 'preds': []}
                else:
                    # sanity check: GT should be consistent across overlapping windows
                    gt_previous = frame_votes[frame_id]['label']
                    if gt_previous != labs_np[i]:
                        print(f"[WARNING] Inconsistent GT at {frame_id}")
                frame_votes[frame_id]['preds'].append(preds[i])

    # CSV 1: window-level (each row is a frame observed in a specific window)
    csv_filename = "predictions_tcn.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "true_label", "predicted_label"])
        for (fid, tlab, plab) in all_ids_list:
            writer.writerow([fid, tlab, plab])
    print(f"[INFO] Window-level CSV written to {csv_filename}")

    # CSV 2: frame-level (majority vote across overlapping windows)
    framewise_ids   = sorted(frame_votes.keys())
    framewise_preds = []
    framewise_label = []
    for fid in framewise_ids:
        arr_preds = frame_votes[fid]['preds']
        gt_label  = frame_votes[fid]['label']
        voted = Counter(arr_preds).most_common(1)[0][0]
        framewise_preds.append(voted)
        framewise_label.append(gt_label)

    csv_filename2 = "predictions_tcn_framewise.csv"
    with open(csv_filename2, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "true_label", "predicted_label"])
        for i, fid in enumerate(framewise_ids):
            writer.writerow([fid, framewise_label[i], framewise_preds[i]])
    print(f"[INFO] Frame-wise (voted) CSV written to {csv_filename2}")

    # Patient-level mean accuracy
    paccs = []
    for pat, (c_, t_) in patient_stats.items():
        pacc = c_ / t_ if t_ > 0 else 0.0
        paccs.append(pacc)
    final_acc = 100.0 * np.mean(paccs) if len(paccs) > 0 else 0.0

    # Global metrics and confusion matrix
    cm = confusion_matrix(all_labels_list, all_preds_list, normalize='true')
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [str(i) for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(num_classes)])

    threshold = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i,
                     f"{cm[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.show()

    recall_test    = recall_score(all_labels_list, all_preds_list, average='macro')
    precision_test = precision_score(all_labels_list, all_preds_list, average='macro')
    f1_test        = f1_score(all_labels_list, all_preds_list, average='macro')

    report_dict = classification_report(all_labels_list, all_preds_list, output_dict=True)
    class_recall_test    = [report_dict[str(i)]['recall']    for i in range(num_classes)]
    class_precision_test = [report_dict[str(i)]['precision'] for i in range(num_classes)]

    results = {
        'acc_pacientes': final_acc,                # mean accuracy across patients (%)
        'confusion_matrix': cm,
        'recall_test': recall_test,
        'precision_test': precision_test,
        'f1_test': f1_test,
        'class_recall_test': class_recall_test,
        'class_precision_test': class_precision_test,
        'patient_stats': patient_stats
    }
    return results


# =====================================================
# 7) Utility: class counts (for optional weighting)
# =====================================================

def get_class_counts(csv_path, num_classes=7):
    """
    Count occurrences per class in the provided CSV.
    """
    df = pd.read_csv(csv_path)
    counts = [(df['label'] == i).sum() for i in range(num_classes)]
    print(f"[INFO] Class counts in {csv_path}:", counts)
    return counts


# =====================================================
# 8) Main
# =====================================================

def main():
    # CSV for ResNet (feature extraction)
    train_csv_res = "train_ResNet_DB.csv"
    val_csv_res   = "val_ResNet_DB.csv"
    test_csv_res  = "test_ResNet_DB.csv"
    img_dir       = "images"
    resnet_in     = "best_spatial_study_weights.pth"

    # Feature directories
    features_train = "features_train"
    features_val   = "features_val"
    features_test  = "features_test"

    # CSV for TCN (must include id=patient_seg_frame, label)
    train_csv_tcn = "train_ResNet_DB.csv"
    val_csv_tcn   = "val_ResNet_DB.csv"
    test_csv_tcn  = "test_ResNet_DB.csv"

    # TCN hyperparameters
    num_classes   = 7
    num_stages    = 3
    hidden        = 64
    num_layers    = 8
    lambda_smooth = 0.15
    tau           = 4.0
    epochs        = 100
    lr            = 1e-4

    # Weighted CE (optional): compute from training CSV
    class_counts = get_class_counts(train_csv_tcn, num_classes=num_classes)
    # If not desired, set: class_counts = None

    # Sliding windows
    window_size     = 300
    step_size       = 100
    window_postproc = 15  # mode filter inside window (optional)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print("Loading fine-tuned ResNet:", resnet_in)
    net = models.resnet50(weights=None)
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_feats, num_classes)
    
    # Safe state_dict loading (avoids FutureWarning; no impact on results)
    try:
        # Newer PyTorch: prefer weights_only=True for safer deserialization
        state = torch.load(resnet_in, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility: older PyTorch versions don't support weights_only
        state = torch.load(resnet_in, map_location=device)
    
    net.load_state_dict(state)
    net.eval()
    
    # Build the backbone (remove FC) for feature extraction
    backbone = nn.Sequential(*list(net.children())[:-1]).to(device)
    backbone.eval()

    # Data augmentation (train)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # No augmentation (val/test)
    transform_valtest = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds_train_res = SimpleImageDataset(train_csv_res, img_dir, transform_train)
    ds_val_res   = SimpleImageDataset(val_csv_res,   img_dir, transform_valtest)
    ds_test_res  = SimpleImageDataset(test_csv_res,  img_dir, transform_valtest)

    dl_train_res = DataLoader(ds_train_res, batch_size=32, shuffle=False, num_workers=2)
    dl_val_res   = DataLoader(ds_val_res,   batch_size=32, shuffle=False, num_workers=2)
    dl_test_res  = DataLoader(ds_test_res,  batch_size=32, shuffle=False, num_workers=2)

    # Extract features if missing
    if not os.path.isdir(features_train):
        extract_features(dl_train_res, backbone, features_train)
    else:
        print("[INFO] features_train/ exists -> skipping extraction")
    if not os.path.isdir(features_val):
        extract_features(dl_val_res, backbone, features_val)
    else:
        print("[INFO] features_val/ exists -> skipping extraction")
    if not os.path.isdir(features_test):
        extract_features(dl_test_res, backbone, features_test)
    else:
        print("[INFO] features_test/ exists -> skipping extraction")

    # Window datasets for TCN
    ds_train = TemporalFeaturesByPatientWindowDataset(
        train_csv_tcn, features_train, window_size, step_size
    )
    ds_val = TemporalFeaturesByPatientWindowDataset(
        val_csv_tcn, features_val, window_size, step_size
    )
    ds_test = TemporalFeaturesByPatientWindowDataset(
        test_csv_tcn, features_test, window_size, step_size
    )
    print(f"[Windows] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # Build TCN
    model = MultiStageTCN(
        num_stages=num_stages, in_channels=2048,
        hidden=hidden, num_layers=num_layers, num_classes=num_classes
    ).to(device)

    # Train (with early stopping)
    model = train_tcn_sliding(
        ds_train, ds_val, model,
        epochs=epochs, lr=lr,
        lambda_smooth=lambda_smooth, tau=tau,
        class_counts=class_counts,
        early_stop_patience=10
    )

    # Save TCN model
    torch.save(model.state_dict(), "model_mstcn.pth")
    print("[INFO] Model saved to model_mstcn.pth")

    # Test => stitch + CSV
    print("\n[TEST] Predicting on windows and stitching results per patient...\n")
    test_results = test_tcn_sliding_per_patient(
        ds_test, model, window_postproc=window_postproc, num_classes=num_classes
    )

    final_acc          = test_results['acc_pacientes']            # percentage
    recall_test        = test_results['recall_test']
    precision_test     = test_results['precision_test']
    f1_test            = test_results['f1_test']
    class_recall_test  = test_results['class_recall_test']
    class_precision_test = test_results['class_precision_test']
    patient_stats      = test_results['patient_stats']

    # Print global metrics
    acc_test = final_acc / 100.0
    print("\nTEST RESULTS:")
    print("- Accuracy:", acc_test)
    print("- Recall:", recall_test)
    print("- Precision:", precision_test)
    print("- F1 Score:", f1_test)
    print("- Class recall:", class_recall_test)
    print("- Class precision:", class_precision_test)

    print(f"\n[TEST] Mean accuracy across patients = {final_acc:.2f}%")
    for pat_id, (cc, tt) in patient_stats.items():
        pacc = 100.0 * cc / tt if tt > 0 else 0.0
        print(f"  Patient={pat_id}, frames={tt}, acc={pacc:.2f}%")

    print("Pipeline finished.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n[Total time] Script executed in {end_time - start_time:.2f} seconds")

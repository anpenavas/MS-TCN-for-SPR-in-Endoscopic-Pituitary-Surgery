#ÁNGEL PÉREZ NAVAS
"""
study_temporal_kfold.py
==========================

Temporal Study — Multi-Stage TCN with 5-Fold Cross-Validation
and frame-level CSV export.

Purpose
-------
This script trains and evaluates a Multi-Stage Temporal Convolutional Network
(MS-TCN) using **5-fold cross-validation** over sliding windows of pre-extracted
ResNet-50 features. For each fold, it:
  • Trains on windows (with early stopping),
  • Validates on a hold-out split from the training windows,
  • Tests on windows and reconstructs frame-level predictions in the **exact**
    temporal order of the original test split,
  • Saves per-fold confusion matrices and CSVs with ground truth and predictions,
  • Reports macro Recall/Precision/F1 and **mean patient-level accuracy**,
  • Computes 95% confidence intervals across folds.

Inputs
------
- 'combined_ResNet_DB.csv'   : Single CSV with columns ['id','label'] for all frames
                               (IDs must be anonymized; format like 'PATIENTX_SEG_FRAME').
- 'images/'                  : Folder with '{id}.png' frames.
- 'best_spatial_study_weights.pth'
                             : Fine-tuned ResNet-50 weights (used to extract features).

Outputs
-------
- 'features_all/'                      : One-time ResNet feature extraction ('{id}.npy').
- 'best_model_fold{fold}.pth'          : Best MS-TCN weights per fold (by val accuracy).
- 'groundtruth_fold{fold}.csv'         : Frame-ordered ground truth for the fold.
- 'predictions_tcn_fold{fold}.csv'     : Frame-ordered predictions for the fold.
- Confusion matrix figure per fold.
- Console metrics:
    * Mean patient-level accuracy (%),
    * Macro Recall / Precision / F1,
    * 95% CI across folds for Accuracy/Recall/Precision/F1.

Usage
-----
1) Adjust paths/hyperparameters in 'main()'.
2) Run:
   python study_temporal_kfold.py

Notes
-----
- By default, the script uses 'cuda:1' if available; otherwise CPU.
  Change the device line if your setup differs.
- Ensure any patient identifiers in CSVs are anonymized before sharing artifacts.
"""

# ===============================================================
#  IMPORTS
# ===============================================================
import os, time, csv, numpy as np, pandas as pd
from collections import Counter
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, recall_score,
                             precision_score, f1_score)
from scipy import stats  # 95% CI

# ===============================================================
# 0)  95% Confidence Interval
# ===============================================================
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    m = a.mean()
    h = stats.sem(a) * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
    return m, m - h, m + h


# ===============================================================
# 1) Image dataset (for ResNet feature extraction)
# ===============================================================
class SimpleImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir, self.tf = img_dir, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_id, label = self.df.loc[idx, ['id', 'label']]
        img = Image.open(f"{self.img_dir}/{img_id}.png").convert('RGB')
        if self.tf: img = self.tf(img)
        return img, label, img_id


def extract_features(loader, backbone, out_dir):
    """
    Pass images through the ResNet backbone and save {id}.npy vectors.
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = next(backbone.parameters()).device
    backbone.eval()
    with torch.no_grad():
        for imgs, _, ids in tqdm(loader, desc="Extract feats"):
            feats = backbone(imgs.to(dev)).view(imgs.size(0), -1)
            for i, fid in enumerate(ids):
                np.save(f"{out_dir}/{fid}.npy", feats[i].cpu().numpy())


# ===============================================================
# 2)  Feature-window dataset (for the TCN)
# ===============================================================
class TemporalFeaturesByPatientWindowDataset(Dataset):
    """
    Returns: feats[T,2048], labels[T], patient_id, start_idx, ids[T]
    Sliding windows of length `window_size`, step `step`.
    """
    def __init__(self, csv_path, feat_dir,
                 window_size=300, step=100):
        self.data_windows = []
        df = pd.read_csv(csv_path)
        self.feat_dir = feat_dir

        # -------- parse id --------
        def parse_id(s: str):
            parts = s.split('_')
            frame = int(parts[-1])
            seg   = int(parts[-2])
            patient = "_".join(parts[:-2])
            return patient, seg, frame

        df[['patient', 'seg', 'frm']] = (
            df['id'].apply(parse_id).apply(pd.Series)
        )

        # -------- build windows ----------
        for pat, grp in df.groupby('patient'):
            grp = grp.sort_values(['seg', 'frm']).reset_index(drop=True)
            ids   = grp['id'].tolist()
            feats = [np.load(f"{feat_dir}/{i}.npy") for i in ids]
            labs  = grp['label'].values.astype('int64')
            feats = np.stack(feats).astype('float32')

            T = len(feats)
            idx = 0
            while idx < T:
                end = idx + window_size
                if end <= T:
                    self.data_windows.append(
                        (pat, idx, feats[idx:end],
                         labs[idx:end], ids[idx:end]))
                idx += step
            # final window if leftover frames exist
            if T > window_size and (T - window_size) % step != 0:
                st = T - window_size
                self.data_windows.append(
                    (pat, st, feats[st:], labs[st:], ids[st:]))

    def __len__(self): return len(self.data_windows)

    def __getitem__(self, i):
        pat, st, X, y, ids = self.data_windows[i]
        return (torch.from_numpy(X),
                torch.from_numpy(y),
                pat, st, ids)


# ===============================================================
# 3)  Multi-Stage TCN model
# ===============================================================
class DilatedResidualLayer(nn.Module):
    def __init__(self, ch, dil=1, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, k, padding=dil, dilation=dil)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.c2(self.relu(self.c1(x))))


class SingleStage(nn.Module):
    def __init__(self, in_ch, hid, n_layers, n_cls):
        super().__init__()
        mods = [nn.Conv1d(in_ch, hid, 1)]
        dil = 1
        for _ in range(n_layers):
            mods.append(DilatedResidualLayer(hid, dil))
            dil = min(dil * 2, 16)
        self.body = nn.Sequential(*mods)
        self.cls = nn.Conv1d(hid, n_cls, 1)

    def forward(self, x):               # x: [B,C,T]
        return self.cls(self.body(x))   # [B,n_cls,T]


class MultiStageTCN(nn.Module):
    def __init__(self, stages, in_ch, hid, layers, n_cls):
        super().__init__()
        self.stg = nn.ModuleList(
            [SingleStage(in_ch, hid, layers, n_cls)] +
            [SingleStage(n_cls, hid, layers, n_cls) for _ in range(stages - 1)]
        )

    def forward(self, x):
        out = self.stg[0](x)
        outs = [out]
        for s in self.stg[1:]:
            out = s(F.softmax(out, 1))
            outs.append(out)
        return outs                     # list of logits


def tMSE(logits, tau=4.):
    """
    Truncated MSE on log-prob differences (temporal smoothing).
    """
    l = torch.log_softmax(logits, 1)
    return ((l[:, :, 1:] - l[:, :, :-1]).abs().clamp(max=tau) ** 2).mean()


# ===============================================================
# 4)  TRAIN / VAL
# ===============================================================
def train_tcn(train_ds, val_ds, model, *, epochs=20, lr=1e-4,
              lam=0.15, tau=4.0, weights=None, patience=10, fold=0):
    dev = next(model.parameters()).device
    tr = DataLoader(train_ds, batch_size=1)
    va = DataLoader(val_ds, batch_size=1)

    # Class weighting (optional)
    w = None
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=dev)
        w = w / w.sum() * len(w)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_w, noimp = 0, None, 0
    for ep in range(1, epochs + 1):
        # --------- TRAIN -------------
        model.train(); tl = 0
        for X, Y, *_ in tr:
            X = X.to(dev).permute(0, 2, 1)  # [1,C,T]
            Y = Y.to(dev)
            opt.zero_grad()
            loss = 0.
            for log in model(X):
                B, C, T = log.shape
                ce = F.cross_entropy(
                    log.permute(0, 2, 1).reshape(-1, C),
                    Y.reshape(-1),
                    weight=w
                )
                loss += ce + lam * tMSE(log, tau)
            loss.backward()
            opt.step()
            tl += loss.item()

        # --------- VAL ---------------
        model.eval(); corr = tot = 0
        with torch.no_grad():
            for X, Y, *_ in va:
                pr = model(X.to(dev).permute(0, 2, 1))[-1].argmax(1)
                corr += (pr.cpu() == Y).sum().item()
                tot  += Y.numel()

        acc = 100 * corr / tot
        print(f"[{ep:02d}] loss={tl/len(tr):.4f}  valAcc={acc:.2f}")

        if acc > best:
            best, best_w, noimp = acc, model.state_dict(), 0
            # Save model whenever validation improves
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
        else:
            noimp += 1
        if noimp >= patience:
            print("  Early-stop."); break

    model.load_state_dict(best_w)
    return model


# ===============================================================
# 5)  TEST  +  frame-level CSV export
# ===============================================================
def mode_filter(vec, w=15):
    """
    Local mode filter over a 1D integer sequence.
    """
    res, half = vec.copy(), w // 2
    for i in range(len(vec)):
        sub = vec[max(0, i-half):min(len(vec), i+half+1)]
        res[i] = Counter(sub).most_common(1)[0][0]
    return res


def _patient_from_id(fid: str):
    """Extract patient identifier from a frame id string."""
    parts = fid.split('_')
    return "_".join(parts[:-2])  # everything except seg and frame


def test_tcn(test_df, test_ds, model,
             *, post=15, n_cls=7, fold=0):
    """
    - test_df : ORIGINAL fold dataframe in exact temporal order
    - test_ds : Windowed feature dataset
    Reconstructs frame-level predictions in the original order, applies
    optional smoothing, and computes metrics + confusion matrix.
    """
    dev = next(model.parameters()).device
    dl  = DataLoader(test_ds, batch_size=1, shuffle=False)

    preds_dict = {}  # id -> pred (only for frames that passed through the net)

    model.eval()
    with torch.no_grad():
        for X, _, _, _, ids in dl:
            X = X.to(dev).permute(0, 2, 1)
            pr = model(X)[-1].argmax(1).squeeze().cpu().numpy()
            if post > 1:
                pr = mode_filter(pr, post)
            for fid, p in zip(ids[0], pr):
                preds_dict[str(fid)] = int(p)

    # ---------- frame-ordered lists (exact test_df order) ----------
    all_ids = test_df['id'].astype(str).tolist()
    y_true  = test_df['label'].tolist()

    # Fill gaps (frames unseen by the network) with the last valid prediction
    y_pred, last_p = [], -1
    for fid in all_ids:
        if fid in preds_dict:
            last_p = preds_dict[fid]
        y_pred.append(last_p)

    # ---------- metrics ----------
    rec  = recall_score   (y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score       (y_true, y_pred, average='macro', zero_division=0)

    # Mean accuracy per patient
    by_pat = {}
    for fid, g, p in zip(all_ids, y_true, y_pred):
        pat = _patient_from_id(fid)
        by_pat.setdefault(pat, [0, 0])
        by_pat[pat][0] += int(g == p)
        by_pat[pat][1] += 1
    acc_patient = np.mean([c / t for c, t in by_pat.values()]) * 100

    # ---------- CSV exports ----------
    pd.DataFrame({'id': all_ids, 'label': y_true}).to_csv(
        f"groundtruth_fold{fold}.csv", index=False)
    pd.DataFrame({'id': all_ids, 'pred':  y_pred }).to_csv(
        f"predictions_tcn_fold{fold}.csv", index=False)

    # ---------- Confusion matrix ----------
    cm = confusion_matrix(y_true, y_pred, labels=range(n_cls),
                          normalize='true')
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix - fold {fold}')
    plt.colorbar()
    ticks = np.arange(n_cls)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, f"{cm[i,j]:.2f}",
                     ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout(); plt.show()

    return dict(acc_pacientes=acc_patient,
                recall_test=rec,
                precision_test=prec,
                f1_test=f1)


# ===============================================================
# 6)  Run ONE fold
# ===============================================================
def run_one_fold(fold, df_trval, df_test, feat_dir, device, hp):
    # helper to create a temporary CSV + its windowed dataset
    def mk_ds(df, csv_name):
        df.to_csv(csv_name, index=False)
        return TemporalFeaturesByPatientWindowDataset(
            csv_name, feat_dir, hp['window_size'], hp['step_size'])

    ds_full_tr = mk_ds(df_trval, f"tmp_tr_fold{fold}.csv")

    # internal split 90-10 (last 10% for validation)
    n_val = int(0.1 * len(ds_full_tr))
    idx = np.arange(len(ds_full_tr))
    sub_tr = torch.utils.data.Subset(ds_full_tr, idx[:-n_val])
    sub_va = torch.utils.data.Subset(ds_full_tr, idx[-n_val:])
    ds_te  = mk_ds(df_test, f"tmp_te_fold{fold}.csv")

    # model
    mdl = MultiStageTCN(hp['num_stages'], 2048,
                        hp['hidden'], hp['num_layers'],
                        hp['num_classes']).to(device)

    counts = [(df_trval['label'] == i).sum()
              for i in range(hp['num_classes'])]

    mdl = train_tcn(sub_tr, sub_va, mdl,
                    epochs=hp['epochs'], lr=hp['lr'],
                    lam=hp['lambda_smooth'], tau=hp['tau'],
                    weights=counts, patience=10, fold=fold)

    # use df_test to preserve exact temporal order
    return test_tcn(df_test, ds_te, mdl,
                    post=hp['window_postproc'],
                    n_cls=hp['num_classes'],
                    fold=fold)


# ===============================================================
# 7)  MAIN
# ===============================================================
def main():
    k = 5
    csv_comb = "combined_ResNet_DB.csv"
    img_dir  = "images"
    feat_dir = "features_all"
    resnet_w = "best_spatial_study_weights.pth"

    hp = dict(num_classes=7, num_stages=3, hidden=64, num_layers=8,
              lambda_smooth=0.15, tau=4.0,
              epochs=100, lr=1e-4,               # adjust epochs as needed
              window_size=300, step_size=100,
              window_postproc=15)

    dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    df_all = pd.read_csv(csv_comb)

    # ---------- extract features ONCE ----------
    if not os.path.isdir(feat_dir):
        print("[INFO] Extracting features with ResNet...")
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406],
                                 [.229, .224, .225])])
        net = models.resnet50(weights=None)
        net.fc = nn.Identity()  # remove classifier
        net.load_state_dict(torch.load(resnet_w,
                                       map_location=dev), strict=False)
        backbone = nn.Sequential(*list(net.children())[:-1]).to(dev).eval()
        dl = DataLoader(SimpleImageDataset(csv_comb, img_dir, tf),
                        batch_size=32, shuffle=False)
        extract_features(dl, backbone, feat_dir)
    else:
        print("[INFO] Features already exist – reusing.")

    # ---------- Cross-Validation ----------
    skf = StratifiedKFold(k, shuffle=False)
    accs = []; recs = []; precs = []; f1s = []

    for fold, (tr, te) in enumerate(skf.split(df_all['id'],
                                             df_all['label']), 1):
        print(f"\n========== FOLD {fold}/{k} ==========")
        met = run_one_fold(fold,
                           df_all.iloc[tr].reset_index(drop=True),
                           df_all.iloc[te].reset_index(drop=True),
                           feat_dir, dev, hp)
        accs.append(met['acc_pacientes'])
        recs.append(met['recall_test'])
        precs.append(met['precision_test'])
        f1s.append(met['f1_test'])

        print(f"Fold {fold}: AccPatients={met['acc_pacientes']:.2f} "
              f"R={met['recall_test']:.3f} "
              f"P={met['precision_test']:.3f} "
              f"F1={met['f1_test']:.3f}")

    # ---------- 95% CI ----------
    for name, vec in zip(['Accuracy (%)', 'Recall',
                          'Precision', 'F1'],
                         [accs, recs, precs, f1s]):
        m, lo, hi = mean_confidence_interval(vec)
        print(f"{name:<13}: {m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")


# ===============================================================
if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[Total time] {time.time()-t0:.1f}s")

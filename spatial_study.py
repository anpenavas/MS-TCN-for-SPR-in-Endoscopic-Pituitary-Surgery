# ÁNGEL PÉREZ NAVAS
"""
spatial_study.py
==================

Spatial Study — ResNet-50 Fine-Tuning for Surgical Phase Recognition

Purpose
-------
This script fine-tunes a ResNet-50 on frame images (spatial study) to learn
robust frame-wise visual features and establish a baseline before adding
temporal modeling (e.g., MS-TCN).

Why spatial first?
------------------
Temporal models benefit from stable per-frame features. A strong spatial
backbone (here, ResNet-50) provides those features and a transparent,
reproducible baseline for downstream temporal architectures.

Inputs
------
1) Image directory:
   - 'images/' containing PNG files named '{id}.png'.

2) CSV splits:
   - 'train_ResNet_DB.csv', 'val_ResNet_DB.csv', 'test_ResNet_DB.csv'
   - Columns: 'id,label'
     * 'id'    → base filename (without '.png')
     * 'label' → integer in '[0..num_classes-1]'

Outputs
-------
- Best model weights saved to: 'best_spatial_study_weights.pth'
- Training plots:
  * Loss curves (train/val)
  * Accuracy curves (train/val)
- Optional (if test CSV exists):
  * Printed test Accuracy and macro F1
  * Normalized confusion matrix figure
  * CSV export of predictions: 'predictions_resnet.csv'
    (columns: 'img_id,true_label,predicted_label')

Usage
-----

1) Adjust paths/hyperparameters in 'main()' and run:
   python spatial_study.py

Notes
-----
- By default, the script uses 'cuda:1' if available; otherwise CPU.
  Change the device line if your setup differs.
- Early stopping with patience=10 and ReduceLROnPlateau (by val accuracy).
- Code logic is intentionally simple and declarative for reproducibility.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import csv
import warnings

# Optional: silence the noisy CUDA initialization warning (Error 804, etc.)
warnings.filterwarnings(
    "ignore",
    message=r".*CUDA initialization: Unexpected error from cudaGetDeviceCount.*"
)

def get_device():
    """
    Return a usable CUDA device if available; otherwise fall back to CPU.
    - Does NOT hardcode 'cuda:1' so it respects CUDA_VISIBLE_DEVICES.
    - Avoids crashing when the driver/runtime is misaligned (Error 804).
    """
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")  # first visible GPU
    except Exception as e:
        print(f"[WARN] CUDA not usable, falling back to CPU: {e}")
    return torch.device("cpu")



class SurgicalPhaseDataset(Dataset):
    """
    CSV format: columns [id, label]
    id    => base name for a .png image (without extension)
    label => integer in [0..num_classes-1]
    """
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.samples = df[['id', 'label']].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        path_img = os.path.join(self.img_dir, img_id + ".png")
        image = Image.open(path_img).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def train_resnet(train_csv, val_csv, test_csv, img_dir,
                 out_model="best_resnet_ft.pth",
                 num_classes=7,
                 batch_size=16, lr=1e-4, epochs=20,
                 freeze_until=6):
    """
    Fine-tune ResNet-50 using train/val CSVs. Optionally evaluate on test_csv.
    - freeze_until: how many top-level ResNet children to freeze (approx. 0..7)
    """

    # 1) Transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds_train = SurgicalPhaseDataset(train_csv, img_dir, transform_train)
    ds_val   = SurgicalPhaseDataset(val_csv,   img_dir, transform_val_test)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=4)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Optional test dataset for final evaluation
    ds_test = None
    dl_test = None
    if os.path.isfile(test_csv):
        ds_test = SurgicalPhaseDataset(test_csv, img_dir, transform_val_test)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        print(f"[INFO] test_csv not found: {test_csv}. Test evaluation will be skipped.")

    device = get_device()

    # 2) Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    # Freeze early layers
    c = 0
    for name, child in model.named_children():
        if c < freeze_until:
            for param in child.parameters():
                param.requires_grad = False
        c += 1

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=3)

    best_acc = 0.0
    best_weights = None

    # Track per-epoch metrics
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Early stopping
    patience = 10
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(dl_train, desc=f"[Epoch {epoch+1}/{epochs} train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in dl_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outv = model(imgs)
                loss_val = criterion(outv, labels)
                val_running_loss += loss_val.item() * len(labels)

                predv = outv.argmax(dim=1)
                val_correct += (predv == labels).sum().item()
                val_total   += len(labels)
        val_loss = val_running_loss / val_total if val_total > 0 else 0
        val_acc  = val_correct / val_total if val_total > 0 else 0
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}. "
                  f"No improvement in val_acc for {patience} consecutive epochs.")
            break

    print(f"[Training complete] Best val_acc={best_acc:.4f}")
    if best_weights:
        torch.save(best_weights, out_model)
        print(f"Weights saved to {out_model}")

    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, len(val_loss_list) + 1),   val_loss_list,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Train Acc')
    plt.plot(range(1, len(val_acc_list) + 1),   val_acc_list,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.show()

    # Optional test evaluation
    if dl_test:
        model.load_state_dict(best_weights)
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        all_ids = []  # to store the id for CSV
        with torch.no_grad():
            test_index = 0
            for imgs, labels in dl_test:
                imgs, labels = imgs.to(device), labels.to(device)
                outs = model(imgs)
                loss_test = criterion(outs, labels)
                pred = outs.argmax(dim=1)

                test_correct += (pred == labels).sum().item()
                test_total   += len(labels)

                # Store for confusion matrix & F1
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                # Store the image IDs for CSV (aligned by batch index)
                bs = len(labels)
                for i in range(bs):
                    sample_id = ds_test.samples[test_index + i][0]
                    all_ids.append(sample_id)
                test_index += bs

        test_acc = test_correct / test_total if test_total > 0 else 0
        print(f"[Test] Accuracy = {test_acc:.4f}")

        # Confusion matrix & macro F1
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"F1-score (macro) on test set: {f1:.4f}")

        # Plot normalized confusion matrix
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Test')
        plt.colorbar()

        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, [str(i) for i in range(num_classes)], rotation=45)
        plt.yticks(tick_marks, [str(i) for i in range(num_classes)])

        # Add cell values
        threshold = cm.max() / 2.
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

        # Export predictions to CSV
        csv_filename = "predictions_resnet.csv"
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["img_id", "true_label", "predicted_label"])
            for i in range(len(all_preds)):
                writer.writerow([all_ids[i], all_labels[i], all_preds[i]])
        print(f"[INFO] Predictions saved to {csv_filename}")


def main():
    """
    Set paths and parameters here.
    """
    train_csv  = "train_ResNet_DB.csv"
    val_csv    = "val_ResNet_DB.csv"
    test_csv   = "test_ResNet_DB.csv"   # <--- if it does not exist, skip the final eval
    img_dir    = "images"
    out_model  = "best_spatial_study_weights.pth"

    num_classes   = 7
    batch_size    = 32
    lr            = 1e-4
    epochs        = 50
    freeze_until  = 6

    print("Starting ResNet fine-tuning with:")
    print(f"train_csv={train_csv}, val_csv={val_csv}, test_csv={test_csv}")
    train_resnet(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        img_dir=img_dir,
        out_model=out_model,
        num_classes=num_classes,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        freeze_until=freeze_until
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n[Total time] Script executed in {end_time - start_time:.2f} seconds")

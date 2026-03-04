# analyze2.py
"""
g2_analyze2.py - Visual analysis of Model 2: ResNet18 (from scratch)

This version uses ONLY public_data:
- Loads group2_data/public_data/*.npz
- Makes a stratified split into train/eval
- Trains on train split, evaluates on eval split

Produces two figures saved in group2_plots/:
  - matrix2.png : Normalized confusion matrix (8x8, varieties 1-8) on eval split
  - ci2.png     : Bootstrap 95% CI on accuracy on eval split

Note:
- All code that produced conv2.png is kept but fully commented out.
"""

import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# The convergence code uses torch, keep imports commented to avoid unnecessary deps/time.
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# --- paths ---
PUBLIC_DIR = os.path.join("group2_data", "public_data")
PLOT_DIR   = "group2_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- style ---
ACCENT  = "#7C3AED"   # purple
GREY    = "#6B7280"
BG      = "#F9FAFB"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
})

# --- helpers ---
def load_npz_folder(folder: str):
    """
    Load all .npz files in a folder and concatenate images + labels.

    Handles corner cases:
    - single image stored as (H,W,C) -> expands to (1,H,W,C)
    - scalar label stored as 0-D array -> converted to (1,) array
    """
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    imgs, labels = [], []

    for f in files:
        data = np.load(f, allow_pickle=False)

        # ---- images ----
        img_arr = None
        for key in ["images", "X", "data", "image"]:
            if key in data:
                img_arr = data[key]
                break

        if img_arr is None:
            for key in data.files:
                arr = data[key]
                if getattr(arr, "ndim", 0) >= 3:
                    img_arr = arr
                    break

        if img_arr is None:
            raise ValueError(f"No image-like array found in {f}")

        if img_arr.ndim == 3:
            img_arr = img_arr[None, ...]
        imgs.append(img_arr)

        # ---- labels ----
        y_arr = None
        for key in ["labels", "y", "label"]:
            if key in data:
                y_arr = data[key]
                break

        if y_arr is not None:
            if np.ndim(y_arr) == 0:
                y_arr = np.array([int(y_arr)], dtype=np.int64)
            labels.append(y_arr)

    X = np.concatenate(imgs, axis=0)
    y = np.concatenate(labels, axis=0) if labels else None
    return X, y

# --- import Model from group2_code2 ---
sys.path.insert(0, ".")
from group2_code2 import Model
# Convergence utilities kept but unused, so commented.
# from group2_code2 import (
#     NumpyImageDataset,
#     resnet18_no_torchvision,
#     stratified_split_indices,
#     seed_everything,
# )

# --- load data ---
print("[*] Loading public data...")
X_all, y_all_raw = load_npz_folder(PUBLIC_DIR)
if y_all_raw is None:
    raise ValueError("Labels not found in public_data npz files.")
y_all = y_all_raw.astype(int)
print(f"    All: {X_all.shape}, labels: {np.unique(y_all)}")

# --- split into train/eval ---
EVAL_RATIO = 0.20
SPLIT_SEED = 42

X_train, X_eval, y_train, y_eval = train_test_split(
    X_all, y_all,
    test_size=EVAL_RATIO,
    stratify=y_all,
    random_state=SPLIT_SEED
)
print(f"    Split: train={X_train.shape[0]}, eval={X_eval.shape[0]}")

# -----------------------------------------------------------------------------
# 1) CONVERGENCE CURVE (conv2.png) - NOT NEEDED
# Everything below is commented out as requested.
# -----------------------------------------------------------------------------
# print("\n[*] Training with convergence logging (train split only)...")
# seed_everything(42)
#
# DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CLASSES  = 8
# LABEL_OFFSET = 1  # labels are 1..8, CE needs 0..7
#
# BATCH_SIZE   = 48 if DEVICE.type == "cuda" else 24
# MAX_EPOCHS   = 25
# LR           = 3e-4
# WEIGHT_DECAY = 1e-4
# PATIENCE     = 6
# VAL_RATIO    = 0.15
#
# print(f"    Device: {DEVICE}")
#
# y_train_mapped = y_train.astype(np.int64) - LABEL_OFFSET
# tr_idx, va_idx = stratified_split_indices(y_train_mapped, VAL_RATIO, seed=42)
# print(f"    Internal split (for convergence): train={len(tr_idx)}, val={len(va_idx)}")
#
# train_ds = NumpyImageDataset(X_train[tr_idx], y_train_mapped[tr_idx], augment=True,  crop_size=224)
# val_ds   = NumpyImageDataset(X_train[va_idx], y_train_mapped[va_idx], augment=False, crop_size=224)
#
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
# val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
#
# net = resnet18_no_torchvision(num_classes=NUM_CLASSES).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
# )
#
# train_accs, val_accs, epochs_log = [], [], []
# best_val_acc = -1.0
# no_improve = 0
#
# for epoch in range(1, MAX_EPOCHS + 1):
#     net.train()
#     total, correct = 0, 0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         optimizer.zero_grad(set_to_none=True)
#         logits = net(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         correct += int((logits.argmax(1) == yb).sum().item())
#         total += int(xb.size(0))
#     train_acc = correct / max(total, 1)
#
#     net.eval()
#     v_total, v_correct = 0, 0
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             preds = net(xb).argmax(1)
#             v_correct += int((preds == yb).sum().item())
#             v_total += int(xb.size(0))
#     val_acc = v_correct / max(v_total, 1)
#
#     train_accs.append(train_acc)
#     val_accs.append(val_acc)
#     epochs_log.append(epoch)
#     print(f"    Epoch {epoch:02d}/{MAX_EPOCHS} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
#
#     scheduler.step(val_acc)
#
#     if val_acc > best_val_acc + 1e-4:
#         best_val_acc = val_acc
#         no_improve = 0
#     else:
#         no_improve += 1
#         if no_improve >= PATIENCE:
#             print(f"    Early stopping at epoch {epoch}.")
#             break
#
# import matplotlib.ticker as mticker
# ACCENT2 = "#D97706"
# fig, ax = plt.subplots(figsize=(8, 5))
# fig.suptitle("Model 2 - ResNet18 | Training Convergence",
#              fontsize=14, fontweight="bold", color="#111827")
# ax.plot(epochs_log, train_accs, color=ACCENT, linewidth=2, marker="o", markersize=4, label="Train Acc")
# ax.plot(epochs_log, val_accs,   color=ACCENT2, linewidth=2, marker="s", markersize=4, linestyle="--", label="Val Acc")
# ax.set_xlabel("Epoch", fontsize=11)
# ax.set_ylabel("Accuracy", fontsize=11)
# ax.set_title("Train vs Validation Accuracy per Epoch", fontsize=12)
# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
# ax.legend(fontsize=10, framealpha=0.7)
# ax.grid(axis="y", linestyle="--", alpha=0.5)
# fig.tight_layout()
#
# conv_path = os.path.join(PLOT_DIR, "conv2.png")
# fig.savefig(conv_path, dpi=150, bbox_inches="tight")
# plt.close(fig)
# print(f"    Saved -> {conv_path}")

# -----------------------------------------------------------------------------
# 2) FINAL MODEL - train on train split, evaluate on eval split
# -----------------------------------------------------------------------------
print("\n[*] Training final model (train split) and evaluating (eval split)...")
model = Model()
model.fit({"X": X_train, "y": y_train})
y_pred = model.predict({"X": X_eval})

acc_eval = accuracy_score(y_eval, y_pred)
print(f"    Eval accuracy: {acc_eval:.4f}")

# -----------------------------------------------------------------------------
# 3) NORMALIZED CONFUSION MATRIX (eval split)
# -----------------------------------------------------------------------------
print("\n[*] Plotting confusion matrix (eval split)...")
CLASSES = [f"Var {i}" for i in range(1, 9)]
cm = confusion_matrix(y_eval, y_pred, labels=list(range(1, 9)), normalize="true")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Purples", cmap="Purples", values_format=".2f")
ax.set_title("Model 2 - Normalized Confusion Matrix\n(ResNet18 from scratch, eval split)",
             fontsize=13, fontweight="bold", color="#111827", pad=12)
ax.set_xlabel("Predicted variety", fontsize=11)
ax.set_ylabel("True variety", fontsize=11)

for text in disp.text_.ravel():
    text.set_fontsize(9)

fig.tight_layout()
mat_path = os.path.join(PLOT_DIR, "matrix2.png")
fig.savefig(mat_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved -> {mat_path}")

# -----------------------------------------------------------------------------
# 4) BOOTSTRAP CI ON ACCURACY (eval split)
# -----------------------------------------------------------------------------
print("\n[*] Computing bootstrap CI on accuracy (eval split)...")
B = 1000
rng = np.random.default_rng(42)
N = len(y_eval)

acc_boots = []
for _ in range(B):
    idx = rng.integers(0, N, size=N)
    acc_boots.append(accuracy_score(y_eval[idx], y_pred[idx]))

acc_boots = np.array(acc_boots)
ci_lo = np.percentile(acc_boots, 2.5)
ci_hi = np.percentile(acc_boots, 97.5)

print(f"    Observed Acc = {acc_eval:.4f} | 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.hist(acc_boots, bins=50, color=ACCENT, alpha=0.75,
        edgecolor="white", linewidth=0.5)
ax.axvline(acc_eval, color="#DC2626", linewidth=2, linestyle="-",
           label=f"Observed Acc = {acc_eval:.4f}")
ax.axvline(ci_lo, color=GREY, linewidth=1.5, linestyle="--",
           label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi, color=GREY, linewidth=1.5, linestyle="--")

ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
ax.fill_betweenx([0, ymax], ci_lo, ci_hi, color=GREY, alpha=0.15)

ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Bootstrap frequency", fontsize=12)
ax.set_title("Model 2 - Bootstrap 95% CI on Accuracy\n(ResNet18 from scratch, B=1000, eval split)",
             fontsize=13, fontweight="bold", color="#111827")
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()

ci_path = os.path.join(PLOT_DIR, "ci2.png")
fig.savefig(ci_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved -> {ci_path}")

print("\n[OK] All plots for Group 2 Model 2 saved in", PLOT_DIR)

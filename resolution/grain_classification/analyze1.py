# analyze1.py
"""
g2_analyze1.py - Visual analysis of Model 1: HGB (HistGradientBoosting)

This version uses ONLY public_data:
- Loads group2_data/public_data/*.npz
- Makes a stratified split into train/eval
- Trains on train split, evaluates on eval split

Produces three figures saved in group2_plots/:
  - matrix1.png : Normalized confusion matrix (8x8, varieties 1-8) on eval split
  - ci1.png     : Bootstrap 95% CI on accuracy on eval split
  - conv1.png   : Training convergence (val accuracy vs boosting iterations)
"""

import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# --- paths ---
PUBLIC_DIR = os.path.join("group2_data", "public_data")
PLOT_DIR   = "group2_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- style ---
ACCENT  = "#2563EB"
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
    """Load all .npz files in a folder and concatenate images + labels."""
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    imgs, labels = [], []
    for f in files:
        data = np.load(f, allow_pickle=False)

        # images
        img_arr = None
        for key in ["images", "X", "data", "image"]:
            if key in data:
                img_arr = data[key]
                break
        if img_arr is None:
            for key in data.files:
                arr = data[key]
                if arr.ndim >= 3:
                    img_arr = arr
                    break
        if img_arr is None:
            raise ValueError(f"No image-like array found in {f}")

        if img_arr.ndim == 3:  # single image (H,W,C) -> add batch dim
            img_arr = img_arr[None, ...]
        imgs.append(img_arr)

        # labels
        y_arr = None
        for key in ["labels", "y", "label"]:
            if key in data:
                y_arr = data[key]
                break
        if y_arr is not None:
            if np.ndim(y_arr) == 0:  # scalar -> 1-element array
                y_arr = np.array([int(y_arr)], dtype=np.int64)
            labels.append(y_arr)

    X = np.concatenate(imgs, axis=0)
    y = np.concatenate(labels, axis=0) if labels else None
    return X, y

# --- import Model (feature extractor) ---
sys.path.insert(0, ".")
from group2_code1 import Model

# --- load data ---
print("[*] Loading public data...")
X_all, y_all_raw = load_npz_folder(PUBLIC_DIR)
if y_all_raw is None:
    raise ValueError("Labels not found in public_data npz files.")
y_all = y_all_raw.astype(int)

print(f"    All: {X_all.shape}, labels: {np.unique(y_all)}")

# --- split into train/eval (two distinct splits) ---
EVAL_RATIO = 0.20
SPLIT_SEED = 42

X_train, X_eval, y_train, y_eval = train_test_split(
    X_all, y_all,
    test_size=EVAL_RATIO,
    stratify=y_all,
    random_state=SPLIT_SEED
)
print(f"    Split: train={X_train.shape[0]}, eval={X_eval.shape[0]}")

# --- feature extraction (reuse Model internal method) ---
print("[*] Extracting hand-crafted features...")
m = Model()
X_train_feat = m._extract_features(X_train)
X_eval_feat  = m._extract_features(X_eval)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_eval_scaled  = scaler.transform(X_eval_feat)

print(f"    Feature dim: {X_train_feat.shape[1]}")

# --- HGB params ---
HGB_PARAMS = dict(
    loss="log_loss",
    max_iter=450,
    learning_rate=0.05,
    max_depth=20,
    min_samples_leaf=80,
    max_leaf_nodes=127,
    l2_regularization=3.5e-05,
    random_state=42,
)

# -----------------------------------------------------------------------------
# 1) CONVERGENCE CURVE
# Train on a sub-train split, track val accuracy vs iterations
# -----------------------------------------------------------------------------
print("\n[*] Computing convergence curve...")

X_cv_tr, X_cv_val, y_cv_tr, y_cv_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

checkpoints = list(range(20, 450, 30)) + [450]
acc_vals, iters = [], []

clf_ws = HistGradientBoostingClassifier(
    **{k: v for k, v in HGB_PARAMS.items() if k != "max_iter"},
    max_iter=1,
    warm_start=True,
)

for n in checkpoints:
    clf_ws.set_params(max_iter=n)
    clf_ws.fit(X_cv_tr, y_cv_tr)
    preds = clf_ws.predict(X_cv_val)
    acc = accuracy_score(y_cv_val, preds)
    acc_vals.append(acc)
    iters.append(n)
    print(f"    iter={n:4d}  val_acc={acc:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Model 1 - HGB | Training Convergence",
             fontsize=14, fontweight="bold", color="#111827")
ax.plot(iters, acc_vals, color=ACCENT, linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Number of boosting iterations", fontsize=11)
ax.set_ylabel("Accuracy (val fold)", fontsize=11)
ax.set_title("Validation Accuracy vs Iterations", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()

conv_path = os.path.join(PLOT_DIR, "conv1.png")
fig.savefig(conv_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved -> {conv_path}")

# -----------------------------------------------------------------------------
# 2) FINAL MODEL - train on train split, evaluate on eval split
# -----------------------------------------------------------------------------
print("\n[*] Training final model (train split) and evaluating (eval split)...")
clf_final = HistGradientBoostingClassifier(**HGB_PARAMS)
clf_final.fit(X_train_scaled, y_train)
y_pred = clf_final.predict(X_eval_scaled)

acc_eval = accuracy_score(y_eval, y_pred)
print(f"    Eval accuracy: {acc_eval:.4f}")

# -----------------------------------------------------------------------------
# 3) NORMALIZED CONFUSION MATRIX (8x8) on eval split
# -----------------------------------------------------------------------------
print("\n[*] Plotting confusion matrix (eval split)...")
CLASSES = [f"Var {i}" for i in range(1, 9)]
cm = confusion_matrix(y_eval, y_pred, labels=list(range(1, 9)), normalize="true")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
ax.set_title("Model 1 - Normalized Confusion Matrix\n(HGB + Hand-crafted features, eval split)",
             fontsize=13, fontweight="bold", color="#111827", pad=12)
ax.set_xlabel("Predicted variety", fontsize=11)
ax.set_ylabel("True variety", fontsize=11)
for text in disp.text_.ravel():
    text.set_fontsize(9)

fig.tight_layout()
mat_path = os.path.join(PLOT_DIR, "matrix1.png")
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
ax.set_title("Model 1 - Bootstrap 95% CI on Accuracy\n(HGB + Hand-crafted features, B=1000, eval split)",
             fontsize=13, fontweight="bold", color="#111827")
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()

ci_path = os.path.join(PLOT_DIR, "ci1.png")
fig.savefig(ci_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved -> {ci_path}")

print("\n[OK] All plots for Group 2 Model 1 saved in", PLOT_DIR)
"""
analyze_group4.py — Visual analysis of Group 4 model: Random Forest
                    for pollinator species classification.

Produces three figures saved in the project root:
  • matrix_group4.png  – Normalized confusion matrix (multi-class)
  • ci_group4.png      – Bootstrap 95% CI on accuracy (B=1000)
  • conv_group4.png    – Training convergence (accuracy vs n_estimators)

Data layout expected (project root):
  X_train.npz   → key: any first array (pre-extracted features, shape N×D)
  y_train.npy   → 1D array of integer labels
  X_test.npz    → same format as X_train.npz
  y_test.json   → list or dict of integer labels

Usage: python analyze_group4.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ── style ──────────────────────────────────────────────────────────────────────
ACCENT  = "#059669"   # green
ACCENT2 = "#D97706"   # amber
GREY    = "#6B7280"
BG      = "#F9FAFB"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
})

# ── load data ──────────────────────────────────────────────────────────────────
print("[*] Loading data…")

X_train = np.load("X_train.npy", allow_pickle=False)
X_test  = np.load("X_test.npy",  allow_pickle=False)
y_train = np.load("y_train.npy", allow_pickle=False)

with open("y_test.json", "r") as f:
    raw = json.load(f)
# y_test.json may be a list or a dict {"labels": [...]} or {"y_test": [...]}
if isinstance(raw, list):
    y_test = np.array(raw)
elif isinstance(raw, dict):
    key = next(iter(raw))          # take whatever the first key is
    y_test = np.array(raw[key])
else:
    raise ValueError(f"Unexpected format in y_test.json: {type(raw)}")
y_test = y_test.astype(int)

print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}  classes: {np.unique(y_train)}")
print(f"    X_test:  {X_test.shape}   y_test:  {y_test.shape}   classes: {np.unique(y_test)}")

# ── scaling ────────────────────────────────────────────────────────────────────
scaler   = RobustScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# ── shared RF hyperparameters (identical to group4.py) ────────────────────────
RF_PARAMS = dict(
    max_depth             = 22,
    min_samples_split     = 10,
    min_samples_leaf      = 2,
    max_features          = 0.9653946665869293,
    max_samples           = 0.5892208809109154,
    min_impurity_decrease = 9.94740876259007e-07,
    class_weight          = "balanced",
    random_state          = 42,
    n_jobs                = -1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONVERGENCE CURVE
#     Random Forest does not update iteratively, but sklearn exposes
#     staged predictions via warm_start: we grow the forest by adding
#     trees incrementally and record validation accuracy at each step.
#     This is the standard way to show the "n_estimators convergence"
#     curve for Random Forests.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Computing convergence curve…")

X_cv_tr, X_cv_val, y_cv_tr, y_cv_val = train_test_split(
    X_tr_sc, y_train, test_size=0.2,
    stratify=y_train, random_state=42
)

checkpoints = list(range(10, 324, 20)) + [324]
acc_vals, n_trees = [], []

clf_ws = RandomForestClassifier(
    **RF_PARAMS,
    n_estimators=10,
    warm_start=True,
)
for n in checkpoints:
    clf_ws.set_params(n_estimators=n)
    clf_ws.fit(X_cv_tr, y_cv_tr)
    preds = clf_ws.predict(X_cv_val)
    acc_vals.append(accuracy_score(y_cv_val, preds))
    n_trees.append(n)
    print(f"    n_estimators={n:4d}  val_acc={acc_vals[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Group 4 — Random Forest  |  Training Convergence",
             fontsize=14, fontweight="bold", color="#111827")
ax.plot(n_trees, acc_vals, color=ACCENT, linewidth=2,
        marker="o", markersize=4)
ax.set_xlabel("Number of trees", fontsize=11)
ax.set_ylabel("Accuracy (val fold)", fontsize=11)
ax.set_title("Validation Accuracy vs Number of Trees", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
conv_path = "conv_group4.png"
fig.savefig(conv_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {conv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FINAL MODEL — train on full training set, predict on test
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Training final model on full training set…")
clf_final = RandomForestClassifier(n_estimators=324, **RF_PARAMS)
clf_final.fit(X_tr_sc, y_train)
y_pred = clf_final.predict(X_te_sc)
print(f"    Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  NORMALIZED CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Plotting confusion matrix…")
classes_sorted = sorted(np.unique(np.concatenate([y_train, y_test])))
n_classes = len(classes_sorted)

# Build readable tick labels: use class index as-is (species IDs)
tick_labels = [str(c) for c in classes_sorted]

cm = confusion_matrix(y_test, y_pred,
                      labels=classes_sorted, normalize="true")

# Adapt figure size to number of classes
fig_size = max(6, n_classes * 0.9)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tick_labels)
disp.plot(ax=ax, cmap="Greens", colorbar=True,
          values_format=".2f")
ax.set_title("Group 4 — Normalized Confusion Matrix\n"
             "(Random Forest, Pollinator Species Classification)",
             fontsize=12, fontweight="bold", color="#111827", pad=12)
ax.set_xlabel("Predicted class", fontsize=10)
ax.set_ylabel("True class", fontsize=10)
fontsize_cell = max(6, 11 - n_classes // 3)
for text in disp.text_.ravel():
    text.set_fontsize(fontsize_cell)

fig.tight_layout()
mat_path = "matrix_group4.png"
fig.savefig(mat_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {mat_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  BOOTSTRAP CI ON ACCURACY (B=1000)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Computing bootstrap CI on accuracy…")
B        = 1000
rng      = np.random.default_rng(42)
N_test   = len(y_test)
acc_boots = []
for _ in range(B):
    idx = rng.integers(0, N_test, size=N_test)
    acc_boots.append(accuracy_score(y_test[idx], y_pred[idx]))

acc_boots = np.array(acc_boots)
acc_obs   = accuracy_score(y_test, y_pred)
ci_lo     = np.percentile(acc_boots, 2.5)
ci_hi     = np.percentile(acc_boots, 97.5)
print(f"    Observed Acc = {acc_obs:.4f}  |  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.hist(acc_boots, bins=50, color=ACCENT, alpha=0.75,
        edgecolor="white", linewidth=0.5)
ax.axvline(acc_obs, color="#DC2626", linewidth=2,  linestyle="-",
           label=f"Observed Acc = {acc_obs:.4f}")
ax.axvline(ci_lo,  color=GREY,     linewidth=1.5, linestyle="--",
           label=f"95% CI  [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi,  color=GREY,     linewidth=1.5, linestyle="--")
ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
ax.fill_betweenx([0, ymax], ci_lo, ci_hi, color=GREY, alpha=0.15)

ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Bootstrap frequency", fontsize=12)
ax.set_title("Group 4 — Bootstrap 95% CI on Accuracy\n"
             "(Random Forest, B=1000 resamples)",
             fontsize=13, fontweight="bold", color="#111827")
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()

ci_path = "ci_group4.png"
fig.savefig(ci_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {ci_path}")

print("\n[✔] All Group 4 plots saved in project root.")
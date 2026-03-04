"""
analyze2.py  —  Visual analysis of Model 2: MobileNetV2 + LightGBM
Produces three figures saved in ourplots/:
  • matrix2.png  – Normalized confusion matrix
  • ci2.png      – Bootstrap 95% confidence interval on F1-score
  • conv2.png    – Training convergence curve (F1 vs n_estimators)

Usage:
    python analyze2.py

Expects:
  • ourdata/train_data.h5, ourdata/train_labels.npy
  • ourdata/test_data.h5,  ourdata/test_labels.npy
  • ourcode2.py (Model class) in the same directory or sys.path
"""

import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "ourdata"
PLOT_DIR  = "ourplots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
ACCENT   = "#7C3AED"   # purple
ACCENT2  = "#D97706"   # amber
GREY     = "#6B7280"
BG       = "#F9FAFB"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
})

# ── load data ──────────────────────────────────────────────────────────────────
print("[*] Loading data…")
with h5py.File(os.path.join(DATA_DIR, "train_data.h5"), "r") as f:
    X_train = f["images"][:]
with h5py.File(os.path.join(DATA_DIR, "test_data.h5"), "r") as f:
    X_test = f["images"][:]
y_train = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ── import Model ───────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from ourcode2 import Model, add_temporal_features

model = Model(temporal_window=2, oversampling=True, ratio=0.15,
              threshold=0.5, tune_threshold=False)   # we handle threshold ourselves

# ── extract features ───────────────────────────────────────────────────────────
print("[*] Extracting MobileNetV2 + temporal features (train)…")
X_train_feat = model._preprocess(X_train)
print("[*] Extracting MobileNetV2 + temporal features (test)…")
X_test_feat  = model._preprocess(X_test)

# ── oversampling ───────────────────────────────────────────────────────────────
ratio           = model.ratio
current_ratio   = np.sum(y_train == 1) / np.sum(y_train == 0)
target_strategy = ratio / (1 - ratio)
if target_strategy > current_ratio:
    ros = RandomOverSampler(sampling_strategy=target_strategy, random_state=42)
    X_res, y_res = ros.fit_resample(X_train_feat, y_train)
else:
    X_res, y_res = X_train_feat, y_train
print(f"    After oversampling: {X_res.shape[0]} samples.")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONVERGENCE CURVE
#     LightGBM supports callbacks that log metrics on an eval set each round.
#     We train with log_loss on a held-out val fold, then recompute F1 at
#     checkpoints using predict_proba with num_iteration parameter.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Computing convergence curve…")

X_cv_tr, X_cv_val, y_cv_tr, y_cv_val = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# Record callback
class LogHistory(lgb.callback.CallbackEnv):
    pass

history = {}
clf_conv = lgb.LGBMClassifier(
    n_estimators=800,
    max_depth=8,
    num_leaves=63,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,
    n_jobs=model.n_cpus,
    random_state=42,
    verbose=-1,
)
clf_conv.fit(
    X_cv_tr, y_cv_tr,
    eval_set=[(X_cv_val, y_cv_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.record_evaluation(history)],
)

logloss_vals = history["valid_0"]["binary_logloss"]
n_rounds     = len(logloss_vals)

# F1 at checkpoints (every 50 rounds)
checkpoints = list(range(49, n_rounds, 50)) + [n_rounds - 1]
checkpoints = sorted(set(checkpoints))
f1_vals, rounds = [], []
for r in checkpoints:
    probs = clf_conv.predict_proba(X_cv_val, num_iteration=r + 1)[:, 1]
    preds = (probs >= 0.5).astype(int)
    f1_vals.append(f1_score(y_cv_val, preds, zero_division=0))
    rounds.append(r + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model 2 — MobileNetV2 + LightGBM  |  Training Convergence", fontsize=14, fontweight="bold", color="#111827")

ax = axes[0]
ax.plot(rounds, f1_vals, color=ACCENT, linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Number of boosting rounds", fontsize=11)
ax.set_ylabel("F1-score (val fold)", fontsize=11)
ax.set_title("F1-score vs Rounds", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle="--", alpha=0.5)

ax2 = axes[1]
ax2.plot(range(1, n_rounds + 1), logloss_vals, color=ACCENT2, linewidth=1.5)
ax2.set_xlabel("Number of boosting rounds", fontsize=11)
ax2.set_ylabel("Log-loss (val fold)", fontsize=11)
ax2.set_title("Log-loss vs Rounds", fontsize=12)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
conv_path = os.path.join(PLOT_DIR, "conv2.png")
fig.savefig(conv_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {conv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FINAL MODEL — train on full resampled set, predict on test
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Training final model on full training set…")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  NORMALIZED CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Plotting confusion matrix…")
cm     = confusion_matrix(y_test, y_pred, normalize="true")
labels = ["No Visitor", "Visitor"]

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format=".2%")
ax.set_title("Model 2 — Normalized Confusion Matrix\n(MobileNetV2 + LightGBM)", fontsize=13, fontweight="bold", color="#111827", pad=12)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label", fontsize=11)
for text in disp.text_.ravel():
    text.set_fontsize(14)

fig.tight_layout()
mat_path = os.path.join(PLOT_DIR, "matrix2.png")
fig.savefig(mat_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {mat_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  BOOTSTRAP CONFIDENCE INTERVAL ON F1-SCORE
#     Same protocol as Model 1 for fair comparison.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Computing bootstrap CI on F1…")
B        = 1000
rng      = np.random.default_rng(42)
N_test   = len(y_test)
f1_boots = []
for _ in range(B):
    idx = rng.integers(0, N_test, size=N_test)
    f   = f1_score(y_test[idx], y_pred[idx], zero_division=0)
    f1_boots.append(f)

f1_boots = np.array(f1_boots)
f1_obs   = f1_score(y_test, y_pred, zero_division=0)
ci_lo    = np.percentile(f1_boots, 2.5)
ci_hi    = np.percentile(f1_boots, 97.5)
print(f"    Observed F1 = {f1_obs:.4f}  |  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.hist(f1_boots, bins=50, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(f1_obs, color="#DC2626", linewidth=2,  linestyle="-",  label=f"Observed F1 = {f1_obs:.4f}")
ax.axvline(ci_lo,  color=GREY,     linewidth=1.5, linestyle="--", label=f"95% CI  [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi,  color=GREY,     linewidth=1.5, linestyle="--")

ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
ax.fill_betweenx([0, ymax], ci_lo, ci_hi, color=GREY, alpha=0.15)

ax.set_xlabel("F1-score", fontsize=12)
ax.set_ylabel("Bootstrap frequency", fontsize=12)
ax.set_title("Model 2 — Bootstrap 95% CI on F1-score\n(MobileNetV2 + LightGBM, B=1000 resamples)", fontsize=13, fontweight="bold", color="#111827")
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
ci_path = os.path.join(PLOT_DIR, "ci2.png")
fig.savefig(ci_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved → {ci_path}")

print("\n[✔] All plots for Model 2 saved in", PLOT_DIR)
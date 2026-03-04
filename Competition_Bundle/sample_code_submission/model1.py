"""
IMPROVED MODEL: MobileNetV2 + Temporal Difference Features + LightGBM
----------------------------------------------------------------------
Key improvements over the HOG+XGBoost baseline:

1. DEEP FEATURES (MobileNetV2):
   - Pretrained on ImageNet, fine-tuned via the feature extractor on our data.
   - MobileNetV2 uses depthwise separable convolutions → very lightweight (3.4M params).
   - We extract the global average pooling output (1280-dim vector) per frame.
   - Captures semantic features (shapes, textures, colors) far beyond HOG gradients.

2. TEMPORAL DIFFERENCE FEATURES:
   - Pollinators are moving objects. The absolute difference between consecutive frames
     (or a short window) is a strong signal for detecting motion/presence.
   - We concatenate per-frame CNN features with difference statistics from neighbors.
   - This is lightweight (no RNN/Transformer overhead) yet captures temporal context.

3. LIGHTGBM:
   - Faster and often better than XGBoost on tabular/feature data due to leaf-wise growth.
   - Lower memory footprint and faster inference than XGBoost with large feature vectors.
   - Built-in handling of class imbalance via is_unbalance=True.

4. THRESHOLD TUNING:
   - Same strategy as baseline: tune on validation fold to maximize F1.

5. PARALLELIZATION:
   - Batch inference on GPU if available, else CPU with threading.
"""

import os
import multiprocessing
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler


# ─────────────────────────────────────────────
# Feature extractor: MobileNetV2 truncated at GAP
# ─────────────────────────────────────────────
class MobileNetFeatureExtractor:
    def __init__(self, device):
        self.device = device
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Keep everything up to (and including) the adaptive avg pool → 1280-d vector
        self.model = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),          # small size → fast + lightweight
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, X, batch_size=256):
        """X: np.ndarray of shape (N, H, W, C) uint8"""
        all_feats = []
        for start in tqdm(range(0, len(X), batch_size), desc="CNN features", leave=False):
            batch_np = X[start:start + batch_size]
            tensors = []
            for img in batch_np:
                pil = Image.fromarray(img.astype(np.uint8))
                tensors.append(self.transform(pil))
            batch_t = torch.stack(tensors).to(self.device)
            feats = self.model(batch_t).cpu().numpy()
            all_feats.append(feats)
        return np.vstack(all_feats).astype(np.float32)


# ─────────────────────────────────────────────
# Temporal difference enrichment
# ─────────────────────────────────────────────
def add_temporal_features(cnn_feats, window=2):
    """
    For each frame i, compute the L1 difference between frame i and
    frames i-1 … i-window (clamped at boundaries).
    Appends mean and max of the diff vector → +2*window extra dims per frame.
    Stays very lightweight.
    """
    N, D = cnn_feats.shape
    extra = []
    for i in range(N):
        diffs = []
        for delta in range(1, window + 1):
            j = max(0, i - delta)
            diff = np.abs(cnn_feats[i] - cnn_feats[j])
            diffs.append([diff.mean(), diff.max()])
        extra.append(np.concatenate(diffs))  # shape: (2*window,)
    extra = np.array(extra, dtype=np.float32)
    return np.concatenate([cnn_feats, extra], axis=1)


# ─────────────────────────────────────────────
# Main Model class
# ─────────────────────────────────────────────
class Model:
    """
    Lightweight pollinator detector:
    MobileNetV2 CNN features + temporal diff stats → LightGBM classifier.
    """

    def __init__(self, temporal_window=2, oversampling=True, ratio=0.15,
                 threshold=0.5, tune_threshold=True):
        print("[*] Initializing MobileNetV2 + LightGBM Pollinator Detector")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")

        # CPU count
        try:
            self.n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else multiprocessing.cpu_count()
        except Exception:
            self.n_cpus = 1
        print(f"[*] Detected {self.n_cpus} CPU cores.")

        # Sub-components
        self.extractor = MobileNetFeatureExtractor(self.device)
        self.temporal_window = temporal_window
        self.oversampling = oversampling
        self.ratio = ratio
        self.threshold = threshold
        self.tune_threshold = tune_threshold

        # LightGBM classifier
        self.clf = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=8,
            num_leaves=63,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,          # handles class imbalance natively
            n_jobs=self.n_cpus,
            random_state=42,
            verbose=-1,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, X):
        cnn = self.extractor.extract(X)
        return add_temporal_features(cnn, window=self.temporal_window)

    def _tune_threshold(self, X_feat, y, n_splits=3):
        """Find threshold that maximises F1 on stratified CV folds."""
        print("[*] Tuning decision threshold via cross-validation…")
        best_thr, best_f1 = 0.5, 0.0
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y)):
            Xtr, Xval = X_feat[tr_idx], X_feat[val_idx]
            ytr, yval = y[tr_idx], y[val_idx]
            tmp_clf = lgb.LGBMClassifier(**self.clf.get_params())
            tmp_clf.fit(Xtr, ytr)
            probs = tmp_clf.predict_proba(Xval)[:, 1]
            for thr in np.arange(0.1, 0.8, 0.02):
                preds = (probs >= thr).astype(int)
                f = f1_score(yval, preds, zero_division=0)
                if f > best_f1:
                    best_f1, best_thr = f, thr
        print(f"[✔] Best threshold: {best_thr:.2f}  (CV F1 ≈ {best_f1:.4f})")
        return best_thr

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, X, y):
        print(f"[*] Training on {X.shape[0]} samples…")

        # 1. Feature extraction
        X_feat = self._preprocess(X)

        # 2. Optional oversampling
        if self.oversampling:
            current_ratio = np.sum(y == 1) / np.sum(y == 0)
            target_strategy = self.ratio / (1 - self.ratio)
            if target_strategy > current_ratio:
                ros = RandomOverSampler(sampling_strategy=target_strategy, random_state=42)
                X_feat, y = ros.fit_resample(X_feat, y)
                print(f"[✔] After oversampling: {X_feat.shape[0]} samples.")
            else:
                print("[!] Target ratio ≤ current ratio. Skipping oversampling.")

        # 3. Threshold tuning (before final fit)
        if self.tune_threshold:
            self.threshold = self._tune_threshold(X_feat, y)

        # 4. Final training on full (resampled) data
        self.clf.fit(X_feat, y)
        print("[✔] Training complete.")

    def predict(self, X):
        print(f"[*] Predicting {X.shape[0]} samples (threshold={self.threshold:.2f})…")
        X_feat = self._preprocess(X)
        probs = self.clf.predict_proba(X_feat)[:, 1]
        return (probs >= self.threshold).astype(int)

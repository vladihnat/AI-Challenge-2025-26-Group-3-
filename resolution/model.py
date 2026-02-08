# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# Created on: 30 Jan, 2026
# Updated on: 08 Feb, 2026

"""
Grain variety classifier (8 classes) based on handcrafted descriptors + tree ensembles

--------
Overview

This model is designed for a setting where each sample is a pre-cropped 224x224x3
"RGB-like" representation of an original hyperspectral grain cube. The goal is to
distinguish visually similar varieties using cues that are mostly:
- spectral / intensity distribution differences across channels,
- simple morphological differences (grain size/elongation),
- texture / surface patterns,
while remaining robust to background artifacts, illumination changes, and rotations
The pipeline has two main stages:
1) Feature extraction (deterministic, no learned CNN).
2) Classification using a blend of two complementary sklearn models

----------------------------------------------
1) Feature extraction (handcrafted, mask-aware)

The feature vector is built by combining several families of descriptors

A) Foreground masking and cropping
   - A grain mask is estimated on a grayscale image (mean over channels).
   - Two threshold candidates are tried:
     * percentile-based threshold (robust, simple),
     * Otsu threshold (adaptive, data-driven).
   - The chosen mask is the one whose foreground area ratio looks "plausible"
     for a centered grain crop (heuristic: not too small, not too large).
   - Guardrails relax the threshold if the mask is too small, and fall back to a
     safer percentile if it is pathological
   Motivation:
   - Many descriptors are polluted by the background. A mask focuses the statistics,
     histograms, and textures on grain pixels only.
   - Cropping around the mask reduces irrelevant borders and improves invariance to
     background position

B) Color / intensity histograms (per channel, masked)
   - For each channel, a normalized histogram of intensities in [0, 1] is computed.
   - Histograms capture distributional spectral signatures beyond mean/std
   Motivation:
   - In hyperspectral-derived RGB-like images, variety differences often appear as
     subtle shifts in intensity distributions rather than obvious shapes

C) Robust per-channel statistics (masked)
   For each channel we compute:
   - mean, std, median, min, max,
   - several percentiles (5, 10, 25, 75, 90, 95),
   - skewness and kurtosis (distribution shape).
   Additionally we compute inter-channel relations:
   - mean ratios (R/G, R/B, G/B) and mean differences
   Motivation:
   - Percentiles + skew/kurt describe asymmetry / heavy tails, which are common in
     natural textures and reflectance patterns.
   - Ratios are often more stable than absolute values when illumination varies

D) Shape descriptors (from the binary mask)
   - area ratio, bounding-box fill ratio,
   - normalized bbox size and aspect ratio,
   - centroid (normalized),
   - second-order central moments and derived elongation proxy,
   - a simple perimeter proxy (boundary pixel count)
   Motivation:
   - Some grain varieties differ by size and elongation.
   - Using mask-derived shape features yields rotation/translation robustness

E) Simple texture features (masked gradients)
   - Gradient magnitude summary statistics per channel on masked pixels,
   - A weighted orientation histogram (coarse HOG-like descriptor)
   Motivation:
   - Surface texture / ridges can be discriminative.
   - Gradients provide a lightweight way to capture texture without CNNs.

F) Radial intensity profile (rotation-invariant)
   - For each channel, compute mean intensity in concentric rings around the mask
     centroid (fixed number of radial bins)
   Motivation:
   - Grain images are typically centered; radial profiles capture systematic
     "center vs border" reflectance patterns and are invariant to rotation

G) Masked average-pooled downsample descriptor
   - The cropped region is pooled to a small grid (ds_h x ds_w), computing masked
     averages per cell and channel, then flattened
   Motivation:
   - This provides a coarse spatial layout of intensities (where the grain is
     brighter/darker) while keeping dimensionality manageable

-------------------------------------------------------------------------
2) Classification: two complementary tree ensembles + probability blending

The model uses two classifiers trained on the same extracted features
A) ExtraTreesClassifier (Extremely Randomized Trees) on raw features
   - Good default for heterogeneous features (mixture of stats, histograms, shapes).
   - Handles non-linear interactions, does not require feature scaling.
   - High-variance / low-bias ensemble tends to work well on medium-size datasets.
   - class_weight="balanced" helps when varieties are not equally represented

B) HistGradientBoostingClassifier on PCA-compressed features
   - Trained on standardized features (StandardScaler) followed by PCA.
   - PCA reduces feature correlation and noise, and makes boosting more stable.
   - Gradient boosting can capture different decision boundaries than ExtraTrees.
   Why StandardScaler + PCA here:
   - PCA is scale-sensitive, so features are standardized first.
   - PCA reduces the risk that high-dimensional, correlated descriptors dominate,
     and may improve generalization for boosting models

C) Probability-level blending (soft voting)
   - We compute class probabilities from both models and blend them:
         P = w * P_ET + (1 - w) * P_HGB
   - The final prediction is argmax(P).
   - A fixed blend weight avoids expensive cross-validation during training
   Motivation:
   - ExtraTrees and gradient boosting often make different errors; blending tends to
     improve accuracy and balanced accuracy compared to either model alone
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("[*] - Initializing Classifier (sklearn-only, improved v2)")

        # Strong tree ensemble on raw handcrafted features
        self.model_et = ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

        # Gradient boosting on a PCA-compressed feature space
        self.model_hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.07,
            max_depth=9,
            max_iter=420,
            l2_regularization=1e-3,
            max_bins=255,
            random_state=42
        )

        # Fixed blend weight (no expensive CV)
        # If HGB improves with PCA, 0.55/0.45 is often a decent starting point.
        self.blend_w = 0.55  # weight for ExtraTrees

        # PCA pipeline for HGB only
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=256, random_state=42)
        self.use_pca_for_hgb = True

        # Histogram settings
        self.n_bins = 24

        # Spatial downsample descriptor size
        self.ds_h, self.ds_w = 20, 20

        # Masking settings
        self.mask_percentile = 20
        self.min_fg_pixels = 1800
        self.bbox_margin = 6

        # Texture settings
        self.orient_bins = 9

        # Radial profile settings (rotation-invariant)
        self.radial_bins = 12

    # ----------------------------------------
    # Helpers: normalization / mask / crop
    # ----------------------------------------
    def _to_float01(self, img):
        """
        Convert image to float32 in [0, 1] range.

        Parameters
        ----------
        img: numpy array of shape (H, W, C)

        Returns
        -------
        imgf: numpy array of shape (H, W, C) in float32, values in [0, 1]
        """
        imgf = img.astype(np.float32, copy=False)
        if imgf.max() > 1.5:
            imgf = imgf / 255.0
        imgf = np.clip(imgf, 0.0, 1.0)
        return imgf

    def _otsu_threshold(self, gray, nbins=256):
        """
        Compute Otsu threshold on grayscale image (no external deps).

        Parameters
        ----------
        gray: numpy array (H, W) in [0, 1]
        nbins: int

        Returns
        -------
        t: float threshold in [0, 1]
        """
        g = gray.ravel()
        g = np.clip(g, 0.0, 1.0)

        hist, bin_edges = np.histogram(g, bins=nbins, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        if hist.sum() <= 0:
            return 0.5

        p = hist / hist.sum()
        omega = np.cumsum(p)
        mu = np.cumsum(p * (np.arange(nbins)))

        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)

        k = int(np.nanargmax(sigma_b2))
        # Convert bin index to threshold value
        t = (bin_edges[k] + bin_edges[k + 1]) * 0.5
        return float(t)

    def _make_mask(self, imgf):
        """
        Build a foreground mask for the grain using two candidates:
        - percentile threshold
        - Otsu threshold
        We pick the mask whose area ratio looks most plausible.

        Parameters
        ----------
        imgf: numpy array (H, W, C) in [0, 1]

        Returns
        -------
        mask: boolean numpy array of shape (H, W)
        """
        gray = imgf.mean(axis=2)

        # Candidate 1: percentile
        t1 = np.percentile(gray, self.mask_percentile)
        m1 = gray > t1

        # Candidate 2: Otsu
        t2 = self._otsu_threshold(gray)
        m2 = gray > t2

        total = gray.size
        a1 = float(m1.sum()) / float(total + 1e-8)
        a2 = float(m2.sum()) / float(total + 1e-8)

        # Heuristic: grain typically occupies a moderate fraction.
        # We prefer masks between 0.05 and 0.70, and closer to 0.25.
        def score_area(a):
            if a < 0.02 or a > 0.90:
                return 1e9
            if a < 0.05:
                return 10.0 + (0.05 - a) * 100.0
            if a > 0.70:
                return 10.0 + (a - 0.70) * 100.0
            return abs(a - 0.25)

        s1 = score_area(a1)
        s2 = score_area(a2)
        mask = m1 if s1 <= s2 else m2

        # Guardrails
        fg = int(mask.sum())
        if fg < self.min_fg_pixels:
            # Relax a bit
            t = np.percentile(gray, max(self.mask_percentile - 10, 5))
            mask = gray > t

        fg = int(mask.sum())
        if fg < 200 or fg > 0.95 * total:
            # Ultimate fallback
            t = np.percentile(gray, 35)
            mask = gray > t

        return mask

    def _bbox_from_mask(self, mask):
        """
        Compute bounding box around True pixels in a mask, with a fixed margin.

        Parameters
        ----------
        mask: boolean numpy array of shape (H, W)

        Returns
        -------
        r0, r1, c0, c1: int
            bounding box coordinates (inclusive start, exclusive end)
        """
        h, w = mask.shape
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 0, h, 0, w

        r0 = max(int(ys.min()) - self.bbox_margin, 0)
        r1 = min(int(ys.max()) + self.bbox_margin + 1, h)
        c0 = max(int(xs.min()) - self.bbox_margin, 0)
        c1 = min(int(xs.max()) + self.bbox_margin + 1, w)
        return r0, r1, c0, c1

    # ----------------------------------------
    # Features: robust color statistics / histograms on grain pixels
    # ----------------------------------------
    def _masked_channel_values(self, imgf, mask, c):
        """
        Returns channel values on masked pixels, safe fallback if empty.
        """
        vals = imgf[:, :, c][mask]
        if vals.size == 0:
            vals = imgf[:, :, c].ravel()
        return vals

    def _color_hist(self, imgf, mask):
        """
        Color histograms per channel on masked pixels.

        Returns
        -------
        hist: 1D numpy array
        """
        feats = []
        for c in range(3):
            vals = self._masked_channel_values(imgf, mask, c)
            hist, _ = np.histogram(vals, bins=self.n_bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)
            feats.append(hist)
        return np.concatenate(feats)

    def _robust_stats(self, imgf, mask):
        """
        Robust per-channel stats on masked pixels.

        Returns
        -------
        stats: 1D numpy array
        """
        feats = []
        for c in range(3):
            v = self._masked_channel_values(imgf, mask, c)

            mean = float(np.mean(v))
            std = float(np.std(v) + 1e-8)

            # Skewness and kurtosis (simple, no scipy)
            z = (v - mean) / std
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))

            feats.extend([
                mean,
                float(np.std(v)),
                float(np.median(v)),
                float(np.percentile(v, 5)),
                float(np.percentile(v, 10)),
                float(np.percentile(v, 25)),
                float(np.percentile(v, 75)),
                float(np.percentile(v, 90)),
                float(np.percentile(v, 95)),
                float(np.min(v)),
                float(np.max(v)),
                skew,
                kurt,
            ])

        # Simple inter-channel ratios on masked pixels (mean ratios)
        # Helps when spectral signature is mainly relative differences.
        v0 = self._masked_channel_values(imgf, mask, 0)
        v1 = self._masked_channel_values(imgf, mask, 1)
        v2 = self._masked_channel_values(imgf, mask, 2)

        m0 = float(np.mean(v0))
        m1 = float(np.mean(v1))
        m2 = float(np.mean(v2))

        feats.extend([
            m0 / (m1 + 1e-8),
            m0 / (m2 + 1e-8),
            m1 / (m2 + 1e-8),
            (m0 - m1),
            (m0 - m2),
            (m1 - m2),
        ])

        return np.array(feats, dtype=np.float32)

    # ----------------------------------------
    # Features: shape + moments from the mask
    # ----------------------------------------
    def _shape_features(self, mask):
        """
        Shape descriptors based on the binary mask.

        Returns
        -------
        shape_feats: 1D numpy array
        """
        h, w = mask.shape
        area = float(mask.sum())
        area_ratio = area / float(h * w + 1e-8)

        ys, xs = np.where(mask)
        if ys.size == 0:
            return np.zeros(14, dtype=np.float32)

        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1

        bbox_h = float(r1 - r0)
        bbox_w = float(c1 - c0)
        bbox_area = bbox_h * bbox_w
        bbox_fill = area / float(bbox_area + 1e-8)

        aspect = bbox_w / float(bbox_h + 1e-8)

        # Centroid (normalized)
        cy = float(ys.mean()) / float(h + 1e-8)
        cx = float(xs.mean()) / float(w + 1e-8)

        # Central moments (normalized)
        y0 = ys.astype(np.float32) - ys.mean()
        x0 = xs.astype(np.float32) - xs.mean()
        mu20 = float(np.mean(x0 * x0))
        mu02 = float(np.mean(y0 * y0))
        mu11 = float(np.mean(x0 * y0))

        # Covariance eigenvalues (elongation / eccentricity proxy)
        cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float32)
        cov += 1e-6 * np.eye(2, dtype=np.float32)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(eig)
        elong = float((eig[1] + 1e-8) / (eig[0] + 1e-8))
        spread = float(eig.sum())

        # Perimeter proxy: count boundary pixels (4-neighborhood)
        m = mask.astype(np.uint8)
        up = np.zeros_like(m); up[1:, :] = m[:-1, :]
        dn = np.zeros_like(m); dn[:-1, :] = m[1:, :]
        lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
        rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
        boundary = (m == 1) & ((up + dn + lf + rt) < 4)
        perim = float(boundary.sum())
        perim_norm = perim / float(np.sqrt(area) + 1e-8)

        return np.array([
            area_ratio,
            bbox_fill,
            bbox_h / float(h + 1e-8),
            bbox_w / float(w + 1e-8),
            aspect,
            cy,
            cx,
            mu20,
            mu02,
            mu11,
            elong,
            spread,
            perim_norm,
            perim / float(h + w + 1e-8),
        ], dtype=np.float32)

    # ----------------------------------------
    # Features: simple texture via gradients
    # ----------------------------------------
    def _texture_features(self, imgf, mask):
        """
        Texture features using gradients.

        Returns
        -------
        tex: 1D numpy array
        """
        gx = imgf[:, 1:, :] - imgf[:, :-1, :]
        gy = imgf[1:, :, :] - imgf[:-1, :, :]

        h = min(gx.shape[0], gy.shape[0])
        w = min(gx.shape[1], gy.shape[1])
        gx = gx[:h, :w, :]
        gy = gy[:h, :w, :]
        m = mask[:h, :w]

        mag = np.sqrt(gx * gx + gy * gy)

        feats = []
        for c in range(3):
            v = mag[:, :, c][m]
            if v.size == 0:
                v = mag[:, :, c].ravel()
            feats.extend([
                float(np.mean(v)),
                float(np.std(v)),
                float(np.median(v)),
                float(np.percentile(v, 75)),
                float(np.percentile(v, 90)),
            ])

        mag_sum = mag.sum(axis=2)
        gx_sum = gx.sum(axis=2)
        gy_sum = gy.sum(axis=2)
        ang = np.arctan2(gy_sum, gx_sum)

        ang_v = ang[m]
        w_v = mag_sum[m]
        if ang_v.size == 0:
            ang_v = ang.ravel()
            w_v = mag_sum.ravel()

        ang01 = (ang_v + np.pi) / (2.0 * np.pi)
        hist, _ = np.histogram(ang01, bins=self.orient_bins, range=(0.0, 1.0), weights=w_v)
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)

        feats.extend(hist.tolist())
        return np.array(feats, dtype=np.float32)

    # ----------------------------------------
    # Features: radial profile (rotation-invariant)
    # ----------------------------------------
    def _radial_profile(self, imgf, mask):
        """
        Compute radial mean intensity profile per channel on masked pixels.

        Parameters
        ----------
        imgf: numpy array (H, W, 3)
        mask: boolean numpy array (H, W)

        Returns
        -------
        prof: 1D numpy array
        """
        ys, xs = np.where(mask)
        if ys.size < 50:
            return np.zeros(self.radial_bins * 3, dtype=np.float32)

        cy = ys.mean()
        cx = xs.mean()

        dy = ys.astype(np.float32) - cy
        dx = xs.astype(np.float32) - cx
        r = np.sqrt(dx * dx + dy * dy)
        rmax = float(np.max(r) + 1e-8)

        # Bin radii in [0, 1]
        rb = np.floor((r / rmax) * self.radial_bins).astype(np.int32)
        rb = np.clip(rb, 0, self.radial_bins - 1)

        feats = []
        for c in range(3):
            vals = imgf[:, :, c][mask]
            sums = np.zeros(self.radial_bins, dtype=np.float32)
            cnts = np.zeros(self.radial_bins, dtype=np.float32)
            for k in range(rb.size):
                b = rb[k]
                sums[b] += vals[k]
                cnts[b] += 1.0
            means = sums / (cnts + 1e-8)
            feats.append(means)

        return np.concatenate(feats, axis=0).astype(np.float32)

    # ----------------------------------------
    # Features: compact spatial descriptor (masked average pooling)
    # ----------------------------------------
    def _downsample_masked(self, imgf, mask, out_h, out_w):
        """
        Average-pool downsample to (out_h, out_w, C) using masked averages.

        Returns
        -------
        pooled_flat: 1D numpy array
        """
        H, W, C = imgf.shape
        bh = H // out_h
        bw = W // out_w

        if bh <= 0 or bw <= 0:
            small = imgf[:out_h, :out_w, :]
            if small.shape[0] < out_h or small.shape[1] < out_w:
                pad_h = out_h - small.shape[0]
                pad_w = out_w - small.shape[1]
                small = np.pad(small, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            return small.reshape(-1).astype(np.float32)

        H2 = bh * out_h
        W2 = bw * out_w
        img2 = imgf[:H2, :W2, :]
        m2 = mask[:H2, :W2].astype(np.float32)

        imgm = img2 * m2[:, :, None]
        sum_blocks = imgm.reshape(out_h, bh, out_w, bw, C).sum(axis=(1, 3))
        cnt_blocks = m2.reshape(out_h, bh, out_w, bw).sum(axis=(1, 3))
        cnt_blocks = cnt_blocks[:, :, None]
        pooled = sum_blocks / (cnt_blocks + 1e-8)

        return pooled.reshape(-1).astype(np.float32)

    # ----------------------------------------
    # Main feature extraction
    # ----------------------------------------
    def _extract_features_one(self, img):
        """
        Extract features from a single image.

        Returns
        -------
        feat: 1D numpy array
        """
        imgf = self._to_float01(img)
        mask = self._make_mask(imgf)

        r0, r1, c0, c1 = self._bbox_from_mask(mask)
        crop = imgf[r0:r1, c0:c1, :]
        m_crop = mask[r0:r1, c0:c1]

        if crop.size == 0 or m_crop.size == 0:
            crop = imgf
            m_crop = mask

        if int(m_crop.sum()) < 50:
            m_crop = np.ones(m_crop.shape, dtype=bool)

        # Features
        hist = self._color_hist(crop, m_crop)
        stats = self._robust_stats(crop, m_crop)
        shape = self._shape_features(mask)  # global shape on full mask
        tex = self._texture_features(crop, m_crop)
        radial = self._radial_profile(crop, m_crop)
        ds = self._downsample_masked(crop, m_crop, self.ds_h, self.ds_w)

        feat = np.concatenate([hist, stats, shape, tex, radial, ds], axis=0)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feat

    def _extract_features(self, X):
        """
        Extract features from images.

        Parameters
        ----------
        X: numpy array of shape (n_samples, H, W, C)

        Returns
        -------
        features: numpy array of shape (n_samples, n_features)
        """
        if isinstance(X, list):
            X = np.array(X)

        feats = []
        for i in range(X.shape[0]):
            feats.append(self._extract_features_one(X[i]))
        return np.stack(feats, axis=0).astype(np.float32)

    # ----------------------------------------
    # Fit / Predict API
    # ----------------------------------------
    def fit(self, train_data):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_data: dict
            contains train data and labels
            - 'X': numpy array of images (n_samples, height, width, channels)
            - 'y': numpy array of labels (n_samples,)

        Returns
        -------
        None
        """
        print("[*] - Training Classifier on the train set")

        X = train_data["X"]
        y = train_data["y"]

        print("[*] - Extracting features...")
        X_feat = self._extract_features(X)
        print(f"[*] - Extracted {X_feat.shape[1]} features per sample")

        # Train ExtraTrees on raw features
        print("[*] - Training ExtraTrees...")
        self.model_et.fit(X_feat, y)

        # Train HGB on PCA features
        print("[*] - Training HistGradientBoosting...")
        if self.use_pca_for_hgb:
            X_scaled = self.scaler.fit_transform(X_feat)
            X_pca = self.pca.fit_transform(X_scaled)
            self.model_hgb.fit(X_pca, y)
        else:
            self.model_hgb.fit(X_feat, y)

        print("[*] - Training complete")

    def _align_proba(self, proba, model_classes, target_classes):
        """
        Align predicted probabilities to a target class ordering.

        Parameters
        ----------
        proba: numpy array (n_samples, K_model)
        model_classes: array-like of labels for columns in proba
        target_classes: array-like of labels in desired order

        Returns
        -------
        aligned: numpy array (n_samples, K_target)
        """
        model_classes = list(model_classes)
        target_classes = list(target_classes)

        aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float32)
        for j, lab in enumerate(target_classes):
            if lab in model_classes:
                k = model_classes.index(lab)
                aligned[:, j] = proba[:, k]
            else:
                aligned[:, j] = 0.0
        return aligned

    def predict(self, test_data):
        """
        This function predicts labels on test data.

        Parameters
        ----------
        test_data: dict
            contains test data
            - 'X': numpy array of images (n_samples, height, width, channels)

        Returns
        -------
        y: 1D numpy array
            predicted labels
        """
        print("[*] - Predicting test set using trained Classifier")

        X = test_data["X"]

        X_feat = self._extract_features(X)

        p_et = self.model_et.predict_proba(X_feat)

        if self.use_pca_for_hgb:
            X_scaled = self.scaler.transform(X_feat)
            X_pca = self.pca.transform(X_scaled)
            p_hgb = self.model_hgb.predict_proba(X_pca)
        else:
            p_hgb = self.model_hgb.predict_proba(X_feat)

        # Align class order (blend expects same ordering)
        classes = self.model_et.classes_
        p_hgb = self._align_proba(p_hgb, self.model_hgb.classes_, classes)

        # Blend probabilities
        w = float(self.blend_w)
        p = w * p_et + (1.0 - w) * p_hgb
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum(axis=1, keepdims=True)

        y_pred = classes[np.argmax(p, axis=1)]

        print(f"[*] - Predicted {len(y_pred)} samples")
        return y_pred
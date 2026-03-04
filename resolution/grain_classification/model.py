"""
Model Enhancement Summary:
--------------------------
This updated model improves the initial baseline by implementing a 
HistGradientBoostingClassifier, replacing the previous RandomForest approach.

Key improvements include:
- Replaced RandomForest with HistGradientBoostingClassifier for better performance and faster training.
  Gradient Boosting usually work better than ensemble methods like Random Forest on structured data,
  and the histogram-based version is optimized for speed and memory efficiency (+ compatible with 
  scikit-learn as the group 2 didn't import XGBoost in their docker image).
- Removed PCA. We kept their features but removed the PCA that lost a lot of information. HGB is already 
  a powerful model that can handle high-dimensional data, so PCA is not necessary and was likely hurting performance.
- Tuned hyperparameters of the HistGradientBoostingClassifier to improve accuracy. We used a
  random search with a wide range of values for max_iter, learning_rate, max_depth, min_samples_leaf, 
  max_leaf_nodes, and l2_regularization, creating 20 differents random combinations and testing them 
  on the validation set. The best combination was then used in the final model.
- Parallelized feature extraction using joblib to speed up processing of large batches of images. 
  This allows us to efficiently extract features from all images in the training and test sets, 
  even when they contain thousands of samples. Also added a save option for the extracted features,
  to avoid re-extracting them every time we train the model, which can save a lot of time during development.

Updated on: 9 Feb, 2026
"""
# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# Created on: 30 Jan, 2026

# ----------------------------------------
# Imports
# ----------------------------------------
import os
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self, max_iter=450, max_leaf_nodes=127, learning_rate=0.05, max_depth=20, min_samples_leaf=80, l2_regularization=3.5e-05, class_weight=None, early_stopping=False):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        max_iter: int, optional
            The maximum number of iterations for the classifier (default is 450).
        max_leaf_nodes: int, optional
            The maximum number of leaf nodes for the trees (default is 127).
        learning_rate: float, optional
            The learning rate for the classifier (default is 0.05).
        max_depth: int, optional
            The maximum depth of the trees (default is 20).
        min_samples_leaf: int, optional
            The minimum number of samples required to be at a leaf node (default is 80).
        l2_regularization: float, optional
            The L2 regularization parameter (default is 3.5e-05).
        class_weight: dict or 'balanced', optional
            Weights associated with classes in the form {class_label: weight} or 'balanced' (default is None).
        early_stopping: bool, optional
            Whether to use early stopping (default is False).

        Returns
        -------
        None
        """
        print("[*] - Initializing Classifier")

        # Optimized HistGradientBoostingClassifier with tuned hyperparameters
        self.clf = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=max_iter,         
            learning_rate=learning_rate,     
            max_depth=max_depth,         
            min_samples_leaf=min_samples_leaf,  
            max_leaf_nodes=max_leaf_nodes,       
            l2_regularization=l2_regularization, 
            class_weight=class_weight,
            early_stopping=early_stopping,  
            random_state=42,
        )

        # Scaler for normalizing features
        self.scaler = StandardScaler()

        # PCA for dimensionality reduction (faster training, less overfitting)
        self.pca = PCA(n_components=100, random_state=42)
        self.use_pca = True

        # Number of histogram bins per channel
        self.n_bins = 32

    def _project_root(self) -> Path:
        return Path.cwd()

    def _features_cache_path(self, n: int) -> Path:
        """
        Cache file path for extracted features (must include n=<n>).
        """
        feat_version = "v1"
        fname = f"grain_features_{feat_version}_n={n}.npz"
        return self._project_root() / fname

    def _extract_color_histogram(self, img):
        """
        Extract color histogram features from an image.

        Parameters
        ----------
        img: numpy array of shape (H, W, C)

        Returns
        -------
        hist: 1D numpy array of histogram features
        """
        histograms = []
        for c in range(img.shape[2]):
            channel = img[:, :, c].flatten()
            # Normalize to 0-1 range
            if channel.max() > 1:
                channel = channel / 255.0
            hist, _ = np.histogram(channel, bins=self.n_bins, range=(0, 1))
            hist = hist / (hist.sum() + 1e-8)  # Normalize histogram
            histograms.append(hist)
        return np.concatenate(histograms)

    def _extract_statistics(self, img):
        """
        Extract statistical features from an image.

        Parameters
        ----------
        img: numpy array of shape (H, W, C)

        Returns
        -------
        stats: 1D numpy array of statistical features
        """
        stats = []
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            stats.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
            ])
        return np.array(stats)
    
    def _extract_features_one(self, img: np.ndarray) -> np.ndarray:
        # Color histograms
        hist_features = self._extract_color_histogram(img)

        # Statistical features
        stat_features = self._extract_statistics(img)

        # Downsample 16x16
        h, w = img.shape[:2]
        new_h, new_w = 16, 16
        block_h, block_w = h // new_h, w // new_w

        if block_h > 0 and block_w > 0:
            downsampled = img[:block_h * new_h, :block_w * new_w].reshape(
                new_h, block_h, new_w, block_w, -1
            ).mean(axis=(1, 3))
        else:
            downsampled = img[:new_h, :new_w]

        flat_features = downsampled.reshape(-1)

        combined = np.concatenate([hist_features, stat_features, flat_features], axis=0)
        return combined.astype(np.float32, copy=False)

    def _extract_features(self, X):
        """
        Extract features from images (parallelized with joblib).
        Uses threads to avoid pickling and copying large numpy arrays.
        """
        if isinstance(X, list):
            X = np.array(X)

        n_samples = X.shape[0]

        # For small batches, parallel overhead is not worth it
        if n_samples < 256:
            feats = [self._extract_features_one(X[i]) for i in range(n_samples)]
            return np.stack(feats, axis=0).astype(np.float32)

        n_jobs = -1  # Use all available cores

        feats = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            batch_size=64
        )(
            delayed(self._extract_features_one)(X[i]) for i in range(n_samples)
        )

        return np.stack(feats, axis=0).astype(np.float32)

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

        # Extract features and labels
        X = train_data['X']
        y = train_data['y']

        # Extract features from images
        cache_path = self._features_cache_path(n=len(X))

        if cache_path.exists():
            print(f"[*] - Loading cached features: {cache_path}")
            data = np.load(cache_path, allow_pickle=False)
            if "X_features" not in data:
                raise ValueError(f"Cache file exists but missing 'X_features': {cache_path}")
            X_features = data["X_features"]
            # Sanity check
            if X_features.shape[0] != len(X):
                raise ValueError(
                    f"Cached features have wrong n: {X_features.shape[0]} != {len(X)} "
                    f"(cache={cache_path})"
                )
        else:
            print("[*] - Extracting features...")
            X_features = self._extract_features(X)
            print(f"[*] - Extracted {X_features.shape[1]} features per sample")

            # Save cache (compressed)
            print(f"[*] - Saving cached features: {cache_path}")
            np.savez_compressed(cache_path, X_features=X_features)

        # Scale features
        X_features = self.scaler.fit_transform(X_features)

        # Train the classifier
        print(f"[*] - Training on {X_features.shape[0]} samples with {X_features.shape[1]} features")
        self.clf.fit(X_features, y)
        print("[*] - Training complete")

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

        # Extract features
        X = test_data['X']

        # Extract features from images
        X_features = self._extract_features(X)

        # Scale features
        X_features = self.scaler.transform(X_features)

        # Predict
        y = self.clf.predict(X_features)

        print(f"[*] - Predicted {len(y)} samples")
        return y

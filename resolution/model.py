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
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self, max_iter=350, max_leaf_nodes=130, learning_rate=0.03, max_depth=20, min_samples_leaf=5, max_bins=255, l2_regularization=1.5e-06, early_stopping=False):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        max_iter: int, optional
            The maximum number of iterations for the classifier (default is 200).
        max_leaf_nodes: int, optional
            The maximum number of leaf nodes for the trees (default is 10).
        learning_rate: float, optional
            The learning rate for the classifier (default is 0.1).
        max_depth: int, optional
            The maximum depth of the trees (default is 30).
        min_samples_leaf: int, optional
            The minimum number of samples required to be at a leaf node (default is 2).
        max_bins: int, optional
            The maximum number of bins to use for discretizing features (default is 255).
        l2_regularization: float, optional
            The L2 regularization parameter (default is 0.0).
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
            max_bins=max_bins,         
            l2_regularization=l2_regularization, 
            early_stopping=early_stopping,  
            random_state=42
        )

        # Scaler for normalizing features
        self.scaler = StandardScaler()

        # PCA for dimensionality reduction (faster training, less overfitting)
        self.pca = PCA(n_components=100, random_state=42)
        self.use_pca = True

        # Number of histogram bins per channel
        self.n_bins = 32

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

    def _extract_features(self, X):
        """
        Extract features from images using multiple methods.

        Parameters
        ----------
        X: numpy array of shape (n_samples, H, W, C)

        Returns
        -------
        features: numpy array of shape (n_samples, n_features)
        """
        if isinstance(X, list):
            X = np.array(X)

        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            img = X[i]

            # 1. Color histograms (n_bins * 3 channels = 96 features)
            hist_features = self._extract_color_histogram(img)

            # 2. Statistical features (7 stats * 3 channels = 21 features)
            stat_features = self._extract_statistics(img)

            # 3. Downsampled image (reduce resolution for faster processing)
            # Resize to smaller size using simple averaging
            h, w = img.shape[:2]
            new_h, new_w = 16, 16  # Downsample to 16x16
            block_h, block_w = h // new_h, w // new_w

            if block_h > 0 and block_w > 0:
                downsampled = img[:block_h * new_h, :block_w * new_w].reshape(
                    new_h, block_h, new_w, block_w, -1
                ).mean(axis=(1, 3))
            else:
                downsampled = img[:new_h, :new_w]

            flat_features = downsampled.flatten()

            # Combine all features
            combined = np.concatenate([hist_features, stat_features, flat_features])
            features_list.append(combined)

        return np.array(features_list)

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
        print("[*] - Extracting features...")
        X_features = self._extract_features(X)
        print(f"[*] - Extracted {X_features.shape[1]} features per sample")

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

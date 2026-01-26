"""
BASELINE MODEL: HOG + XGBOOST PIPELINE
--------------------------------------
This model serves as our baseline for insect detection. Our choices are justified as follows:

1. HOG (Histogram of Oriented Gradients):
   - Insects are characterized by specific geometric shapes, edges, and textures (wings, legs, antennae).
   - HOG is invariant to local geometric and photometric transformations, making it superior to raw pixels.
   - Multiscale approach captures both global body shape and fine-grained structural details.

2. OVERSAMPLING (RandomOverSampler):
   - The dataset is extremely imbalanced (~3% visitors). Without oversampling, the model would likely 
     converge to a trivial solution (predicting '0' for everything).
   - Oversampling ensures the gradient updates in XGBoost see enough positive samples to learn 
     discriminant features.

3. XGBOOST (Extreme Gradient Boosting):
   - It handles high-dimensional HOG vectors efficiently and provides built-in regularization 
     to prevent overfitting on small datasets.
   - The 'scale_pos_weight' and 'max_delta_step' parameters offer surgical control over 
     class imbalance and training stability.
"""

import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.feature import hog
from skimage.color import rgb2gray
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

class Model:
    """
    Advanced model using HOG Features and XGBoost.
    Optimized for insect detection with high class imbalance.
    """

    def __init__(self, grayscale=False, multiscale=True):
        print("[*] - Initializing HOG + XGBoost Classifier (Baseline)")
         # Parameters for parallelization
        try:
            self.n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else multiprocessing.cpu_count()
        except:
            self.n_cpus = 1
        print(f"[*] Detected {self.n_cpus} CPU cores for parallel processing.")

        # HOG settings
        self.grayscale = grayscale
        self.multiscale = multiscale
        
        # HOG optimal parameters
        self.best_h_orient = 9
        self.best_h_pix = (8, 8)
        self.best_h_block = (1, 1)
        
        # Multi-scale HOG optimal parameters (S1 global, S2 local)
        self.s1_config = (12, (16, 16))
        self.s2_config = (9, (8, 8))

        # Classifier initialization with tuned hyperparameters
        self.clf = XGBClassifier(
            n_estimators=1500,      
            max_depth=7,            
            learning_rate=0.005,    
            subsample=0.8,         
            colsample_bytree=0.8,  
            scale_pos_weight=3,
            max_delta_step=0,
            min_child_weight=15,
            gamma=0.5,  
            reg_alpha=0,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42,
            n_jobs=self.n_cpus,
            enable_categorical=False
        )

    # --- Helpers for HOG extraction ---
    def _process_single(self, img):
        """Classic HOG extraction."""
        img_proc = rgb2gray(img) if self.grayscale else img
        c_axis = None if self.grayscale else -1
        return hog(img_proc, orientations=self.best_h_orient, 
                   pixels_per_cell=self.best_h_pix, cells_per_block=self.best_h_block, 
                   transform_sqrt=False, channel_axis=c_axis)

    def _process_multi(self, img):
        """Multi-scale HOG extraction."""
        img_proc = rgb2gray(img) if self.grayscale else img
        c_axis = None if self.grayscale else -1
        h1 = hog(img_proc, orientations=self.s1_config[0], pixels_per_cell=self.s1_config[1],
                 cells_per_block=(1, 1), transform_sqrt=False, channel_axis=c_axis)
        h2 = hog(img_proc, orientations=self.s2_config[0], pixels_per_cell=self.s2_config[1],
                 cells_per_block=(1, 1), transform_sqrt=False, channel_axis=c_axis)
        return np.concatenate([h1, h2])

    def _preprocess(self, X):
        """
        Extract HOG features instead of simple flattening.
        Uses Parallel processing to handle large datasets efficiently.
        """
        print(f"[*] - Preprocessing: Extracting HOG features (Multiscale={self.multiscale})...")
        
        if self.multiscale:
            features = Parallel(n_jobs=self.n_cpus)(
                delayed(self._process_multi)(img) for img in tqdm(X, desc="HOG extraction", leave=False)
            )
        else:
            features = Parallel(n_jobs=self.n_cpus)(
                delayed(self._process_single)(img) for img in tqdm(X, desc="HOG extraction", leave=False)
            )
            
        return np.array(features).astype('float32')

    def fit(self, X, y, oversampling=True, ratio=0.1):
        """
        Train the XGBoost model on HOG features with optional oversampling.
        """
        print(f"[*] - Training on {X.shape[0]} samples...")
        
        # Feature extraction
        X_features = self._preprocess(X)
        
        # Oversampling to handle imbalance
        if oversampling:
            print(f"[*] - Applying Random Oversampling (Target ratio: {ratio})")
            # Calculate required strategy based on target ratio
            # strategy = ratio / (1 - ratio)
            current_ratio = np.sum(y == 1) / np.sum(y == 0)
            target_strategy = ratio / (1 - ratio)
            
            if target_strategy > current_ratio:
                ros = RandomOverSampler(sampling_strategy=target_strategy, random_state=42)
                X_resampled, y_resampled = ros.fit_resample(X_features, y)
                print(f"[✔] - Oversampling complete: {X_resampled.shape[0]} samples total.")
            else:
                print("[!] - Target ratio is lower than current. Skipping oversampling.")
                X_resampled, y_resampled = X_features, y
        else:
            X_resampled, y_resampled = X_features, y
        
        # Training
        self.clf.fit(X_resampled, y_resampled)
        print("[✔] - Training complete.")

    def predict(self, X, threshold=0.5):
        """
        Predict labels (0 or 1) using a probability threshold.
        Lower threshold increases sensitivity (more TP, but potentially more FP).
        """
        print(f"[*] - Predicting on {X.shape[0]} samples with threshold {threshold}...")
        
        # Feature extraction
        X_features = self._preprocess(X)
        
        # Obtain probabilities for the positive class
        probs = self.clf.predict_proba(X_features)[:, 1]
        
        # Apply custom threshold
        return (probs >= threshold).astype(int)
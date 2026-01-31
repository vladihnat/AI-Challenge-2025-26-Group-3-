"""
The model.py file should define a class named 'Model' with the following interface::
- __init__(self)
- fit(self, X, y)
- predict(self, X)

When you submit your code to codabench, this 'Model' class will be instantiated, trained, and evaluated.
You can have all other helper functions and classes in this file as needed, 
but only these three methods will be called by the evaluation framework.

BASELINE MODEL: HOG + XGBOOST PIPELINE
--------------------------------------
This model serves as our baseline for insect detection. Our choices are justified as follows:

1. HOG (Histogram of Oriented Gradients):
   - Insects are characterized by specific geometric shapes, edges, and textures (wings, legs, antennae).
   - HOG is invariant to local geometric and photometric transformations, making it superior to raw pixels.
   - It's a lightweight feature extractor, allowing efficient training and inference.
   - We experimented with both grayscale and color HOG, as well as multi-scale HOG to capture
     features at different resolutions. The final choice was based on cross-validation performance.

2. OVERSAMPLING (RandomOverSampler):
   - The dataset is extremely imbalanced (~3% visitors). Without oversampling, the model would likely 
     converge to a trivial solution (predicting '0' for everything).
   - Oversampling ensures the gradient updates in XGBoost see enough positive samples to learn 
     discriminant features.
   - SMOTE was considered but discarded due to the bad separability of the feature space, which could
     lead to synthetic samples that do not represent real insects well.

3. XGBOOST (Extreme Gradient Boosting):
   - It handles high-dimensional HOG vectors efficiently and provides built-in regularization 
     to prevent overfitting on small datasets.
   - The 'scale_pos_weight' and 'max_delta_step' parameters offer surgical control over 
     class imbalance and training stability. We played with them to find the best trade-off between
     true positives and false positives (see our Starting_Kit for further explanations).
   - max_depth and min_child_weight also work as a great pair to control overfitting and
     maximize F1-score.
   - Random Forest was considered but discarded due to it not being able to predict any positive samples
     in our initial experiments, likely due to the extreme class imbalance.

4. THRESHOLD TUNING:
    - Instead of using the default 0.5 threshold, we tune it via cross-validation to find 
      the optimal balance between precision and recall, maximizing the F1-score. This is 
      crucial in imbalanced settings where the default threshold may not be optimal. The model
      may output probabilities that are skewed towards the majority class, so adjusting the threshold
      helps capture more true positives without excessively increasing false positives.

5. PARALLELIZATION:
   - HOG feature extraction and XGBoost training can be computationally intensive, especially 
     for large datasets. We leverage all available CPU cores to speed up these processes,
     ensuring efficient use of resources and reduced training time.
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

    def __init__(self, grayscale=False, multiscale=False, oversampling=True, ratio=0.1, threshold=0.5):
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
        self.best_h_pix = (16, 16)
        self.best_h_block = (1, 1)
        self.best_h_transform_sqrt = False
        
        # Multi-scale HOG optimal parameters (S1 global, S2 local)
        self.s1_config = (12, (16, 16), (1, 1), False)
        self.s2_config = (9, (8, 8), (1, 1), False)

        # Classifier initialization with tuned hyperparameters
        self.clf = XGBClassifier(
            n_estimators=1500,      
            max_depth=15,            
            learning_rate=0.005,    
            subsample=0.8,         
            colsample_bytree=0.6,  
            scale_pos_weight=10,
            max_delta_step=0,
            min_child_weight=15,
            gamma=0.5,  
            reg_alpha=1.0,
            reg_lambda=0.1,
            tree_method='hist',
            random_state=42,
            n_jobs=self.n_cpus,
            enable_categorical=False
        )

        # Threshold for classification
        self.threshold = threshold

        # Oversampling strategy
        self.oversampling = oversampling
        self.ratio = ratio

    # --- Helpers for HOG extraction ---
    def _process_single(self, img):
        """Classic HOG extraction."""
        img_proc = rgb2gray(img) if self.grayscale else img
        c_axis = None if self.grayscale else -1
        return hog(img_proc, orientations=self.best_h_orient, 
                   pixels_per_cell=self.best_h_pix, cells_per_block=self.best_h_block, 
                   transform_sqrt=self.best_h_transform_sqrt, channel_axis=c_axis)

    def _process_multi(self, img):
        """Multi-scale HOG extraction."""
        img_proc = rgb2gray(img) if self.grayscale else img
        c_axis = None if self.grayscale else -1
        h1 = hog(img_proc, orientations=self.s1_config[0], pixels_per_cell=self.s1_config[1],
                 cells_per_block=self.s1_config[2], transform_sqrt=self.s1_config[3], channel_axis=c_axis)
        h2 = hog(img_proc, orientations=self.s2_config[0], pixels_per_cell=self.s2_config[1],
                 cells_per_block=self.s2_config[2], transform_sqrt=self.s2_config[3], channel_axis=c_axis)
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

    def fit(self, X, y):
        """
        Train the XGBoost model on HOG features with optional oversampling.
        """
        print(f"[*] - Training on {X.shape[0]} samples...")
        
        # Feature extraction
        X_features = self._preprocess(X)
        
        # Oversampling to handle imbalance
        if self.oversampling:
            print(f"[*] - Applying Random Oversampling (Target ratio: {self.ratio})")
            # Calculate required strategy based on target ratio
            # strategy = ratio / (1 - ratio)
            current_ratio = np.sum(y == 1) / np.sum(y == 0)
            target_strategy = self.ratio / (1 - self.ratio)
            
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

    def predict(self, X):
        """
        Predict labels (0 or 1) using a probability threshold.
        Lower threshold increases sensitivity (more TP, but potentially more FP).
        """
        print(f"[*] - Predicting on {X.shape[0]} samples with threshold {self.threshold}...")
        
        # Feature extraction
        X_features = self._preprocess(X)
        
        # Obtain probabilities for the positive class
        probs = self.clf.predict_proba(X_features)[:, 1]
        
        # Apply custom threshold
        return (probs >= self.threshold).astype(int)
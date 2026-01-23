import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    Base model using Random Forest.
    Adapted to handle strong class imbalance (97% vs 3%).
    """

    def __init__(self):
        print("[*] - Initializing Random Forest Classifier")
        
        # We chose Random Forest for its robustness and ability to handle 
        # high-dimensional raw pixel data without requiring complex preprocessing
        
        # Use class_weight="balanced" to give more importance to the
        # minority class (the visits) automatically (crucial given the imbalance here)*

        self.clf = RandomForestClassifier(
            n_estimators=50,      # Number of trees (50 for efficiency)
            max_depth=10,         # Limit depth to avoid overfitting
            class_weight="balanced", # CRUCIAL for a highly imbalanced dataset
            n_jobs=-1,            # Uses all cores of your processor
            random_state=42
        )

    def _preprocess(self, X):
        """
        Prepare data for scikit-learn.
        Flatten images and reduce memory load if needed.
        """
        # X is initially (n_samples, height, width, 3)
        # We have to transform it to (n_samples, height * width * 3)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Optional : conversion to float32 for scikit-learn
        return X.astype('float32')

    def fit(self, X, y):
        """
        Train the model.
        """
        print(f"[*] - Training on {X.shape[0]} samples...")
        
        X_flat = self._preprocess(X)
        
        # Training
        self.clf.fit(X_flat, y)
        print("[✔] - Training complete.")

    def predict(self, X):
        """
        Predict labels (0 or 1).
        """
        print(f"[*] - Predicting on {X.shape[0]} samples...")
        
        X_flat = self._preprocess(X)
        return self.clf.predict(X_flat)
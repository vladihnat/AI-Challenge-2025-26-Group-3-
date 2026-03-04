import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

class Model:
    def __init__(self):
        print("[*] - Initializing Random Forest Classifier")
        self.scaler    = RobustScaler()
        self.clf       = RandomForestClassifier(
            n_estimators  = 324,
            max_depth     = 22,
            min_samples_split = 10,
            min_samples_leaf  = 2,
            max_features  = 0.9653946665869293,
            max_samples   = 0.5892208809109154,
            min_impurity_decrease = 9.94740876259007e-07,
            class_weight  = 'balanced',
            random_state  = 42,
            n_jobs        = -1,
        )
        self.classes_  = None
        self.is_fitted = False

    def fit(self, train_data):
        print("[*] - Training...")
        X = np.array(train_data["X_train"])
        y = np.array(train_data["y_train"])
        print(f"[*] - Shape: {X.shape}, Classes: {np.unique(y)}")

        self.classes_ = np.unique(y)
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

        self.is_fitted = True
        print("[*] - Done.")

    def predict(self, X_test):
        print("[*] - Predicting...")
        X = np.array(X_test["X_test"])
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
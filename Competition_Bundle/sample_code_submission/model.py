import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    Modèle de base utilisant une Random Forest.
    Adapté pour gérer le fort déséquilibre des classes (97% vs 3%).
    """

    def __init__(self):
        print("[*] - Initializing Random Forest Classifier")
        
        # On utilise class_weight="balanced" pour donner plus d'importance 
        # à la classe minoritaire (les visites) automatiquement.
        self.clf = RandomForestClassifier(
            n_estimators=50,      # Nombre d'arbres (50 pour rester rapide)
            max_depth=10,         # On limite la profondeur pour éviter l'overfitting
            class_weight="balanced", # CRUCIAL pour ton dataset déséquilibré
            n_jobs=-1,            # Utilise tous les cœurs de ton processeur
            random_state=42
        )

    def _preprocess(self, X):
        """
        Prépare les données pour scikit-learn.
        Aplatit les images et réduit la charge mémoire si nécessaire.
        """
        # X est initialement (n_samples, height, width, 3)
        # On doit le transformer en (n_samples, height * width * 3)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Optionnel : conversion en float32 pour scikit-learn
        return X.astype('float32')

    def fit(self, X, y):
        """
        Entraîne le modèle.
        """
        print(f"[*] - Training on {X.shape[0]} samples...")
        
        X_flat = self._preprocess(X)
        
        # Entraînement
        self.clf.fit(X_flat, y)
        print("[✔] - Training complete.")

    def predict(self, X):
        """
        Prédit les labels (0 ou 1).
        """
        print(f"[*] - Predicting on {X.shape[0]} samples...")
        
        X_flat = self._preprocess(X)
        return self.clf.predict(X_flat)
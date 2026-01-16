# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:
    """
    Simple scikit-learn style classification model compatible with HDF5 arrays.
    """

    def __init__(self):
        """
        Initialize the classifier.
        """
        print("[*] - Initializing Classifier")

        # Simple baseline classifier
        self.clf = LogisticRegression(
            max_iter=100, # Réduit pour tester rapidement
            solver="lbfgs"
        )

    def fit(self, X, y):
        """
        Train the model using training data.

        Parameters
        ----------
        X : numpy.ndarray
            Images ou caractéristiques (ex: [n_samples, height, width, channels])
        y : numpy.ndarray
            Labels (ex: [n_samples])
        """
        print(f"[*] - Training Classifier on {X.shape[0]} samples")

        # Redimensionnement : Si X est un bloc d'images (ex: 100x64x64), 
        # on l'aplatit en (100, 4096) pour que la LogisticRegression comprenne.
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        self.clf.fit(X, y)

    def predict(self, X):
        """
        Predict labels for test data.

        Parameters
        ----------
        X : numpy.ndarray
            Images ou caractéristiques de test.

        Returns
        -------
        y : numpy.ndarray
            Predicted labels
        """
        print(f"[*] - Predicting on {X.shape[0]} test samples")

        # Même transformation pour le test set
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        return self.clf.predict(X)
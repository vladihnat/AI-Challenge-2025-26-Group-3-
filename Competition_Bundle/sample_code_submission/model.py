# Model file which contains a model class in scikit-learn style
# The Model class must implement:
# - __init__
# - fit
# - predict
#
# Created by: Ihsan Ullah
# Created on: 13 Jan, 2026

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
    Simple scikit-learn style classification model.
    """

    def __init__(self):
        """
        Initialize the classifier.
        """
        print("[*] - Initializing Classifier")

        # Simple baseline classifier
        self.clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )

    def fit(self, train_data):
        """
        Train the model using training data.

        Parameters
        ----------
        train_data : dict
            Dictionary containing:
                - "X": feature matrix
                - "y": target labels
        """
        print("[*] - Training Classifier on the train set")

        X = np.array(train_data["X"])
        y = np.array(train_data["y"])

        self.clf.fit(X, y)

    def predict(self, test_data):
        """
        Predict labels for test data.

        Parameters
        ----------
        test_data : dict
            Dictionary containing:
                - "X": feature matrix

        Returns
        -------
        y : numpy.ndarray
            Predicted labels
        """
        print("[*] - Predicting test set using trained Classifier")

        X = np.array(test_data["X"])
        y = self.clf.predict(X)

        return y
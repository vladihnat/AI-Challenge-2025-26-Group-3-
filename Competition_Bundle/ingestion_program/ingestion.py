# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import h5py
import numpy as np
from datetime import datetime as dt


class Ingestion:
    """
    Class handling the ingestion process for HDF5 data.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.model = None
        
        # On sépare data et labels pour coller à la structure H5
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        
        self.predictions = None
        self.ingestion_result = None

    # ... (start_timer, stop_timer, get_duration, save_duration restent identiques) ...

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def save_duration(self, output_dir=None):
        duration = self.get_duration()
        if duration is None: return
        duration_in_mins = duration.total_seconds() / 60
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        with open(duration_file, "w") as f:
            json.dump({"ingestion_duration": duration_in_mins}, f, indent=4)

    # ------------------------------------------
    # Data loading (Version H5)
    # ------------------------------------------

    def load_train_and_test_data(self, input_dir):
        """
        Load training and testing data from H5 files.
        """
        print(f"[*] Loading data from {input_dir}")

        # Définition des chemins selon tes fichiers créés
        train_data_path = os.path.join(input_dir, "train_data.h5")
        train_labels_path = os.path.join(input_dir, "train_labels.h5")
        test_data_path = os.path.join(input_dir, "test_data.h5")

        # Chargement Train Data & Labels
        if os.path.exists(train_data_path) and os.path.exists(train_labels_path):
            with h5py.File(train_data_path, "r") as f:
                self.train_data = f["images"][:]
            with h5py.File(train_labels_path, "r") as f:
                self.train_labels = f["labels"][:]
            print(f"[+] Loaded Train: {self.train_data.shape}")
        else:
            print("[!] Train files missing, running in test-only mode")

        # Chargement Test Data (Obligatoire)
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"test_data.h5 not found in {input_dir}")

        with h5py.File(test_data_path, "r") as f:
            self.test_data = f["images"][:]
        print(f"[+] Loaded Test: {self.test_data.shape}")

    # ------------------------------------------
    # Model handling
    # ------------------------------------------

    def init_submission(self, Model):
        print("[*] Initializing submitted model")
        self.model = Model()

    def fit_submission(self):
        """
        Fit le modèle avec data ET labels.
        """
        if self.train_data is None or self.train_labels is None:
            print("[!] Missing train data or labels, skipping training")
            return

        print("[*] Fitting submitted model")
        # On passe X et Y au fit
        self.model.fit(self.train_data, self.train_labels)

    def predict_submission(self):
        print("[*] Running prediction")
        self.predictions = self.model.predict(self.test_data)

    # ------------------------------------------
    # Result handling
    # ------------------------------------------

    def compute_result(self):
        print("[*] Computing ingestion result")
        # On transforme les prédictions en liste pour le JSON si c'est du Numpy
        preds_to_save = self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions
        
        self.ingestion_result = {
            "num_test_samples": len(self.test_data) if self.test_data is not None else 0,
            "predictions": preds_to_save,
            "status": "success"
        }

    def save_result(self, output_dir=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(self.ingestion_result, f, indent=4)
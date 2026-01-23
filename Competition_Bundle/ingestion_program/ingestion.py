import os
import json
import h5py
import numpy as np
from datetime import datetime as dt

class Ingestion:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.predictions = None
        self.ingestion_result = None

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def save_duration(self, output_dir):
        duration = self.get_duration()
        if duration is None: return
        duration_in_mins = duration.total_seconds() / 60
        with open(os.path.join(output_dir, "ingestion_duration.json"), "w") as f:
            json.dump({"ingestion_duration": duration_in_mins}, f, indent=4)

    # ------------------------------------------
    # Data loading
    # ------------------------------------------
    def load_train_and_test_data(self, input_dir):
        """
        Charge images from H5 and labels from NPY.
        """
        print(f"[*] Loading data from {input_dir}")

        # File paths
        train_data_path = os.path.join(input_dir, "train_data.h5")
        train_labels_path = os.path.join(input_dir, "train_labels.npy")
        test_data_path = os.path.join(input_dir, "test_data.h5")

        # Train loading
        if os.path.exists(train_data_path) and os.path.exists(train_labels_path):
            with h5py.File(train_data_path, "r") as f:
                # We use [:] because the RAM allows it (30k images 224x224x3 uint8 ~ 4.5 Go)
                self.train_data = f["images"][:]
            self.train_labels = np.load(train_labels_path)
            print(f"[+] Loaded Train: {self.train_data.shape} samples")
        else:
            print("[!] Train data/labels missing. Ingestion will continue for test only.")

        # Loading Test
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"test_data.h5 not found in {input_dir}")
        
        with h5py.File(test_data_path, "r") as f:
            self.test_data = f["images"][:]
        print(f"[+] Loaded Test: {self.test_data.shape} samples")

    # ------------------------------------------
    # Model handling
    # ------------------------------------------
    def init_submission(self, Model):
        print("[*] Initializing submitted model")
        self.model = Model()

    def fit_submission(self):
        if self.train_data is not None and self.train_labels is not None:
            print("[*] Fitting submitted model...")
            self.model.fit(self.train_data, self.train_labels)
        else:
            print("[!] Skipping fit: no training data provided.")

    def predict_submission(self):
        print("[*] Running prediction on test data...")
        self.predictions = self.model.predict(self.test_data)

    # ------------------------------------------
    # Result handling
    # ------------------------------------------
    def compute_result(self):
        # Codabench formatting (either .txt or .json depending the scorer)
        preds = self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions
        self.ingestion_result = preds

    def save_result(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save predictions (usually named 'labels.predict')
        result_file = os.path.join(output_dir, "labels.predict")
        with open(result_file, "w") as f:
            for p in self.ingestion_result:
                f.write(f"{p}\n")
        print(f"[✔] Predictions saved to {result_file}")
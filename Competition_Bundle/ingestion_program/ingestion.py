import os
import sys
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

    def load_train_and_test_data(self, input_dir):
        """Charge images from H5 and labels from NPY."""
        print(f"[*] Loading data from {input_dir}")

        train_data_path = os.path.join(input_dir, "train_data.h5")
        train_labels_path = os.path.join(input_dir, "train_labels.npy")
        test_data_path = os.path.join(input_dir, "test_data.h5")

        if os.path.exists(train_data_path) and os.path.exists(train_labels_path):
            with h5py.File(train_data_path, "r") as f:
                self.train_data = f["images"][:]
            self.train_labels = np.load(train_labels_path)
            print(f"[+] Loaded Train: {self.train_data.shape} samples")
        
        if os.path.exists(test_data_path):
            with h5py.File(test_data_path, "r") as f:
                self.test_data = f["images"][:]
            print(f"[+] Loaded Test: {self.test_data.shape} samples")
        else:
            raise FileNotFoundError(f"test_data.h5 not found in {input_dir}")

    def init_submission(self, model_dir):
        """
        Import the Model class from the participant's submission.
        """
        print(f"[*] Searching for model.py in {model_dir}")
        sys.path.append(model_dir)
        try:
            from model import Model
            self.model = Model()
            print("[✔] Model initialized successfully.")
        except ImportError as e:
            print(f"[-] Could not import Model from submission: {e}")
            raise

    def fit_submission(self):
        if self.train_data is not None and self.train_labels is not None:
            print(f"[*] Fitting model on {len(self.train_data)} samples...")
            self.model.fit(self.train_data, self.train_labels)
        else:
            print("[!] Skipping fit: training data missing.")

    def predict_submission(self):
        print(f"[*] Running prediction on {len(self.test_data)} test samples...")
        self.predictions = self.model.predict(self.test_data)

    def compute_result(self):
        if isinstance(self.predictions, np.ndarray):
            self.ingestion_result = self.predictions.tolist()
        else:
            self.ingestion_result = self.predictions

    def save_result(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result_file = os.path.join(output_dir, "labels.predict")
        with open(result_file, "w") as f:
            for p in self.ingestion_result:
                f.write(f"{int(p)}\n")
        print(f"[✔] Predictions saved to {result_file}")
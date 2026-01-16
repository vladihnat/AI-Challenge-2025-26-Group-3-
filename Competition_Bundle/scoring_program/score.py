# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import h5py
import numpy as np
from datetime import datetime as dt


class Scoring:
    """
    Class for computing scores by comparing HDF5 reference labels 
    with JSON ingestion predictions.
    """

    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.reference_data = None
        self.ingestion_result = None
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    # ------------------------------------------
    # Data Loading
    # ------------------------------------------

    def load_reference_data(self, reference_dir):
        """
        Load the ground truth labels from the HDF5 file.
        """
        print(f"[*] Reading reference data from {reference_dir}")
        ref_file = os.path.join(reference_dir, "test_labels.h5")
        
        try:
            with h5py.File(ref_file, "r") as f:
                # On récupère le dataset 'labels'
                self.reference_data = f["labels"][:]
            print(f"[+] Loaded {len(self.reference_data)} reference labels.")
        except Exception as e:
            print(f"[-] Error loading reference data: {e}")
            self.reference_data = None

    def load_ingestion_result(self, predictions_dir):
        """
        Load the ingestion result (predictions) from the JSON file.
        """
        print(f"[*] Reading ingestion result from {predictions_dir}")
        result_file = os.path.join(predictions_dir, "result.json")

        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                # On récupère la liste des prédictions stockée par l'ingestion
                self.ingestion_result = np.array(data.get("predictions", []))
            print(f"[+] Loaded {len(self.ingestion_result)} predictions.")
        except Exception as e:
            print(f"[-] Error loading ingestion result: {e}")
            self.ingestion_result = None

    # ------------------------------------------
    # Scoring Logic
    # ------------------------------------------

    def compute_scores(self):
        """
        Compute the Accuracy score.
        """
        print("[*] Computing scores")

        if self.reference_data is None or self.ingestion_result is None:
            print("[-] Missing data to compute scores.")
            return

        if len(self.reference_data) != len(self.ingestion_result):
            print(f"[-] Dimension mismatch: Ref({len(self.reference_data)}) vs Pred({len(self.ingestion_result)})")
            # Optionnel: ajuster ou tronquer selon le besoin
        
        # Calcul de l'accuracy
        correct = np.sum(self.reference_data == self.ingestion_result)
        accuracy = float(correct) / len(self.reference_data) if len(self.reference_data) > 0 else 0

        self.scores_dict = {
            "accuracy": accuracy,
            "error_rate": 1.0 - accuracy,
            "n_samples": len(self.reference_data)
        }
        
        print(f"[✔] Accuracy: {accuracy:.4f}")

    def write_scores(self, output_dir):
        """
        Save the scores to scores.json.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"[*] Writing scores to {output_dir}")
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f_score:
            json.dump(self.scores_dict, f_score, indent=4)
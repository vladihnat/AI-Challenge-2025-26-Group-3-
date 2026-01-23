import os
import json
import numpy as np
from datetime import datetime as dt
from sklearn.metrics import balanced_accuracy_score, f1_score

class Scoring:
    def __init__(self):
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

    def load_reference_data(self, reference_dir):
        """Charge les labels réels (0 ou 1)."""
        print(f"[*] Reading reference data from {reference_dir}")
        ref_file = os.path.join(reference_dir, "test_labels.npy")
        try:
            self.reference_data = np.load(ref_file)
            print(f"[+] Loaded {len(self.reference_data)} reference labels.")
        except Exception as e:
            print(f"[-] Error loading reference data: {e}")

    def load_ingestion_result(self, predictions_dir):
        """Charge les prédictions du modèle."""
        print(f"[*] Reading ingestion result from {predictions_dir}")
        result_file = os.path.join(predictions_dir, "labels.predict")
        try:
            with open(result_file, "r") as f:
                preds = [float(line.strip()) for line in f if line.strip()]
            self.ingestion_result = np.array(preds)
            print(f"[+] Loaded {len(self.ingestion_result)} predictions.")
        except Exception as e:
            print(f"[-] Error loading ingestion result: {e}")

    def compute_scores(self):
        """Calcule les métriques clés pour données déséquilibrées."""
        print("[*] Computing scores")

        if self.reference_data is None or self.ingestion_result is None:
            print("[-] Missing data to compute scores.")
            return

        # Alignement des dimensions au cas où
        min_len = min(len(self.reference_data), len(self.ingestion_result))
        y_true = self.reference_data[:min_len]
        y_pred = self.ingestion_result[:min_len]

        # 1. Balanced Accuracy : moyenne de l'accuracy de chaque classe (Base 0.5)
        bac = balanced_accuracy_score(y_true, y_pred)
        
        # 2. F1-Score : moyenne harmonique Précision/Rappel (Idéal pour les classes rares)
        f1 = f1_score(y_true, y_pred)

        self.scores_dict = {
            "balanced_accuracy": float(bac),
            "f1_score": float(f1),
        }
        
        print(f"--- RESULTS ---")
        print(f"Balanced Accuracy: {bac:.4f}")
        print(f"F1-Score:          {f1:.4f}")

    def write_scores(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f:
            json.dump(self.scores_dict, f, indent=4)
        print(f"[*] Scores saved to {score_file}")
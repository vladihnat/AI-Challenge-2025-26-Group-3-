# ------------------------------------------
# Imports
# ------------------------------------------
import sys
import argparse
import os
import subprocess
import importlib

# ------------------------------------------
# Directories
# ------------------------------------------
module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(module_dir)

# ------------------------------------------
# Args
# ------------------------------------------
parser = argparse.ArgumentParser(
    description="This is script to run ingestion program for the competition"
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)

# ------------------------------------------
# Ensure required packages are installed
# ------------------------------------------

def check_and_install_dependencies(submission_dir):
    """
    Installs missing dependencies from requirements.txt only if not already present.
    """
    req_path = os.path.join(submission_dir, "requirements.txt")
    
    if not os.path.exists(req_path):
        print("[*] No requirements.txt found. Using default environment.")
        return

    print("[*] Checking requirements.txt for missing libraries...")
    with open(req_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    to_install = []
    for req in requirements:
        # Extraire le nom de la librairie (ex: 'xgboost==1.7.0' -> 'xgboost')
        package_name = req.split('==')[0].split('>=')[0].split('>')[0].strip().replace('-', '_')
        
        try:
            importlib.import_module(package_name)
            # print(f"[✔] {package_name} is already installed.")
        except ImportError:
            to_install.append(req)

    if to_install:
        print(f"[*] Installing missing dependencies: {', '.join(to_install)}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir", *to_install
            ])
            print("[✔] Installation complete.")
        except Exception as e:
            print(f"[!] Pip installation failed: {e}")
    else:
        print("[✔] All requirements are already met. Skipping installation.")

# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":

    print("\n----------------------------------------------")
    print("Ingestion Program started!")
    print("----------------------------------------------\n\n")

    from ingestion import Ingestion

    args = parser.parse_args()

    if not args.codabench:
        # DO NOT CHANGE THESE PATHS UNLESS YOU CHANGE THE FOLDER NAMES IN THE BUNDLE
        input_dir = os.path.join(root_dir_name, "input_data")
        output_dir = os.path.join(root_dir_name, "sample_result_submission")
        program_dir = os.path.join(root_dir_name, "ingestion_program")
        submission_dir = os.path.join(root_dir_name, "sample_code_submission")
    else:
        # DO NOT CHANGE THESE PATHS. THESE ARE USED ON THE CODABENCH PLATFORM
        input_dir = "/app/input_data"
        output_dir = "/app/output"
        program_dir = "/app/program"
        submission_dir = "/app/ingested_program"

    sys.path.append(input_dir)
    sys.path.append(output_dir)
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    # Check and install dependencies from submission dir
    check_and_install_dependencies(submission_dir)

    # Import model from submission dir
    from model import Model

    # Initialize Ingestions
    ingestion = Ingestion()

    # Start timer
    ingestion.start_timer()

    # Load train and test data
    ingestion.load_train_and_test_data(input_dir)

    # initialize submission
    ingestion.init_submission(Model)

    # fit submission
    ingestion.fit_submission()

    # predict submission
    ingestion.predict_submission()

    # compute result
    ingestion.compute_result()

    # save result
    ingestion.save_result(output_dir)

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    print("\n------------------------------------")
    print(f"[✔] Total duration: {ingestion.get_duration()}")
    print("------------------------------------")

    # Save Duration
    ingestion.save_duration(output_dir)

    print("\n----------------------------------------------")
    print("[✔] Ingestion Program executed successfully!")
    print("----------------------------------------------\n\n")

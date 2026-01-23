# AI-Challenge-2025-26-Group-3-Pollinator

Master 1 AI challenge creation and resolution - Pollinator Detection Project

## 📋 Project Overview

This repository contains a complete machine learning challenge focused on **Pollinator Detection** using image sequence data. The project is designed to be deployed on Codabench and includes all necessary components for participants to understand, develop, and submit their solutions.

---

## 🗂️ Repository Structure

```
AI-Challenge-2025-26-Group-3-/
│
├── README.md                    # This file - project overview and structure
├── split_data.py                # Utility script for cleaning and splitting original dataset into train/test sets
│
├── Starting_Kit/                # Participant starting resources
│   ├── README.md               # Guide for getting started with the challenge
│   └── README.ipynb            # Interactive Jupyter notebook with data exploration and baseline models
│
└── Competition_Bundle/          # Complete Codabench competition package
    ├── competition.yaml        # Competition configuration for Codabench
    ├── logo.png               # Competition logo
    │
    ├── ingestion_program/     # Loads data and runs participant models
    │   ├── README.md          # Instructions for ingestion program
    │   └── run_ingestion.py   # Main ingestion script
    │   └── ingestion.py       # Ingestion script used by the main program
    │   └── metadata.yml       # DO NOT MODIFY
    │
    ├── scoring_program/       # Evaluates model predictions
    │   └── README.md          # Instructions for scoring program
    │   └── run_scoring.py     # Main scoring script
    │   └── score.py           # Scoring script used by the main program
    │   └── metadata.yml       # DO NOT MODIFY
    │
    ├── input_data/            # Training and test data
    │   └── README.md          # Instructions for input_data
    │   └── train_data.h5      # .h5 file containing train image sequences
    │   └── test_data.h5       # .h5 file containing test image sequences
    │   └── train_labels.npy   # .npy file containing train labels
    │
    ├── reference_data/        # Ground truth labels for evaluation
    │   └── README.md          # Instructions for reference_data
    │   └── test_labels.npy    # .npy file containing test labels
    │
    ├── sample_code_submission/ # Example participant submission
    │   └── model.py            # Sample model implementation
    │
    ├── sample_result_submission/ # Example prediction outputs
    │   └── README.md             # Instructions
    │
    ├── pages/                 # Competition description pages (HTML/Markdown)
    │   └── overview.md        # Overview page
    │   └── terms.md           # Terms page
    │
    └── utilities/             # Helper scripts and tools
        └── compile_bundle.py  # Script to compile the competition bundle
```

---

## 🎯 Challenge Description

The challenge focuses on **binary classification** of image sequences to detect the presence of pollinators. Each data sample consists of a sequence of frames (similar to stop-motion recording) stored in `.h5` format, with a single binary label indicating whether a pollinator is present.

### Key Characteristics:
- **Data Format**: HDF5 (`.h5`) files containing image sequences
- **Task**: Binary classification (pollinator present/absent)
- **Challenge**: Highly imbalanced dataset
- **Evaluation**: Metrics appropriate for imbalanced classification

---

## 🚀 Getting Started

### For Participants

1. **Start with the Starting Kit**:
   - Navigate to `Starting_Kit/`
   - Open `README.ipynb` in Jupyter Notebook/Lab
   - The notebook will automatically download the dataset
   - Explore the data visualizations and baseline models

2. **Develop Your Model**:
   - Use the examples in `sample_code_submission/` as a template
   - Implement your model following the required format
   - Test locally using the ingestion program

3. **Test Locally**:
   ```bash
   cd Competition_Bundle
   python3 ingestion_program/run_ingestion.py
   ```

4. **Submit to Codabench**:
   - Package your code according to submission guidelines
   - Upload to the competition platform

---

## 📊 Data Structure

- **Format**: HDF5 (`.h5`) files
- **Content**: Each file contains a sequence of images (frames)
- **Labels**: Binary (0 or 1) indicating pollinator presence
- **Distribution**: Imbalanced - significantly more negative samples than positive

## 📝 Important Notes

- **Local vs Codabench**: Directory structures differ between local testing and Codabench deployment. The bundle handles this automatically.
- **Metadata Files**: Do not modify or delete `metadata.yaml` files - they are required by Codabench.
- **Dependencies**: Make sure to include all required dependencies in your submission.

---

## 👥 Contributors

Master 1 AI - Group 3

---

## 📄 License


---

## 🤝 Support

For questions about the challenge, please refer to the competition pages in the `Competition_Bundle/pages/` directory or contact the organizers through Codabench.
```

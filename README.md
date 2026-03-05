# AI-Challenge-2025-26-Group-3-Pollinator

Master 1 AI challenge creation and resolution - Pollinator Detection Project

## 📋 Project Overview

This repository contains a complete machine learning challenge focused on **Pollinator Detection** using image sequence data. The project is designed to be deployed on Codabench and includes all necessary components for participants to understand, develop, and submit their solutions.

---

## 🗂️ Repository Structure

```
├── AI_Challenge_report.pdf
├── Competition_Bundle
│   ├── competition.yaml
│   ├── ingestion_program
│   │   ├── README.md
│   │   ├── ingestion.py
│   │   ├── metadata.yaml
│   │   └── run_ingestion.py
│   ├── input_data
│   │   └── README.md
│   ├── logo.png
│   ├── pages
│   │   ├── data.md
│   │   ├── evaluation.md
│   │   ├── overview.md
│   │   ├── starting_kit.md
│   │   └── terms.md
│   ├── reference_data
│   │   └── README.md
│   ├── sample_code_submission
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── model1.py
│   │   └── requirements.txt
│   ├── sample_result_submission
│   │   └── README.md
│   ├── scoring_program
│   │   ├── README.md
│   │   ├── metadata.yaml
│   │   ├── run_scoring.py
│   │   └── score.py
│   └── utilities
│       └── compile_bundle.py
├── slides.pdf
├── README.md
├── Starting_Kit
│   ├── README.ipynb
│   ├── README.md
│   ├── best_model_probas.png
│   ├── data
│   │   └── README.md
│   ├── sample_code_submission
│   │   ├── analyze1.py
│   │   ├── analyze2.py
│   │   ├── ci1.png
│   │   ├── ci2.png
│   │   ├── conv1.png
│   │   ├── conv2.png
│   │   ├── matrix1.png
│   │   ├── matrix2.png
│   │   ├── model.py
│   │   └── requirements.txt
│   ├── scale_pos_weight_optimization.png
│   └── submission
│       └── Submission_Code_26-01-31-17-21.zip
├── resolution
│   ├── grain_classification
│   │   ├── README.md
│   │   ├── analyze1.py
│   │   ├── analyze2.py
│   │   ├── ci1.png
│   │   ├── ci2.png
│   │   ├── conv1.png
│   │   ├── conv2.png
│   │   ├── matrix1.png
│   │   ├── matrix2.png
│   │   ├── model.py
│   │   └── model1.py
│   └── pollinators_classification
│       ├── analyze.py
│       ├── ci_group4.png
│       ├── conv_group4.png
│       ├── matrix_group4.png
│       └── model.py
└── split_data.py
```
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

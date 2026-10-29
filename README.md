# Credit Card Fraud Detection

A production-style credit card fraud detection project that combines an unsupervised autoencoder with a supervised XGBoost classifier to detect suspicious transactions in a highly imbalanced dataset.

This repository demonstrates an end-to-end ML workflow for fraud detection, including data preprocessing, feature engineering, model training, evaluation, a FastAPI prediction service, and a Streamlit dashboard.

## Overview

Credit card fraud detection is a classic anomaly detection and class imbalance problem. In this project:

- An autoencoder is trained on normal transactions to learn the expected pattern of legitimate behavior.
- Reconstruction error is used as an anomaly score.
- Engineered behavioral and statistical features are added to improve detection quality.
- A hybrid XGBoost classifier is trained using both engineered features and the anomaly score.
- The trained models are served through an API and can be explored through a dashboard.

## Features

- Data preprocessing and scaling for imbalanced fraud datasets
- Feature engineering using transaction behavior and amount statistics
- PyTorch autoencoder for anomaly detection
- XGBoost classifier for final fraud prediction
- Model evaluation with classification metrics and plots
- FastAPI inference endpoint
- Streamlit dashboard for interactive predictions
- Saved trained model artifacts in the `models/` directory

## Project Structure

```text
credit_card_fraud_detection/
├── api/
│   └── main.py                  # FastAPI service for fraud prediction
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── raw/
│   │   └── creditcard.csv      # Kaggle credit card dataset (not included in repo)
│   └── processed/
│       ├── scaled_data.npy
│       ├── labels.npy
│       └── engineered_features.npy
├── models/
│   ├── autoencoder.pt          # Trained autoencoder weights
│   ├── xgboost.json            # Trained XGBoost model
│   └── evaluation_plots/       # Generated evaluation plots
├── src/
│   ├── data_preprocessing.py  # Scaling and data prep
│   ├── feature_engineering.py  # Engineered transaction features
│   ├── model_training.py       # Autoencoder + XGBoost training pipeline
│   ├── evaluation.py           # Metrics, ROC/PR curves, SHAP analysis
│   ├── utils.py               # Utility placeholder
│   └── models/
│       └── autoencoder.py     # Autoencoder architecture
├── .gitignore
├── requirements.txt
├── main.py
├── README.md
└── LICENSE (if added later)
```

## Dataset

This project is designed for the Kaggle credit card fraud detection dataset:

- Dataset name: Credit Card Fraud Detection
- Source: Kaggle
- Rows: 284,807
- Fraudulent transactions: 492
- Highly imbalanced, making it a strong real-world fraud detection benchmark

Place the dataset in:

```text
data/raw/creditcard.csv
```

If the file is missing, the pipeline will raise a `FileNotFoundError`.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages include:

- pandas
- numpy
- scikit-learn
- torch
- xgboost
- matplotlib
- seaborn
- shap
- fastapi
- streamlit
- uvicorn

## Quick Start

### 1. Prepare the dataset

Download the Kaggle dataset and save it to:

```bash
mkdir -p data/raw
# place creditcard.csv in data/raw/
```

### 2. Run preprocessing

```bash
python src/data_preprocessing.py
```

This creates scaled feature arrays and labels in `data/processed/`.

### 3. Run feature engineering

```bash
python src/feature_engineering.py
```

This adds transaction behavior features such as:

- `Time_diff`
- `Rolling_count_1h`
- `Rolling_amount_1h`
- `Amount_deviation`
- `Amount_zscore`
- `Amount_log`

### 4. Train the models

```bash
python src/model_training.py
```

This trains:

- the autoencoder on legitimate transactions
- the XGBoost model using engineered features plus anomaly scores

The trained artifacts are saved in `models/`.

### 5. Evaluate the model

```bash
python src/evaluation.py
```

This generates evaluation outputs such as:

- confusion matrix
- ROC curve
- precision-recall curve
- anomaly score distribution
- SHAP summary plot

## Run the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Then open the docs at:

```text
http://127.0.0.1:8000/docs
```

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000.0,
    "Amount": 200.0,
    "V1": 0.1,
    "V2": -0.2,
    "V3": 1.5,
    "V4": -0.4,
    "V5": 0.2,
    "V6": 0.8,
    "V7": -1.0,
    "V8": 0.3,
    "V9": -0.1,
    "V10": 0.5,
    "V11": -0.2,
    "V12": 0.6,
    "V13": 1.2,
    "V14": -0.8,
    "V15": 0.4,
    "V16": -0.5,
    "V17": 1.0,
    "V18": -0.3,
    "V19": 0.7,
    "V20": -0.1,
    "V21": 0.2,
    "V22": 0.5,
    "V23": -0.7,
    "V24": 1.1,
    "V25": -0.6,
    "V26": 0.9,
    "V27": -1.2,
    "V28": 0.4,
    "Time_diff": 0.0,
    "Rolling_count_1h": 1.0,
    "Rolling_amount_1h": 50.0,
    "Amount_deviation": 0.0,
    "Amount_zscore": 0.0,
    "Amount_log": 3.9
  }'
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard allows interactive transaction inspection and displays:

- anomaly score
- fraud probability
- prediction result
- SHAP-based feature importance

## Model Architecture

### Autoencoder

The autoencoder learns the patterns of legitimate transactions by reconstructing inputs from a compressed latent representation. Transactions with large reconstruction errors are more likely to be fraudulent.

### XGBoost classifier

The XGBoost model uses:

- PCA-transformed transaction features
- engineered behavioral features
- autoencoder anomaly score

This hybrid approach usually performs better than using only a single model, especially when fraudulent examples are rare.

## Evaluation Metrics

The project tracks metrics such as:

- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- precision-recall curve

These are useful because in fraud detection, false negatives can be expensive.

## Notes

- This repo is intended as an end-to-end fraud detection prototype and learning project.
- The raw dataset is not included in the repository and must be downloaded separately.
- Model performance depends on the dataset, model tuning, and feature engineering choices.

## License

This project is provided for educational and demonstration purposes. Add a license file if you plan to publish or share the repository publicly.

## Contributing

Contributions are welcome. Suggested improvements include:

- better class imbalance handling
- more advanced feature engineering
- hyperparameter tuning
- experiment tracking and model versioning
- deployment with Docker

## Contact

For questions or collaboration, use the repository owner or project maintainer contact information associated with this GitHub project.

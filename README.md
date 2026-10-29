# Credit Card Fraud Detection

A hybrid fraud detection system combining a PyTorch autoencoder for anomaly detection and an XGBoost classifier for final prediction. The project is designed for the highly imbalanced credit card fraud dataset and includes preprocessing, feature engineering, model training, evaluation, an API, and a Streamlit dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/XGBoost-1.x-green?logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

## Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are a small fraction of total activity. This project addresses that imbalance using a two-stage approach:

1. Autoencoder-based anomaly detection
   - The model learns the pattern of normal transactions.
   - Transactions with large reconstruction errors are treated as suspicious.

2. Hybrid supervised classification
   - Engineered behavioral and statistical features are added.
   - An XGBoost model is trained using these features plus the autoencoder anomaly score.

This makes the system more robust than relying on a single classifier or a pure anomaly detector.

## Features

- Data preprocessing and feature scaling
- Behavioral feature engineering
- Statistical feature engineering
- Autoencoder training in PyTorch
- XGBoost fraud classification
- Model evaluation with standard metrics and plots
- FastAPI prediction endpoint
- Streamlit dashboard for interactive predictions
- Saved trained model artifacts under `models/`

## Tech Stack

- Python
- PyTorch
- XGBoost
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- SHAP
- FastAPI
- Streamlit

## Repository Structure

```text
credit_card_fraud_detection/
├── api/
│   └── main.py                 # FastAPI app for fraud prediction
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── raw/
│   │   └── creditcard.csv      # Kaggle dataset (required)
│   └── processed/
│       ├── scaled_data.npy
│       ├── labels.npy
│       └── engineered_features.npy
├── models/
│   ├── autoencoder.pt          # Trained autoencoder weights
│   ├── xgboost.json           # Trained XGBoost model
│   └── evaluation_plots/      # Plots generated during evaluation
├── src/
│   ├── data_preprocessing.py  # Scaling and preprocessing pipeline
│   ├── feature_engineering.py  # Behavioral/statistical feature creation
│   ├── model_training.py       # Autoencoder + XGBoost training workflow
│   ├── evaluation.py           # Metrics and visualization
│   ├── utils.py               # Utility file placeholder
│   └── models/
│       └── autoencoder.py     # Autoencoder architecture
├── .gitignore
├── requirements.txt
├── main.py
├── README.md
└── LICENSE                    # optional if added later
```

## Dataset

This project uses the Kaggle credit-card fraud detection dataset.

- Dataset: `creditcard.csv`
- Rows: 284,807
- Fraudulent transactions: 492
- Class imbalance: extremely high

The dataset should be placed in:

```bash
data/raw/creditcard.csv
```

If the file is not present, the scripts will raise a `FileNotFoundError`.

## Installation

Clone the repository:

```bash
git clone https://github.com/Tjiaz/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Preprocess the data

```bash
python src/data_preprocessing.py
```

This script:
- loads the raw dataset
- separates features and labels
- scales the features
- saves processed arrays to `data/processed/`

### 2. Run feature engineering

```bash
python src/feature_engineering.py
```

This adds engineered features such as:

- `Time_diff`
- `Rolling_count_1h`
- `Rolling_amount_1h`
- `Amount_deviation`
- `Amount_zscore`
- `Amount_log`

### 3. Train the models

```bash
python src/model_training.py
```

This trains:
- the PyTorch autoencoder on legitimate transactions
- the XGBoost classifier using engineered features plus anomaly score

The trained model files are saved in the `models/` directory.

### 4. Evaluate the model

```bash
python src/evaluation.py
```

This generates evaluation outputs including:
- confusion matrix
- ROC curve
- precision-recall curve
- anomaly distribution plot
- SHAP summary plot

## Running the API

Start the FastAPI application:

```bash
uvicorn api.main:app --reload
```

Then open:

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

The API returns:

```json
{
  "anomaly_score": 0.0123,
  "fraud_probability": 0.89,
  "fraud_prediction": 1,
  "status": "fraud"
}
```

## Running the Dashboard

Launch the Streamlit app:

```bash
streamlit run dashboard/app.py
```

The dashboard allows you to input a transaction and inspect:
- anomaly score
- probability of fraud
- fraud prediction label
- SHAP feature importance
- reconstruction error visualization

## Model Details

### Autoencoder

The autoencoder is trained only on non-fraudulent data. It learns the typical structure of legitimate transactions and reconstructs them with low error. Fraudulent transactions tend to deviate from this pattern and produce a larger reconstruction error.

### XGBoost hybrid classifier

The final XGBoost model uses:
- PCA-transformed transaction features
- engineered behavioral/statistical features
- anomaly score from the autoencoder

This hybrid approach often performs better than simple logistic regression or pure anomaly detection on rare-event datasets.

## Evaluation

The project evaluates its results using metrics such as:

- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- precision-recall curve

Because fraud detection is imbalanced, recall and precision are especially important.

## Notes

- The raw dataset is not included in the repository and must be downloaded separately from Kaggle.
- This project is intended as an end-to-end ML prototype and learning project.
- Model performance depends on tuning, data quality, and feature engineering choices.

## License

This project is provided for educational and demonstration purposes. If you plan to distribute or publish it publicly, add an appropriate license file such as MIT.

## Contributing

Contributions are welcome. Ideas for improvement include:
- better class-imbalance handling
- hyperparameter tuning
- experiment tracking with MLflow or Weights & Biases
- model versioning
- Docker deployment

## Contact

For questions or collaboration, use the repository owner or maintainer contact details associated with this GitHub project.

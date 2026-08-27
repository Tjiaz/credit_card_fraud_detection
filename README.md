# credit_card_fraud_detection

📘 Credit Card Fraud Detection — End‑to‑End Machine Learning System
🔍 Overview
This project is a production‑ready fraud detection system built using:

PyTorch Autoencoder for anomaly detection

XGBoost for supervised classification

Hybrid modeling combining anomaly scores + engineered features

Modular Python pipeline (data preprocessing → feature engineering → model training → evaluation)

Industry‑grade engineering practices

The goal is to detect fraudulent credit card transactions using a combination of unsupervised learning and supervised learning, achieving high precision and recall on an extremely imbalanced dataset.

🧱 Project Architecture

fraud_detection/
│
├── data/
│ ├── raw/ # creditcard.csv (Kaggle dataset)
│ └── processed/ # scaled data, engineered features, labels
│
├── src/
│ ├── data_preprocessing.py # scaling, cleaning, saving processed data
│ ├── feature_engineering.py # behavioural + statistical fraud features
│ ├── model_training.py # autoencoder + XGBoost hybrid training
│ ├── evaluation.py # ROC, PR, confusion matrix, SHAP
│ └── models/
│ └── autoencoder.py # PyTorch autoencoder architecture
│
├── models/ # saved .pt and .json models
├── notebooks/ # optional exploratory notebooks
├── dashboard/ # Streamlit app (future)
├── api/ # FastAPI service (future)
└── README.md

📊 Dataset
Source: Kaggle — Credit Card Fraud Detection  
Rows: 284,807
Fraud cases: 492 (0.172%)
Features: PCA‑transformed V1–V28 + Time + Amount + engineered features

This dataset is highly imbalanced, making it ideal for anomaly detection.

🧪 Modeling Approach
1️⃣ Unsupervised Learning — PyTorch Autoencoder
The autoencoder learns normal transaction patterns and produces an anomaly score using reconstruction error.

Trained only on normal transactions

Reconstruction error = anomaly score

Higher score → more suspicious

This mirrors real bank fraud systems.

2️⃣ Feature Engineering
Added behavioural and statistical features:

Behavioural Features
Time difference between transactions

Rolling transaction count (1 hour)

Rolling amount sum (1 hour)

Deviation from mean spending

Statistical Features
Log‑transformed amount

Z‑score of amount

These features significantly improve model performance.

3️⃣ Hybrid Model — XGBoost
XGBoost uses:

Engineered features

Autoencoder anomaly score

This hybrid approach achieves high precision and recall, outperforming pure anomaly detection.

📈 Results
Autoencoder
Loss decreased from extremely high initial values to stable reconstruction error — indicating successful learning of normal patterns.

XGBoost Performance

precision (fraud): 0.94
recall (fraud): 0.82
f1-score (fraud): 0.87
ROC-AUC: 0.91

Interpretation
High precision → flagged fraud is almost always correct

Strong recall → model catches most fraud cases

High ROC-AUC → excellent separation between fraud and normal

This is enterprise‑grade performance.

📉 Evaluation Tools
The evaluation pipeline includes:

Confusion Matrix

ROC Curve

Precision‑Recall Curve

Anomaly Score Distribution

SHAP Explainability for XGBoost

These tools provide full transparency and model interpretability.

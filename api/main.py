from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import xgboost as xgb

from src.models.autoencoder import Autoencoder

# -----------------------------
# Paths
# -----------------------------
AUTOENCODER_MODEL_PATH = "models/autoencoder.pt"
XGBOOST_MODEL_PATH = "models/xgboost.json"

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Hybrid Model: PyTorch Autoencoder + XGBoost",
    version="1.0"
)

# -----------------------------
# Request Schema
# -----------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Time_diff: float = 0.0
    Rolling_count_1h: float = 0.0
    Rolling_amount_1h: float = 0.0
    Amount_deviation: float = 0.0
    Amount_zscore: float = 0.0
    Amount_log: float = 0.0


# -----------------------------
# Load Models
# -----------------------------
def load_models(input_dim):
    autoencoder = Autoencoder(input_dim=input_dim)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH))
    autoencoder.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGBOOST_MODEL_PATH)

    return autoencoder, xgb_model


# -----------------------------
# Compute anomaly score
# -----------------------------
def compute_anomaly_score(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    return float(mse.numpy()[0])


# -----------------------------
# API Endpoint
# -----------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to numpy array
    features = np.array([[value for value in transaction.dict().values()]], dtype=float)

    # Load models
    autoencoder, xgb_model = load_models(input_dim=features.shape[1])

    # Compute anomaly score
    anomaly_score = compute_anomaly_score(autoencoder, features)

    # Add anomaly score as last feature
    features_with_anomaly = np.hstack([features, [[anomaly_score]]])

    # Fraud probability
    fraud_prob = float(xgb_model.predict_proba(features_with_anomaly)[0][1])
    fraud_pred = int(xgb_model.predict(features_with_anomaly)[0])

    return {
        "anomaly_score": anomaly_score,
        "fraud_probability": fraud_prob,
        "fraud_prediction": fraud_pred,
        "status": "fraud" if fraud_pred == 1 else "normal"
    }

import streamlit as st
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from src.models.autoencoder import Autoencoder

# Paths
AUTOENCODER_MODEL_PATH = "models/autoencoder.pt"
XGBOOST_MODEL_PATH = "models/xgboost.json"

# Load models
@st.cache_resource
def load_models(input_dim):
    autoencoder = Autoencoder(input_dim=input_dim)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH))
    autoencoder.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGBOOST_MODEL_PATH)

    return autoencoder, xgb_model


# Compute anomaly score
def compute_anomaly_score(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    return mse.numpy()[0]


# Streamlit UI
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Hybrid Model: PyTorch Autoencoder + XGBoost")

st.divider()

st.header("🔢 Enter Transaction Features")

# Input fields
feature_names = [
    "Time", "Amount",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
    "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28",
    "Time_diff", "Rolling_count_1h", "Rolling_amount_1h", "Amount_deviation", "Amount_zscore", "Amount_log"
]

inputs = {}

cols = st.columns(4)
for i, feature in enumerate(feature_names):
    col = cols[i % 4]
    inputs[feature] = col.number_input(feature, value=0.0)

# Convert to array
transaction = np.array([list(inputs.values())]).astype(float)

# Load models. The anomaly score is added only for the XGBoost input.
autoencoder, xgb_model = load_models(input_dim=transaction.shape[1])

if st.button("Run Fraud Detection"):
    st.subheader("📌 Results")

    # Compute anomaly score
    anomaly_score = compute_anomaly_score(autoencoder, transaction)

    # Add anomaly score as feature
    transaction_with_anomaly = np.hstack([transaction, [[anomaly_score]]])

    # Fraud probability
    fraud_prob = xgb_model.predict_proba(transaction_with_anomaly)[0][1]
    fraud_pred = xgb_model.predict(transaction_with_anomaly)[0]

    # Display results
    st.metric("Anomaly Score", f"{anomaly_score:.6f}")
    st.metric("Fraud Probability", f"{fraud_prob:.4f}")

    if fraud_pred == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✔️ Transaction Appears Normal")

    st.divider()

    # SHAP Explanation
    st.subheader("🔍 SHAP Feature Importance")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transaction_with_anomaly)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=transaction_with_anomaly[0]), 
                        max_display=10)
    st.pyplot(fig)

    st.divider()

    # Reconstruction error visualization
    st.subheader("📉 Reconstruction Error Visualization")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["Reconstruction Error"], [anomaly_score], color="red")
    ax2.set_ylabel("Error")
    st.pyplot(fig2)

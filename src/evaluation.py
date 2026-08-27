import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import shap
import xgboost as xgb
import torch
try:
    from src.models.autoencoder import Autoencoder
except ModuleNotFoundError:
    from models.autoencoder import Autoencoder

# Paths
ENGINEERED_FEATURES_PATH = os.path.join("data", "processed", "engineered_features.npy")
LABELS_PATH = os.path.join("data", "processed", "labels.npy")
AUTOENCODER_MODEL_PATH = os.path.join("models", "autoencoder.pt")
XGBOOST_MODEL_PATH = os.path.join("models", "xgboost.json")
PLOTS_DIR = os.path.join("models", "evaluation_plots")


def load_data():
    """Load engineered features and labels."""
    X = np.load(ENGINEERED_FEATURES_PATH)
    y = np.load(LABELS_PATH)
    return X, y


def load_models(input_dim):
    """Load trained autoencoder and XGBoost model."""
    autoencoder = Autoencoder(input_dim=input_dim)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH))
    autoencoder.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGBOOST_MODEL_PATH)

    return autoencoder, xgb_model


def compute_anomaly_scores(autoencoder, X):
    """Compute reconstruction error for each transaction."""
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = autoencoder(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    return mse.numpy()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), bbox_inches="tight")
    plt.close()


def plot_precision_recall(y_true, y_scores):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"), bbox_inches="tight")
    plt.close()


def plot_anomaly_distribution(anomaly_scores, y):
    """Plot anomaly score distribution for fraud vs normal."""
    plt.figure(figsize=(8, 5))
    sns.histplot(anomaly_scores[y == 0], color="blue", label="Normal", bins=50, kde=True)
    sns.histplot(anomaly_scores[y == 1], color="red", label="Fraud", bins=50, kde=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Reconstruction Error")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_distribution.png"), bbox_inches="tight")
    plt.close()


def shap_explain_xgboost(model, X):
    """Generate SHAP explainability for XGBoost."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:500])  # limit for speed

    shap.summary_plot(shap_values, X[:500], show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), bbox_inches="tight")
    plt.close()


def run_evaluation():
    """Full evaluation pipeline."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Loading data...")
    X, y = load_data()

    print("Loading models...")
    autoencoder, xgb_model = load_models(input_dim=X.shape[1])

    print("Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(autoencoder, X)

    # Add anomaly score as last feature
    X_with_anomaly = np.hstack([X, anomaly_scores.reshape(-1, 1)])

    print("Generating predictions...")
    y_pred = xgb_model.predict(X_with_anomaly)
    y_scores = xgb_model.predict_proba(X_with_anomaly)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y, y_pred)

    print("\nPlotting ROC curve...")
    plot_roc_curve(y, y_scores)

    print("\nPlotting Precision-Recall curve...")
    plot_precision_recall(y, y_scores)

    print("\nPlotting anomaly score distribution...")
    plot_anomaly_distribution(anomaly_scores, y)

    print("\nGenerating SHAP explainability...")
    shap_explain_xgboost(xgb_model, X_with_anomaly)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    run_evaluation()

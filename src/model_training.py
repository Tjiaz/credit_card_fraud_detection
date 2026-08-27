import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

try:
    from src.models.autoencoder import Autoencoder
except ModuleNotFoundError:
    from models.autoencoder import Autoencoder

# Paths
ENGINEERED_FEATURES_PATH = os.path.join("data", "processed", "engineered_features.npy")
LABELS_PATH = os.path.join("data", "processed", "labels.npy")

AUTOENCODER_MODEL_PATH = os.path.join("models", "autoencoder.pt")
XGBOOST_MODEL_PATH = os.path.join("models", "xgboost.json")


def load_data():
    """Load engineered features and labels."""
    X = np.load(ENGINEERED_FEATURES_PATH)
    y = np.load(LABELS_PATH)
    return X, y


def train_autoencoder(X, y, epochs=50, batch_size=32, lr=1e-4):
    """Train PyTorch autoencoder on normal transactions."""
    print("Training autoencoder...")

    # Train only on normal transactions
    X_normal = X[y == 0]

    model = Autoencoder(input_dim=X.shape[1])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_normal, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), AUTOENCODER_MODEL_PATH)
    print(f"Autoencoder saved to {AUTOENCODER_MODEL_PATH}")

    return model


def compute_anomaly_scores(model, X):
    """Compute reconstruction error for each transaction."""
    print("Computing anomaly scores...")

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

    return mse.numpy()


def train_xgboost(X, y):
    """Train XGBoost classifier using engineered features + anomaly score."""
    print("Training XGBoost classifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)

    print("XGBoost Results:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {auc:.4f}")

    # Save model
    model.save_model(XGBOOST_MODEL_PATH)
    print(f"XGBoost model saved to {XGBOOST_MODEL_PATH}")

    return model


def run_training_pipeline():
    """Full training pipeline."""
    print("Loading data...")
    X, y = load_data()

    print("Training autoencoder...")
    autoencoder = train_autoencoder(X, y)

    print("Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(autoencoder, X)

    # Add anomaly score as a new feature
    X_with_anomaly = np.hstack([X, anomaly_scores.reshape(-1, 1)])

    print("Training hybrid model (XGBoost)...")
    train_xgboost(X_with_anomaly, y)

    print("Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()

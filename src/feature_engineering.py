import os
import pandas as pd
import numpy as np


PROCESSED_DATA_PATH = os.path.join("data", "processed", "scaled_data.npy")
LABELS_PATH = os.path.join("data", "processed", "labels.npy")
RAW_DATA_PATH = os.path.join("data", "raw", "creditcard.csv")

FEATURES_OUTPUT_PATH = os.path.join("data", "processed", "engineered_features.npy")

def load_data():
    """Load raw CSV and processed scaled data."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Raw dataset missing. Place creditcard.csv in data/raw/")
    
    df = pd.read_csv(RAW_DATA_PATH)

    X_scaled = np.load(PROCESSED_DATA_PATH)
    y = np.load(LABELS_PATH)

    return df, X_scaled, y


def add_behavioural_features(df):
    """
    Add behavioural fraud features:
    - Transactiona time differences 
    - Rolling transaction count
    - Rolling transaction amount sum 
    - Amount devaiton from mean
    """
    df = df.copy()

    #Time difference between transactions
    df['Time_diff'] = df['Time'].diff().fillna(0)

    #Rolling count of transactions in last 1hour (3600 seconds)
    df['Rolling_count_1h'] = df['Time'].rolling(window=3600, min_periods=1).count()

    #Rolling sum of transaction amounts in last 1hour (3600 seconds)
    df['Rolling_amount_1h'] = df['Amount'].rolling(window=3600, min_periods=1).sum()

    #Amount deviation from mean
    df["Amount_mean"] = df["Amount"].rolling(window=100).mean().fillna(df["Amount"].mean())
    df['Amount_deviation'] = df['Amount'] - df['Amount_mean']

    return df

def add_statistical_features(df):
    """
    Add statistical features:
    - Z-score of transaction amount
    - Log transformation of transaction amount
    """
    df = df.copy()

    #Z-score of transaction amount
    df['Amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    #Log transformation of transaction amount
    df['Amount_log'] = np.log1p(df['Amount'])

    return df

def merge_engineered_features(X_scaled, df):
    """
    Merge engineered features with scaled features.
    """
    # Select engineered features
    engineered_features = df[['Time_diff', 'Rolling_count_1h', 'Rolling_amount_1h', 'Amount_deviation', 'Amount_zscore', 'Amount_log']].values

    # Concatenate scaled features with engineered features
    X_final = np.hstack((X_scaled, engineered_features))

    return X_final

def save_engineered_features(features):
    """Save engineered features to a .npy file."""
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    np.save(FEATURES_OUTPUT_PATH, features)
    print(f"Engineered features saved to {FEATURES_OUTPUT_PATH}")

def run_feature_engineering():
    """Run the full feature engineering pipeline."""
    print("Loading data...")
    df, X_scaled, y = load_data()


    print("Adding behavioural features...")
    df = add_behavioural_features(df)

    print("Adding statistical features...")
    df = add_statistical_features(df)

    print("Merging engineered features with scaled features...")
    X_final = merge_engineered_features(X_scaled, df)

    print("Saving engineered features...")
    save_engineered_features(X_final)

if __name__ == "__main__":
    run_feature_engineering()
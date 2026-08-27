import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


#Path 
RAW_DATA_PATH = os.path.join("data", "raw", "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "scaled_data.npy")
LABELS_PATH = os.path.join("data", "processed", "labels.npy")


def load_raw_data():
    """ load the raw credot card dataset"""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    return df


def preprocess_data(df):
    """ Preprocess the raw credit card dataset:
    - Seperate features and labels
    - Scale features 
    """
    #Seperate labels
    y = df['Class'].values

    #Drop label column
    X = df.drop(columns=['Class']).values

    #Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def save_processed_data(X_scaled, y):
    """ Save processed data to the processed folder """
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    np.save(PROCESSED_DATA_PATH, X_scaled)
    np.save(LABELS_PATH, y)

    print(f"Processed data saved to {PROCESSED_DATA_PATH} and labels saved to {LABELS_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


def run_preprocessing():
    """ Run full preprocessing pipeline"""
    print("Loading raw data...")
    df = load_raw_data()

    print("Preprocessing data...")
    X_scaled, y = preprocess_data(df)

    print("Saving processed data...")
    save_processed_data(X_scaled, y)

if __name__ == "__main__":
    run_preprocessing()
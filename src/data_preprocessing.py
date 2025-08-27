# src/data_preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath, save_scaler_path=None, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    df = df.dropna()
    X = df.drop("default", axis=1)
    y = df["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)

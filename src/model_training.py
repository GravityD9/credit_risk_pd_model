# src/model_training.py
import os
import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, model_path="models/log_reg_model.pkl"):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

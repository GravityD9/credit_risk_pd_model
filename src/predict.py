# src/predict.py
import joblib
import numpy as np

def predict_default(sample_features, model_path="models/log_reg_model.pkl", scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    sample_arr = np.array(sample_features).reshape(1, -1)
    sample_scaled = scaler.transform(sample_arr)

    prob_default = model.predict_proba(sample_scaled)[:, 1][0]
    prediction = int(model.predict(sample_scaled)[0])
    return prediction, prob_default

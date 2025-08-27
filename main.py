# main.py
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.evaluate_model import evaluate_model
from src.predict import predict_default
import os
import joblib

if __name__ == "__main__":
    data_path = "data/credit_data.csv"
    model_path = "models/log_reg_model.pkl"
    scaler_path = "models/scaler.pkl"

    os.makedirs("models", exist_ok=True)

    # Load & preprocess (this will save scaler to models/scaler.pkl)
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_preprocess_data(
        data_path, save_scaler_path=scaler_path
    )

    # Train
    model = train_model(X_train, y_train, model_path)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    print("Feature order (use this when calling predict):", feature_cols)

    # Predict sample applicant (order must match feature_cols)
    sample_applicant = [50000, 5, 0.30, 2]  # [income, years_employed, debt_ratio, dependents]
    prediction, pd_prob = predict_default(sample_applicant, model_path=model_path, scaler_path=scaler_path)
    print("\nSample Applicant Prediction:", "Default" if prediction == 1 else "No Default")
    print("Probability of Default (PD):", round(pd_prob, 4))

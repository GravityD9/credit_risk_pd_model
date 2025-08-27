# src/evaluate_model.py
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

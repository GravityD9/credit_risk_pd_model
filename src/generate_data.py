# src/generate_data.py
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_samples = 1000

data = {
    "income": np.random.normal(50000, 15000, n_samples).astype(int),
    "years_employed": np.random.randint(0, 40, n_samples),
    "debt_ratio": np.round(np.random.uniform(0, 1, n_samples), 2),
    "dependents": np.random.randint(0, 5, n_samples),
}

logits = (
    -3
    + 0.00003 * (60000 - data["income"])
    + 2.5 * data["debt_ratio"]
    + 0.05 * (5 - data["years_employed"])
)

prob_default = 1 / (1 + np.exp(-logits))
default = np.random.binomial(1, prob_default)

data["default"] = default
df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_data.csv", index=False)
print("Saved synthetic dataset to data/credit_data.csv")

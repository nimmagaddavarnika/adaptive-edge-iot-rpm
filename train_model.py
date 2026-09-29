import os

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

import joblib


INPUT_FILE = "data/patient_demo.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "anomaly_model.joblib"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


# ------------------------------------------------------------
# Select features
# ------------------------------------------------------------

features = [
    "bpm",
    "spo2"
]


for feature in features:

    if feature not in df.columns:

        raise ValueError(
            f"Missing feature: {feature}"
        )


X = df[features].copy()


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)


model.fit(X)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)


print(
    "Anomaly model trained successfully."
)

print(
    f"Features: {features}"
)

print(
    f"Saved to: {MODEL_FILE}"
)
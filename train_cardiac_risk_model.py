"""
Trains an interpretable Random Forest model to classify cardiac risk level
from physiological features. Uses rule-based synthetic labeling since
real labeled cardiac-event data is not available for this academic prototype.

Labels: 0 = NORMAL, 1 = CARDIAC ANOMALY, 2 = HIGH-RISK CARDIAC PATTERN

Features (all derived from live sensor pipeline, ECG is a placeholder
until AD8232 hardware is integrated):
  hr                  - current heart rate (from MAX30102)
  hr_dev_from_baseline- |hr - personal_baseline_hr|
  spo2                - blood oxygen (placeholder until real SpO2 calc added)
  signal_quality      - 0-1 score, based on IR amplitude stability
  motion_intensity    - magnitude of accel deviation from rest (MPU6050)
  ecg_abnormality     - 0-1 placeholder score (real ECG feature slot)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

np.random.seed(42)
N_SAMPLES = 3000


def generate_synthetic_row():
    baseline_hr = np.random.normal(72, 6)

    # 70% normal, 20% anomaly, 10% high-risk — realistic class imbalance
    r = np.random.rand()
    if r < 0.70:
        hr = baseline_hr + np.random.normal(0, 5)
        spo2 = np.random.normal(97.5, 1.0)
        signal_quality = np.random.uniform(0.7, 1.0)
        motion_intensity = np.random.uniform(0, 0.3)
        ecg_abnormality = np.random.uniform(0, 0.2)
        label = 0
    elif r < 0.90:
        hr = baseline_hr + np.random.normal(25, 10)
        spo2 = np.random.normal(94.5, 1.5)
        signal_quality = np.random.uniform(0.4, 0.8)
        motion_intensity = np.random.uniform(0, 0.6)
        ecg_abnormality = np.random.uniform(0.2, 0.5)
        label = 1
    else:
        hr = baseline_hr + np.random.normal(45, 15)
        spo2 = np.random.normal(90.0, 2.0)
        signal_quality = np.random.uniform(0.5, 1.0)  # good quality, genuinely abnormal
        motion_intensity = np.random.uniform(0, 0.3)   # low motion, so it's not just an artifact
        ecg_abnormality = np.random.uniform(0.5, 0.9)
        label = 2

    hr = max(30, hr)
    spo2 = min(100, max(70, spo2))
    signal_quality = min(1.0, max(0.0, signal_quality))
    motion_intensity = max(0.0, motion_intensity)
    ecg_abnormality = min(1.0, max(0.0, ecg_abnormality))

    hr_dev_from_baseline = abs(hr - baseline_hr)

    # Add small amount of realistic overlap/noise so classes aren't perfectly separable
    if np.random.rand() < 0.05:
        label = np.random.choice([0, 1, 2])

    return {
        "hr": hr,
        "hr_dev_from_baseline": hr_dev_from_baseline,
        "spo2": spo2,
        "signal_quality": signal_quality,
        "motion_intensity": motion_intensity,
        "ecg_abnormality": ecg_abnormality,
        "label": label,
    }


print("Generating synthetic training data...")
rows = [generate_synthetic_row() for _ in range(N_SAMPLES)]
df = pd.DataFrame(rows)

print(df["label"].value_counts().sort_index())
print()

FEATURES = ["hr", "hr_dev_from_baseline", "spo2", "signal_quality",
            "motion_intensity", "ecg_abnormality"]

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test, y_pred,
    target_names=["NORMAL", "CARDIAC ANOMALY", "HIGH-RISK CARDIAC PATTERN"]
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/cardiac_risk_model.joblib"
joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

print()
print(f"Model saved to {MODEL_PATH}")
print("TRAINING COMPLETE")
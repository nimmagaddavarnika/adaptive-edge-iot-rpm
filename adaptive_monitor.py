import os

import pandas as pd
import numpy as np

import joblib


DATA_FILE = "data/patient_demo.csv"

BASELINE_FILE = "data/patient_baselines.csv"

MODEL_FILE = "models/anomaly_model.joblib"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

baseline = pd.read_csv(
    BASELINE_FILE
)

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# BASELINE
# ============================================================

bpm_mean = float(
    baseline.loc[
        baseline["metric"] == "bpm",
        "baseline_mean"
    ].iloc[0]
)

bpm_std = float(
    baseline.loc[
        baseline["metric"] == "bpm",
        "baseline_std"
    ].iloc[0]
)


if bpm_std == 0:

    bpm_std = 1.0


# ============================================================
# ML PREDICTION
# ============================================================

X = df[
    [
        "bpm",
        "spo2"
    ]
]


df["ml_prediction"] = model.predict(
    X
)


# IsolationForest:
#
# 1  = normal
# -1 = anomaly

df["ml_anomaly"] = (
    df["ml_prediction"] == -1
)


# ============================================================
# PERSONALIZED DEVIATION
# ============================================================

df["bpm_zscore"] = (
    (df["bpm"] - bpm_mean)
    / bpm_std
)


# ============================================================
# ADAPTIVE DECISION
# ============================================================

def adaptive_decision(row):

    bpm = row["bpm"]

    z = abs(
        row["bpm_zscore"]
    )

    activity = row.get(
        "activity",
        "UNKNOWN"
    )

    ml_anomaly = bool(
        row["ml_anomaly"]
    )


    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if (
        z < 2
        and not ml_anomaly
    ):

        return (
            "NORMAL",
            "NORMAL_SAMPLING"
        )


    # --------------------------------------------------------
    # Activity-related change
    # --------------------------------------------------------

    if (
        activity == "HIGH_ACTIVITY"
        and bpm > bpm_mean
        and z < 3
    ):

        return (
            "ACTIVITY_RELATED",
            "NORMAL_SAMPLING"
        )


    # --------------------------------------------------------
    # Strong anomaly
    # --------------------------------------------------------

    if (
        z >= 3
        and ml_anomaly
    ):

        return (
            "CRITICAL",
            "HIGH_SAMPLING"
        )


    # --------------------------------------------------------
    # General warning
    # --------------------------------------------------------

    return (
        "WARNING",
        "INCREASED_SAMPLING"
    )


results = df.apply(
    adaptive_decision,
    axis=1
)


df[
    "adaptive_status"
] = [
    r[0] for r in results
]


df[
    "sampling_policy"
] = [
    r[1] for r in results
]


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/analyzed_patient_data.csv"
)


df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "Adaptive Edge-IoT Analysis"
)
print(
    "==========================="
)
print()

print(
    f"Personalized baseline BPM: "
    f"{bpm_mean:.2f}"
)

print()

print(
    "Adaptive status:"
)

print(
    df["adaptive_status"]
    .value_counts()
)

print()

print(
    "Sampling policy:"
)

print(
    df["sampling_policy"]
    .value_counts()
)

print()

print(
    f"Saved to: {output_file}"
)
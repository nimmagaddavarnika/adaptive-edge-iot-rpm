import os
import pandas as pd
import numpy as np


INPUT_FILE = "data/patient_demo.csv"

BASELINE_FILE = "data/patient_baselines.csv"

OUTPUT_FILE = "data/analyzed_patient_data.csv"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)

baseline = pd.read_csv(
    BASELINE_FILE
)


# ------------------------------------------------------------
# Extract baseline
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Prevent division by zero
# ------------------------------------------------------------

if bpm_std == 0:

    bpm_std = 1.0


# ------------------------------------------------------------
# Personalized deviation
# ------------------------------------------------------------

df["bpm_deviation"] = (
    df["bpm"] - bpm_mean
)

df["bpm_zscore"] = (
    df["bpm_deviation"]
    / bpm_std
)


# ------------------------------------------------------------
# Activity-aware interpretation
# ------------------------------------------------------------

def classify(row):

    bpm = row["bpm"]

    activity = row.get(
        "activity",
        "UNKNOWN"
    )

    z = abs(
        row["bpm_zscore"]
    )


    if z < 2:

        return "NORMAL"


    if (
        activity == "HIGH_ACTIVITY"
        and bpm > bpm_mean
    ):

        return "ACTIVITY_RELATED"


    if z >= 3:

        return "CRITICAL"


    return "WARNING"


df["status"] = df.apply(
    classify,
    axis=1
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Data analysis completed."
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()

print(
    df["status"].value_counts()
)
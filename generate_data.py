import os
import numpy as np
import pandas as pd


DATA_DIR = "data"

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "patient_demo.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)


np.random.seed(42)


N = 1000


timestamps = pd.date_range(
    start="2026-01-01 09:00:00",
    periods=N,
    freq="s"
)


# ------------------------------------------------------------
# Baseline physiological values
# ------------------------------------------------------------

bpm = np.random.normal(
    loc=75,
    scale=4,
    size=N
)

spo2 = np.random.normal(
    loc=98,
    scale=0.5,
    size=N
)


# ------------------------------------------------------------
# Activity
# ------------------------------------------------------------

activity = np.random.choice(
    [
        "REST",
        "LIGHT_ACTIVITY",
        "HIGH_ACTIVITY"
    ],
    size=N,
    p=[
        0.60,
        0.30,
        0.10
    ]
)


# ------------------------------------------------------------
# Modify HR according to activity
# ------------------------------------------------------------

for i in range(N):

    if activity[i] == "LIGHT_ACTIVITY":

        bpm[i] += np.random.normal(
            10,
            3
        )

    elif activity[i] == "HIGH_ACTIVITY":

        bpm[i] += np.random.normal(
            30,
            6
        )


# ------------------------------------------------------------
# Add a few synthetic anomalies
# ------------------------------------------------------------

anomaly_indices = np.random.choice(
    N,
    size=20,
    replace=False
)


for i in anomaly_indices:

    bpm[i] += np.random.choice(
        [
            -30,
            45
        ]
    )

    spo2[i] -= np.random.uniform(
        2,
        5
    )


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "bpm": bpm,
        "spo2": spo2,
        "activity": activity
    }
)


df["bpm"] = df["bpm"].round(1)

df["spo2"] = df["spo2"].round(1)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Generated {len(df)} patient records."
)

print(
    f"Saved to: {OUTPUT_FILE}"
)
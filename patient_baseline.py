import os
import pandas as pd


INPUT_FILE = "data/patient_demo.csv"

OUTPUT_FILE = "data/patient_baselines.csv"


df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "bpm"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ------------------------------------------------------------
# Calculate baseline
# ------------------------------------------------------------

baseline_bpm = df["bpm"].mean()

baseline_std = df["bpm"].std()

baseline_min = df["bpm"].quantile(0.05)

baseline_max = df["bpm"].quantile(0.95)


baseline = pd.DataFrame(
    {
        "metric": [
            "bpm"
        ],

        "baseline_mean": [
            baseline_bpm
        ],

        "baseline_std": [
            baseline_std
        ],

        "baseline_5_percentile": [
            baseline_min
        ],

        "baseline_95_percentile": [
            baseline_max
        ]
    }
)


baseline.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Patient baseline created.")

print(
    f"Mean BPM: {baseline_bpm:.2f}"
)

print(
    f"Standard deviation: {baseline_std:.2f}"
)

print(
    f"5th percentile: {baseline_min:.2f}"
)

print(
    f"95th percentile: {baseline_max:.2f}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)
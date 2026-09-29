import streamlit as st
import serial
import time
import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from collections import deque

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
MODEL_PATH = "models/cardiac_risk_model.joblib"

LABELS = ["NORMAL", "CARDIAC ANOMALY", "HIGH-RISK CARDIAC PATTERN"]
LABEL_COLORS = {"NORMAL": "green", "CARDIAC ANOMALY": "orange", "HIGH-RISK CARDIAC PATTERN": "red"}

st.set_page_config(page_title="Explainable Cardiac Risk Module", layout="wide")
st.title("Explainable Cardiac Risk & Cause-Analysis Module")
st.caption("Academic research prototype — decision-support/monitoring system, NOT a clinical diagnostic device.")

# --- Load model ---
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# --- Load reference data for LIME background (same distribution style as training) ---
np.random.seed(1)
reference_data = pd.DataFrame({
    "hr": np.random.normal(85, 20, 500),
    "hr_dev_from_baseline": np.abs(np.random.normal(15, 15, 500)),
    "spo2": np.random.normal(95, 3, 500),
    "signal_quality": np.random.uniform(0, 1, 500),
    "motion_intensity": np.random.uniform(0, 0.6, 500),
    "ecg_abnormality": np.random.uniform(0, 1, 500),
})[features]

lime_explainer = LimeTabularExplainer(
    training_data=reference_data.values,
    feature_names=features,
    class_names=LABELS,
    mode="classification",
    discretize_continuous=True,
)

shap_explainer = shap.TreeExplainer(model)

# --- Connect to ESP32 for real HR + motion ---
if "ser" not in st.session_state:
    try:
        st.session_state.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        st.session_state.connected = True
    except Exception as e:
        st.session_state.connected = False
        st.session_state.error = str(e)

if "baseline_hr" not in st.session_state:
    st.session_state.baseline_hr = 72  # default resting baseline; replace with patient_baselines.csv value if available

if "last_real" not in st.session_state:
    st.session_state.last_real = {"hr": 72, "ax": 0, "ay": 0, "az": 16000}

# ============================================================
# SIDEBAR: Simulated inputs (until ECG/SpO2 hardware is ready) + Patient Profile
# ============================================================
st.sidebar.header("Simulated Inputs (hardware pending)")
st.sidebar.caption("HR and motion below come from REAL sensors. SpO2 and ECG abnormality are SIMULATED until AD8232/SpO2 are integrated.")

sim_spo2 = st.sidebar.slider("Simulated SpO2 (%)", 80.0, 100.0, 97.0, 0.1)
sim_ecg_abnormality = st.sidebar.slider("Simulated ECG Abnormality Score", 0.0, 1.0, 0.10, 0.01)

st.sidebar.header("Patient Risk Factors")
st.sidebar.caption("Static profile information — NOT sensor-derived. Does not imply causation.")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
smoking = st.sidebar.checkbox("Smoking / Tobacco use")
hypertension = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
high_cholesterol = st.sidebar.checkbox("High cholesterol")
obesity = st.sidebar.checkbox("Obesity / Overweight")
inactivity = st.sidebar.checkbox("Physical inactivity")
family_history = st.sidebar.checkbox("Family history of heart disease")
other_history = st.sidebar.text_area("Other relevant medical history", "")

# ============================================================
# Read live serial data (non-blocking-ish, grabs what's waiting)
# ============================================================
if st.session_state.connected:
    ser = st.session_state.ser
    lines_read = 0
    while ser.in_waiting and lines_read < 50:
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        lines_read += 1
        if raw.startswith("DATA,"):
            parts = raw.split(",")
            if len(parts) == 12:
                try:
                    finger = int(parts[2])
                    if finger == 1:
                        st.session_state.last_real = {
                            "hr": float(parts[4]),
                            "ax": int(parts[6]),
                            "ay": int(parts[7]),
                            "az": int(parts[8]),
                        }
                except ValueError:
                    pass

real = st.session_state.last_real
motion_intensity = min(1.0, np.sqrt(real["ax"]**2 + real["ay"]**2 + (real["az"]-16000)**2) / 20000)
hr_dev = abs(real["hr"] - st.session_state.baseline_hr)

# Signal quality placeholder: treat as good if HR is in plausible range
signal_quality = 0.9 if 40 <= real["hr"] <= 180 else 0.4

feature_row = pd.DataFrame([{
    "hr": real["hr"],
    "hr_dev_from_baseline": hr_dev,
    "spo2": sim_spo2,
    "signal_quality": signal_quality,
    "motion_intensity": motion_intensity,
    "ecg_abnormality": sim_ecg_abnormality,
}])[features]

prediction = model.predict(feature_row)[0]
probabilities = model.predict_proba(feature_row)[0]
predicted_label = LABELS[prediction]
confidence = probabilities[prediction] * 100

# ============================================================
# SECTION 1: LIVE VITALS
# ============================================================
st.header("Live Vitals")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Heart Rate (real)", f"{real['hr']:.1f} bpm")
c2.metric("SpO2 (simulated)", f"{sim_spo2:.1f} %")
c3.metric("ECG Abnormality (simulated)", f"{sim_ecg_abnormality:.2f}")
c4.metric("Motion Intensity (real)", f"{motion_intensity:.2f}")
c5.metric("Signal Quality", f"{signal_quality:.2f}")

# ============================================================
# SECTION 2: AI CARDIAC ASSESSMENT
# ============================================================
st.header("AI Cardiac Assessment")
color = LABEL_COLORS[predicted_label]
st.markdown(f"### Prediction: :{color}[{predicted_label}]")
st.progress(confidence / 100)
st.write(f"Confidence: **{confidence:.1f}%**")

prob_df = pd.DataFrame({"Class": LABELS, "Probability": probabilities})
st.bar_chart(prob_df.set_index("Class"))

st.caption("This is an AI risk-pattern classification from an academic prototype, NOT a medical diagnosis.")

# ============================================================
# SECTION 3: WHY DID THE AI FLAG THIS?
# ============================================================
st.header("Why Did the AI Flag This?")
st.caption("SHAP and LIME are explainability methods for the AI model's output — not diagnostic tools.")

shap_values = shap_explainer.shap_values(feature_row)
if isinstance(shap_values, list):
    class_shap = shap_values[prediction][0]
else:
    class_shap = shap_values[0, :, prediction]

contributions = list(zip(features, class_shap, feature_row.iloc[0].values))
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

st.subheader("SHAP Feature Contributions")
shap_df = pd.DataFrame({
    "Feature": [c[0] for c in contributions],
    "Contribution": [c[1] for c in contributions],
})
st.bar_chart(shap_df.set_index("Feature"))

for feat, shap_val, actual_val in contributions:
    direction = "increased" if shap_val > 0 else "decreased"
    st.write(f"- **{feat}** (value={actual_val:.2f}) {direction} the risk score by {abs(shap_val):.4f}")

st.subheader("LIME Local Explanation")
lime_exp = lime_explainer.explain_instance(
    data_row=feature_row.iloc[0].values,
    predict_fn=model.predict_proba,
    num_features=len(features),
    top_labels=1,
)
for feature_desc, weight in lime_exp.as_list(label=prediction):
    direction = "supports" if weight > 0 else "opposes"
    st.write(f"- {feature_desc} — **{direction}** this prediction (weight={weight:.4f})")

# ============================================================
# SECTION 4: PATIENT RISK FACTORS
# ============================================================
st.header("Patient Risk Factors")
st.caption("Static profile information. Does NOT imply that any single factor caused the current reading.")

risk_factors = {
    "Smoking / Tobacco use": smoking,
    "Hypertension": hypertension,
    "Diabetes": diabetes,
    "High cholesterol": high_cholesterol,
    "Obesity / Overweight": obesity,
    "Physical inactivity": inactivity,
    "Family history": family_history,
}
present = [k for k, v in risk_factors.items() if v]
st.write(f"**Age:** {age}")
if present:
    st.write("**Reported risk factors:** " + ", ".join(present))
else:
    st.write("**Reported risk factors:** None selected")
if other_history.strip():
    st.write(f"**Other history:** {other_history}")

# ============================================================
# SECTION 5: POSSIBLE UNDERLYING MECHANISMS
# ============================================================
st.header("Possible Underlying Mechanisms")
st.caption("The system cannot determine the biological cause of a cardiac event from these sensors alone. "
           "These are established medical mechanisms shown for educational context only.")

mechanisms = [
    "Coronary artery disease / plaque buildup",
    "Plaque rupture and clot formation",
    "Coronary artery spasm",
    "Coronary embolism",
    "Spontaneous coronary artery dissection (SCAD)",
]
for m in mechanisms:
    st.write(f"- {m}")

time.sleep(0.5)
st.rerun()
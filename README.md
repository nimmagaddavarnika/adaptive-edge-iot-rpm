
## Hardware

| Component | Role |
|---|---|
| ESP32 Dev Module | Edge processor |
| MAX30102 | PPG, heart rate, (SpO2 planned) |
| MPU6050 | 6-axis motion/activity context |
| AD8232 | Single-lead ECG |
| Active buzzer | Physical alerting |

## Explainability

Every AI prediction is broken into five layers, kept explicitly separate
so the system never conflates measurement, prediction, and interpretation:

1. **Live Vitals** — what the sensors actually measured
2. **AI Cardiac Assessment** — prediction + confidence
3. **Why did the AI flag this?** — SHAP feature contributions + LIME local explanation
4. **Patient Risk Factors** — static profile info (age, smoking, etc.) shown
   for context only, never implied as causal
5. **Possible Underlying Mechanisms** — established medical background,
   explicitly labeled as non-diagnostic

## Engineering decisions & known limitations (documented honestly)

- Instant BPM readings are filtered against a physiological plausibility
  range (20–255 BPM) to reject transient beat-detection glitches
- AD8232, wired on a breadboard, showed susceptibility to ~50Hz ambient
  noise. A moving-average filter (window=8) was applied at the edge.
  Clinical-grade fidelity would require analog shielding / PCB integration.
- The cardiac risk classifier is trained on rule-based synthetic data,
  explicitly documented as such — this validates the explainability
  pipeline and system architecture, not real-world diagnostic accuracy.
- HRV (RMSSD) is derived from single-channel PPG beat detection rather
  than ECG R-peaks, which is a simplification worth noting.

## Tech stack

`C++ (Arduino/ESP32)` · `Python` · `Streamlit` · `scikit-learn` · `SHAP` · `LIME` · `pandas`

## Running it

```bash
# 1. Flash firmware/combined_sensor.ino to an ESP32 (Arduino IDE + ESP32 board package)

# 2. Set up Python environment
pip install -r requirements.txt

# 3. Train the cardiac risk model
python train_cardiac_risk_model.py

# 4. Run the live sensor dashboard
streamlit run dashboard.py

# 5. Run the explainable cardiac risk dashboard (separate port)
streamlit run cardiac_risk_dashboard.py --server.port 8502
```

## Roadmap

- [ ] Real ECG electrode integration feeding live cardiac risk predictions
- [ ] Waveform-level deep learning (1D-CNN) instead of hand-engineered features
- [ ] Per-patient personalized baseline adaptation
- [ ] Experimental evaluation: fusion vs. single-signal false alert rates
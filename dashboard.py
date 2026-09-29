import streamlit as st
import serial
import time
import pandas as pd
from collections import deque

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
MAX_POINTS = 200

st.set_page_config(page_title="Adaptive Edge-IoT RPM", layout="wide")

st.title("Adaptive Edge-IoT Remote Patient Monitoring")
st.caption("Multimodal Edge Monitoring Prototype — Sensors: MAX30102 + MPU6050 + AD8232 ECG | Processor: ESP32")
st.caption(f"Serial Port: {PORT} | Baud Rate: {BAUD}")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
except Exception as e:
    st.error(f"Could not open serial port: {e}")
    st.stop()

status_placeholder = st.empty()
metrics_placeholder = st.empty()
ecg_status_placeholder = st.empty()
hrv_status_placeholder = st.empty()

st.subheader("MAX30102 IR/PPG Waveform (AC component, DC offset removed)")
ppg_placeholder = st.empty()
st.subheader("Heart Rate Trend (Average BPM)")
bpm_placeholder = st.empty()
st.subheader("Heart Rate Variability Trend (RMSSD, ms)")
hrv_placeholder = st.empty()
st.subheader("AD8232 ECG Waveform (smoothed)")
ecg_placeholder = st.empty()
st.subheader("MPU6050 Motion Data (Acceleration)")
motion_placeholder = st.empty()
st.subheader("Recent Sensor Data")
recent_placeholder = st.empty()

timestamps = deque(maxlen=MAX_POINTS)
ir_ac = deque(maxlen=MAX_POINTS)
bpm_values = deque(maxlen=MAX_POINTS)
hrv_values = deque(maxlen=MAX_POINTS)
accel_x = deque(maxlen=MAX_POINTS)
accel_y = deque(maxlen=MAX_POINTS)
accel_z = deque(maxlen=MAX_POINTS)
ecg_smoothed_vals = deque(maxlen=MAX_POINTS)

record_count = 0
last_row = None
ir_baseline = None

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if line.startswith("DATA,"):
        parts = line.split(",")
        if len(parts) == 17:
            try:
                row = {
                    "timestamp": int(parts[1]),
                    "finger": int(parts[2]),
                    "ir": int(parts[3]),
                    "bpm": float(parts[4]),
                    "avg_bpm": int(parts[5]),
                    "ax": int(parts[6]),
                    "ay": int(parts[7]),
                    "az": int(parts[8]),
                    "gx": int(parts[9]),
                    "gy": int(parts[10]),
                    "gz": int(parts[11]),
                    "ecg_leads_off": int(parts[12]),
                    "ecg_raw": int(parts[13]),
                    "ecg_smoothed": int(parts[14]),
                    "buzzer": int(parts[15]),
                    "hrv": float(parts[16]),
                }
            except ValueError:
                row = None

            if row:
                record_count += 1
                last_row = row

                if ir_baseline is None:
                    ir_baseline = row["ir"]
                else:
                    ir_baseline = 0.95 * ir_baseline + 0.05 * row["ir"]

                ac_component = row["ir"] - ir_baseline

                timestamps.append(row["timestamp"])
                ir_ac.append(ac_component)
                bpm_values.append(row["avg_bpm"])
                if row["hrv"] > 0:
                    hrv_values.append(row["hrv"])
                accel_x.append(row["ax"])
                accel_y.append(row["ay"])
                accel_z.append(row["az"])
                ecg_smoothed_vals.append(row["ecg_smoothed"])

    if last_row and record_count % 15 == 0:
        status_placeholder.success(f"Connected — receiving live data ({record_count} records)")

        with metrics_placeholder.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Finger Status", "DETECTED" if last_row["finger"] == 1 else "NOT DETECTED")
            col2.metric("Current BPM", f"{last_row['bpm']:.1f}" if last_row["finger"] == 1 else "--")
            col3.metric("Average BPM", last_row["avg_bpm"] if last_row["finger"] == 1 else "--")
            col4.metric("IR Value", last_row["ir"])
            col5.metric("DATA Records", record_count)

        with ecg_status_placeholder.container():
            ecol1, ecol2, ecol3 = st.columns(3)
            ecg_status = "NO CONTACT" if last_row["ecg_leads_off"] == 1 else "CONTACT OK"
            ecol1.metric("ECG Lead Status", ecg_status)
            ecol2.metric("ECG Raw", last_row["ecg_raw"] if last_row["ecg_leads_off"] == 0 else "--")
            ecol3.metric("Buzzer Alert", "ON" if last_row["buzzer"] == 1 else "OFF")

        with hrv_status_placeholder.container():
            hcol1, hcol2 = st.columns(2)
            hrv_display = f"{last_row['hrv']:.1f} ms" if last_row["hrv"] > 0 else "Collecting..."
            hcol1.metric("HRV (RMSSD)", hrv_display)
            if last_row["hrv"] > 0:
                if last_row["hrv"] < 20:
                    interp = "Low variability"
                elif last_row["hrv"] > 100:
                    interp = "High variability"
                else:
                    interp = "Typical range"
            else:
                interp = "Need more beats"
            hcol2.metric("HRV Interpretation", interp)

        if len(ir_ac) > 1:
            ppg_placeholder.line_chart(pd.DataFrame({"IR (AC)": list(ir_ac)}))

        if len(bpm_values) > 1:
            bpm_placeholder.line_chart(pd.DataFrame({"Avg BPM": list(bpm_values)}))

        if len(hrv_values) > 1:
            hrv_placeholder.line_chart(pd.DataFrame({"HRV (RMSSD ms)": list(hrv_values)}))
        else:
            hrv_placeholder.info("Waiting for enough beats to compute HRV...")

        if len(ecg_smoothed_vals) > 1:
            ecg_placeholder.line_chart(pd.DataFrame({"ECG (smoothed)": list(ecg_smoothed_vals)}))

        if len(accel_x) > 1:
            motion_placeholder.line_chart(pd.DataFrame({
                "ax": list(accel_x), "ay": list(accel_y), "az": list(accel_z)
            }))

        recent_placeholder.json(last_row)
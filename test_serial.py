import serial
import time

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

print(f"Opening {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Waiting for one DATA line to parse (17-field format with HRV)...")
print("-" * 50)

parsed_ok = False
start_time = time.time()

while time.time() - start_time < 20 and not parsed_ok:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if line.startswith("DATA,"):
        parts = line.split(",")

        if len(parts) != 17:
            print(f"Unexpected field count: {len(parts)} -> {line}")
            continue

        parsed = {
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

        print("RAW LINE:")
        print(line)
        print()
        print("PARSED FIELDS:")
        for key, value in parsed.items():
            print(f"  {key:15s} = {value}")

        parsed_ok = True

ser.close()

print("-" * 50)
if parsed_ok:
    print("PARSING TEST: SUCCESS (17-field format with HRV)")
else:
    print("PARSING TEST: FAILED - no DATA line parsed")
print("PORT CLOSED")
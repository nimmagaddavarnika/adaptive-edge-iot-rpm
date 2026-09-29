#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <MPU6050.h>

// ---------- MAX30102 ----------
MAX30105 particleSensor;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

// ---------- HRV ----------
const byte HRV_WINDOW = 10;
long beatIntervals[HRV_WINDOW];
byte hrvIndex = 0;
bool hrvBufferFilled = false;
float hrvRMSSD = 0;

// ---------- MPU6050 ----------
MPU6050 mpu;

// ---------- AD8232 ECG ----------
#define ECG_OUTPUT 34
#define LO_MINUS   32
#define LO_PLUS    33
const int FILTER_SIZE = 8;
int ecgBuffer[FILTER_SIZE];
int bufferIndex = 0;
long bufferSum = 0;
bool bufferFilled = false;

// ---------- Buzzer ----------
#define BUZZER_PIN 25

int smoothECG(int newValue) {
  bufferSum -= ecgBuffer[bufferIndex];
  ecgBuffer[bufferIndex] = newValue;
  bufferSum += newValue;
  bufferIndex = (bufferIndex + 1) % FILTER_SIZE;
  if (bufferIndex == 0) bufferFilled = true;
  return bufferSum / (bufferFilled ? FILTER_SIZE : max(bufferIndex, 1));
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(21, 22);

  // MAX30102
  Serial.println("Starting MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found. Check wiring.");
    while (1);
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 detected.");

  // MPU6050
  Serial.println("Starting MPU6050...");
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 not found. Check wiring.");
    while (1);
  }
  Serial.println("MPU6050 detected.");

  // AD8232
  pinMode(LO_MINUS, INPUT);
  pinMode(LO_PLUS, INPUT);
  for (int i = 0; i < FILTER_SIZE; i++) ecgBuffer[i] = 0;
  Serial.println("AD8232 ECG ready.");

  // Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("Buzzer ready.");

  Serial.println("DATA,timestamp,finger,ir,bpm,avg_bpm,ax,ay,az,gx,gy,gz,ecgLeadsOff,ecgRaw,ecgSmoothed,buzzer,hrv");
}

void loop() {
  // ---- MAX30102 ----
  long irValue = particleSensor.getIR();
  bool fingerDetected = (irValue > 50000);

  if (fingerDetected) {
    if (checkForBeat(irValue)) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      float instantBpm = 60 / (delta / 1000.0);

      if (instantBpm < 255 && instantBpm > 20) {
        beatsPerMinute = instantBpm;
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
        beatAvg /= RATE_SIZE;

        // ---- HRV: record beat-to-beat interval ----
        beatIntervals[hrvIndex] = delta;
        hrvIndex = (hrvIndex + 1) % HRV_WINDOW;
        if (hrvIndex == 0) hrvBufferFilled = true;

        // ---- Calculate RMSSD (standard HRV metric) ----
        if (hrvBufferFilled) {
          float sumSquaredDiffs = 0;
          for (byte i = 0; i < HRV_WINDOW - 1; i++) {
            float diff = beatIntervals[i + 1] - beatIntervals[i];
            sumSquaredDiffs += diff * diff;
          }
          hrvRMSSD = sqrt(sumSquaredDiffs / (HRV_WINDOW - 1));
        }
      }
    }
  }

  // ---- MPU6050 ----
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // ---- AD8232 ----
  bool ecgLeadsOff = (digitalRead(LO_MINUS) == HIGH) || (digitalRead(LO_PLUS) == HIGH);
  int ecgRaw = analogRead(ECG_OUTPUT);
  int ecgSmoothed = smoothECG(ecgRaw);

  // ---- Buzzer alert logic ----
  bool alert = false;
  if (fingerDetected && (beatsPerMinute > 120 || beatsPerMinute < 45)) {
    alert = true;
  }
  digitalWrite(BUZZER_PIN, alert ? HIGH : LOW);

  // ---- Send DATA line ----
  Serial.print("DATA,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(fingerDetected ? 1 : 0);
  Serial.print(",");
  Serial.print(irValue);
  Serial.print(",");
  Serial.print(fingerDetected ? beatsPerMinute : 0);
  Serial.print(",");
  Serial.print(beatAvg);
  Serial.print(",");
  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print(",");
  Serial.print(gx);
  Serial.print(",");
  Serial.print(gy);
  Serial.print(",");
  Serial.print(gz);
  Serial.print(",");
  Serial.print(ecgLeadsOff ? 1 : 0);
  Serial.print(",");
  Serial.print(ecgRaw);
  Serial.print(",");
  Serial.print(ecgSmoothed);
  Serial.print(",");
  Serial.print(alert ? 1 : 0);
  Serial.print(",");
  Serial.println(hrvRMSSD, 2);

  delay(20);
}
import joblib
import pandas as pd
import shap
import numpy as np

MODEL_PATH = "models/cardiac_risk_model.joblib"

print("Loading model...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

print(f"Features expected: {features}")
print()

# --- Simulate ONE example patient reading (a plausible high-risk case) ---
example = pd.DataFrame([{
    "hr": 118,
    "hr_dev_from_baseline": 42,
    "spo2": 91.5,
    "signal_quality": 0.85,
    "motion_intensity": 0.15,
    "ecg_abnormality": 0.68,
}])[features]

print("Example input:")
print(example)
print()

prediction = model.predict(example)[0]
probabilities = model.predict_proba(example)[0]

labels = ["NORMAL", "CARDIAC ANOMALY", "HIGH-RISK CARDIAC PATTERN"]
print(f"PREDICTION: {labels[prediction]}")
print(f"CONFIDENCE: {probabilities[prediction]*100:.1f}%")
print(f"All class probabilities: {dict(zip(labels, np.round(probabilities, 3)))}")
print()

print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(example)

# shap_values shape: (n_classes, n_samples, n_features) in recent SHAP versions,
# or a list of arrays per class in older versions. Handle both.
if isinstance(shap_values, list):
    class_shap = shap_values[prediction][0]
else:
    class_shap = shap_values[0, :, prediction]

print()
print("=" * 60)
print(f"WHY DID THE AI FLAG THIS AS: {labels[prediction]}?")
print("=" * 60)

contributions = list(zip(features, class_shap, example.iloc[0].values))
contributions.sort(key=lambda x: abs(x[1]), reverse=True)

for feat, shap_val, actual_val in contributions:
    direction = "INCREASED" if shap_val > 0 else "DECREASED"
    print(f"  {feat:22s} value={actual_val:8.2f}  ->  {direction} risk score by {abs(shap_val):.4f}")

print()
print("TEST FINISHED")

print()
print("=" * 60)
print("LIME LOCAL EXPLANATION (same example)")
print("=" * 60)

from lime.lime_tabular import LimeTabularExplainer

# LIME needs a background training-like distribution to sample around the instance.
# We reuse the same generation logic conceptually by building a small reference set
# from randomized plausible values, since we don't keep the training set in memory here.
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
    class_names=labels,
    mode="classification",
    discretize_continuous=True,
)

lime_exp = lime_explainer.explain_instance(
    data_row=example.iloc[0].values,
    predict_fn=model.predict_proba,
    num_features=len(features),
    top_labels=1,
)

print(f"LIME explanation for predicted class: {labels[prediction]}")
print()
for feature_desc, weight in lime_exp.as_list(label=prediction):
    direction = "supports" if weight > 0 else "opposes"
    print(f"  {feature_desc:35s} {direction} this prediction (weight={weight:.4f})")

print()
print("LIME TEST FINISHED")
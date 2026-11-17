
from pathlib import Path

import joblib
import pandas as pd


# Locate the ML folder
ML_DIR = Path(__file__).resolve().parent

# Load the trained model and preprocessor
model = joblib.load(ML_DIR / "roadguard_rf_model.pkl")
preprocessor = joblib.load(ML_DIR / "roadguard_preprocessor.pkl")

# Original target labels
TARGET_LABELS = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

# Expected 18 input features
EXPECTED_FEATURES = list(preprocessor.feature_names_in_)


def predict_severity(input_data):

    if not isinstance(input_data, dict):
        raise TypeError("Input must be a dictionary")

    # Check for missing features
    missing = set(EXPECTED_FEATURES) - set(input_data.keys())

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    # Arrange features in the correct order
    input_df = pd.DataFrame([input_data])[EXPECTED_FEATURES]

    # Apply saved preprocessing
    processed_input = preprocessor.transform(input_df)

    # Predict severity
    prediction = int(model.predict(processed_input)[0])

    return TARGET_LABELS[prediction]

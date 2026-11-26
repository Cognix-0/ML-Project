"""Predict the appliance type from raw PLAID waveform CSVs with the Logistic Regression model.

Usage:  python predict_lr.py submetered_new/1.csv [more.csv ...]
CSV format: column 0 = current, column 1 = voltage, no header (same as PLAID submetered).
Needs model_lr.joblib, made by 08_LogisticRegression.ipynb.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
from plaid_features import load_wave, extract_features

MODEL_FILE = Path(__file__).parent / "model_lr.joblib"


def predict_csv(path, bundle):
    i, v = load_wave(path)                              # 1. raw CSV -> current and voltage arrays
    feats = extract_features(i, v)                      # 2. same feature code as training (plaid_features.py)
    X = pd.DataFrame([feats])[bundle["features"]]       # 3. one row, the 20 model features in training order
    model = bundle["pipeline"]
    probs = model.predict_proba(X)[0]                   # 4. pipeline: (imputer) -> Logistic Regression
    best = probs.argmax()
    warnings = [name for name in ("stable_found", "window_ok") if not feats[name]]
    return {"type": model.classes_[best], "confidence": round(float(probs[best]), 2), "warnings": warnings}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if not MODEL_FILE.exists():
        sys.exit(f"{MODEL_FILE.name} not found: run 08_LogisticRegression.ipynb first")
    bundle = joblib.load(MODEL_FILE)
    for path in sys.argv[1:]:
        print(path, predict_csv(path, bundle))

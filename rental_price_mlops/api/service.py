from pathlib import Path
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from rental_price_mlops.config import PROJ_ROOT

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

MODEL_NAME = "RandomForestRegressor"
MODEL_VERSION = "baseline-v1"

FEATURES_EXPECTED = [
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "days_since_last_review",
    "has_last_review",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def make_feature_frame(payload: dict) -> pd.DataFrame:
    row = {feature: payload[feature] for feature in FEATURES_EXPECTED}
    return pd.DataFrame([row])


def predict(model, payload: dict) -> dict:
    X = make_feature_frame(payload)
    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))

    return {
        "predicted_log_price": pred_log,
        "predicted_price": pred_price,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


def read_latest_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrain_model() -> tuple[str, str]:
    cmd = [sys.executable, "-m", "rental_price_mlops.modeling.train"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return "success", "Model retraining completed successfully."

    return "error", result.stderr[-1000:] if result.stderr else "Retraining failed."
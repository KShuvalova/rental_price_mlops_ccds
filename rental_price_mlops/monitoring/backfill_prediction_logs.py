import argparse
import json
import pickle

import numpy as np
import pandas as pd

from rental_price_mlops.api.service import FEATURES_EXPECTED, MODEL_NAME, MODEL_VERSION
from rental_price_mlops.api.storage import utc_now_iso
from rental_price_mlops.config import PROJ_ROOT

DATA_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"
LOGS_DIR = PROJ_ROOT / "logs"

MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
TEST_PATH = DATA_DIR / "test.parquet"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def to_python(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(TEST_PATH)
    n_rows = min(args.rows, len(df))
    sampled = df.sample(n=n_rows, random_state=42).reset_index(drop=True)

    if args.replace:
        with open(PREDICTIONS_LOG, "w", encoding="utf-8"):
            pass

    rows_written = 0

    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            request_data = {feature: to_python(row.get(feature)) for feature in FEATURES_EXPECTED}

            X = pd.DataFrame([request_data])
            predicted_log_price = float(model.predict(X)[0])
            predicted_price = float(np.expm1(predicted_log_price))

            payload = {
                "timestamp": utc_now_iso(),
                "request_data": request_data,
                "predicted_log_price": predicted_log_price,
                "predicted_price": predicted_price,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "actual_price": to_python(row.get("price")),
                "actual_log_price": to_python(row.get("target")),
                "source": "test_dataset_backfill",
            }

            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows_written += 1

    print("Predictions appended to:", PREDICTIONS_LOG)
    print("Rows written:", rows_written)
    print("Mode:", "replace" if args.replace else "append")


if __name__ == "__main__":
    main()
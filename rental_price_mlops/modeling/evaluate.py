from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rental_price_mlops.config import PROJ_ROOT

DATA_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(MODELS_DIR / "baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    target_col = "target"

    drop_cols = [
        "target",
        "price",
        "id",
        "host_id",
        "last_review",
        "name",
        "host_name",
    ]

    feature_cols = [col for col in test_df.columns if col not in drop_cols]

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    test_pred_log = model.predict(X_test)

    metrics = {
        "mae_log": float(mean_absolute_error(y_test, test_pred_log)),
        "rmse_log": float(rmse(y_test, test_pred_log)),
        "r2_log": float(r2_score(y_test, test_pred_log)),
    }

    test_pred_price = np.expm1(test_pred_log)
    y_test_price = np.expm1(y_test)

    metrics["mae_price"] = float(mean_absolute_error(y_test_price, test_pred_price))
    metrics["rmse_price"] = float(rmse(y_test_price, test_pred_price))
    metrics["r2_price"] = float(r2_score(y_test_price, test_pred_price))

    predictions = test_df[["id", "price"]].copy()
    predictions["predicted_price"] = test_pred_price
    predictions["abs_error"] = np.abs(predictions["price"] - predictions["predicted_price"])

    predictions.to_parquet(REPORTS_DIR / "test_predictions.parquet", index=False)

    with open(REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Test evaluation finished.")
    print("Metrics saved to:", REPORTS_DIR / "test_metrics.json")
    print("Predictions saved to:", REPORTS_DIR / "test_predictions.parquet")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
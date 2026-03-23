from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rental_price_mlops.config import PROJ_ROOT

DATA_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")

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

    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].copy()

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_pred_log = pipeline.predict(X_val)

    metrics = {
        "mae_log": float(mean_absolute_error(y_val, val_pred_log)),
        "rmse_log": float(rmse(y_val, val_pred_log)),
        "r2_log": float(r2_score(y_val, val_pred_log)),
    }

    val_pred_price = np.expm1(val_pred_log)
    y_val_price = np.expm1(y_val)

    metrics["mae_price"] = float(mean_absolute_error(y_val_price, val_pred_price))
    metrics["rmse_price"] = float(rmse(y_val_price, val_pred_price))
    metrics["r2_price"] = float(r2_score(y_val_price, val_pred_price))

    with open(MODELS_DIR / "baseline_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open(REPORTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print("Features used:", len(feature_cols))
    print("Model saved to:", MODELS_DIR / "baseline_model.pkl")
    print("Metrics saved to:", REPORTS_DIR / "baseline_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    cat_features_idx = [X_train.columns.get_loc(col) for col in categorical_cols]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features_idx,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    val_pred_log = model.predict(X_val)

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

    model.save_model(str(MODELS_DIR / "catboost_model.cbm"))

    with open(REPORTS_DIR / "catboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("CatBoost training finished.")
    print("Features used:", len(feature_cols))
    print("Model saved to:", MODELS_DIR / "catboost_model.cbm")
    print("Metrics saved to:", REPORTS_DIR / "catboost_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
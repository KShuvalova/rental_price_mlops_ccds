import json

from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rental_price_mlops.config import PROJ_ROOT

DATA_DIR = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "rental-price-baselines"
REGISTERED_MODEL_NAME = "rental-price-catboost"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def round_metrics(metrics: dict) -> dict:
    return {k: round(float(v), 3) for k, v in metrics.items()}


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

    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
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

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="catboost_baseline") as run:
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
            "mae_log": mean_absolute_error(y_val, val_pred_log),
            "rmse_log": rmse(y_val, val_pred_log),
            "r2_log": r2_score(y_val, val_pred_log),
        }

        val_pred_price = np.expm1(val_pred_log)
        y_val_price = np.expm1(y_val)

        metrics["mae_price"] = mean_absolute_error(y_val_price, val_pred_price)
        metrics["rmse_price"] = rmse(y_val_price, val_pred_price)
        metrics["r2_price"] = r2_score(y_val_price, val_pred_price)

        metrics = round_metrics(metrics)

        model.save_model(str(MODELS_DIR / "catboost_model.cbm"))

        with open(REPORTS_DIR / "catboost_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        mlflow.log_param("model_type", "CatBoostRegressor")
        mlflow.log_param("target", "log1p(price)")
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("iterations", 1000)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("depth", 8)
        mlflow.log_param("early_stopping_rounds", 100)

        mlflow.set_tags(
            {
                "project": "rental_price_mlops",
                "stage": "baseline",
                "algorithm": "catboost",
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(REPORTS_DIR / "catboost_metrics.json"))

        sample_input = X_train.head(5)
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        print("CatBoost training finished.")
        print("Run ID:", run.info.run_id)
        print("Registered model version:", registered_model.version)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

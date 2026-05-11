import json
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
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

EXPERIMENT_NAME = "rental-price-baselines"
REGISTERED_MODEL_NAME = "rental-price-random-forest"


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

    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "string"]).columns.tolist()

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

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random_forest_baseline") as run:
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

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("target", "log1p(price)")
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("min_samples_leaf", 2)

        mlflow.set_tags(
            {
                "project": "rental_price_mlops",
                "stage": "baseline",
                "algorithm": "random_forest",
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(REPORTS_DIR / "baseline_metrics.json"))

        sample_input = X_train.head(5)
        sample_output = pipeline.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        print("Training finished.")
        print("Run ID:", run.info.run_id)
        print("Registered model version:", registered_model.version)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
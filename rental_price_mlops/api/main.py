from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from rental_price_mlops.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    RetrainResponse,
    LatestMetricsResponse,
)
from rental_price_mlops.api.storage import (
    append_prediction_log,
    read_prediction_logs,
    utc_now_iso,
)
from rental_price_mlops.api.service import (
    load_model,
    predict,
    read_latest_metrics,
    retrain_model,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_PATH,
    FEATURES_EXPECTED,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Rental Price MLOps API",
    description="Local FastAPI service for rental price prediction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_path=str(MODEL_PATH),
        target="log1p(price)",
        features_expected=FEATURES_EXPECTED,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PredictionRequest):
    try:
        result = predict(app.state.model, payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    append_prediction_log(
        {
            "timestamp": utc_now_iso(),
            "request_data": payload.model_dump(),
            **result,
        }
    )

    return PredictionResponse(**result)


@app.get("/predictions")
def get_predictions(limit: int = Query(default=20, ge=1, le=200)):
    return {"items": read_prediction_logs(limit=limit)}


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    status, message = retrain_model()

    if status == "success":
        app.state.model = load_model()
        return RetrainResponse(status=status, message=message)

    raise HTTPException(status_code=500, detail=message)


@app.get("/metrics/latest", response_model=LatestMetricsResponse)
def latest_metrics():
    return LatestMetricsResponse(
        source="reports/baseline_metrics.json",
        metrics=read_latest_metrics(),
    )
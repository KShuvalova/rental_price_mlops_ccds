from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import make_asgi_app

from rental_price_mlops.api.schemas import (
    LatestMetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    RetrainResponse,
)
from rental_price_mlops.api.service import (
    FEATURES_EXPECTED,
    MODEL_NAME,
    MODEL_PATH,
    load_model,
    predict,
    read_latest_metrics,
    retrain_model,
)
from rental_price_mlops.api.storage import (
    append_prediction_log,
    read_prediction_logs,
    utc_now_iso,
)
from rental_price_mlops.monitoring.metrics import (
    PREDICT_COUNT,
    RETRAIN_COUNT,
    MetricsMiddleware,
    update_drift_metrics,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    update_drift_metrics()
    yield


app = FastAPI(
    title="Rental Price MLOps API",
    description="Local FastAPI service for rental price prediction",
    version="0.1.0",
    lifespan=lifespan,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
DRIFT_REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "drift_report.json"

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MLFLOW_URL = "http://127.0.0.1:5000"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(MetricsMiddleware)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def render_monitoring(request: Request, monitoring, retrain_status=None, retrain_message=None):
    return templates.TemplateResponse(
        request=request,
        name="monitoring.html",
        context={
            "request": request,
            "title": "Monitoring",
            "monitoring": monitoring,
            "retrain_status": retrain_status,
            "retrain_message": retrain_message,
        },
    )

def render_experiments(
    request: Request,
    baseline_metrics,
    test_metrics,
    catboost_metrics,
    available_models,
    available_reports,
    primary_model,
):
    return templates.TemplateResponse(
        request=request,
        name="experiments.html",
        context={
            "request": request,
            "title": "Experiments",
            "baseline_metrics": baseline_metrics,
            "test_metrics": test_metrics,
            "catboost_metrics": catboost_metrics,
            "available_models": available_models,
            "available_reports": available_reports,
            "primary_model": primary_model,
            "mlflow_url": MLFLOW_URL,
        },
    )

def load_monitoring_report():
    if DRIFT_REPORT_PATH.exists():
        with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_json_file(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def pick_primary_model(baseline_metrics, catboost_metrics):
    if baseline_metrics and catboost_metrics:
        baseline_rmse = baseline_metrics.get("rmse_price")
        catboost_rmse = catboost_metrics.get("rmse_price")

        if baseline_rmse is not None and catboost_rmse is not None:
            return "CatBoost" if catboost_rmse < baseline_rmse else "Random Forest"

    if catboost_metrics:
        return "CatBoost"

    if baseline_metrics:
        return "Random Forest"

    return "Unknown"

def render_index(request: Request, prediction=None, error=None, form_data=None):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "title": "Rental Price UI",
            "prediction": prediction,
            "error": error,
            "form_data": form_data,
        },
    )

def render_predictions(request: Request, items):
    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={
            "request": request,
            "title": "Predictions",
            "items": items,
        },
    )

@app.get("/predictions-ui", response_class=HTMLResponse)
def predictions_ui(request: Request, limit: int = Query(default=20, ge=1, le=200)):
    items = read_prediction_logs(limit=limit)
    return render_predictions(request=request, items=items)


@app.get("/", response_class=HTMLResponse)
def ui_index(request: Request):
    return render_index(request=request)


@app.get("/monitoring-ui", response_class=HTMLResponse)
def monitoring_ui(request: Request):
    monitoring = load_monitoring_report()
    return render_monitoring(request=request, monitoring=monitoring)

@app.post("/retrain-ui", response_class=HTMLResponse)
def retrain_ui(request: Request):
    RETRAIN_COUNT.inc()

    status, message = retrain_model()

    if status == "success":
        app.state.model = load_model()
        update_drift_metrics()

    monitoring = load_monitoring_report()

    return render_monitoring(
        request=request,
        monitoring=monitoring,
        retrain_status=status,
        retrain_message=message,
    )

@app.get("/experiments-ui", response_class=HTMLResponse)
def experiments_ui(request: Request):
    baseline_metrics = load_json_file(REPORTS_DIR / "baseline_metrics.json")
    test_metrics = load_json_file(REPORTS_DIR / "test_metrics.json")
    catboost_metrics = load_json_file(REPORTS_DIR / "catboost_metrics.json")

    available_models = []
    if MODELS_DIR.exists():
        available_models = sorted([p.name for p in MODELS_DIR.iterdir() if p.is_file()])

    available_reports = []
    if REPORTS_DIR.exists():
        available_reports = sorted(
            [p.name for p in REPORTS_DIR.iterdir() if p.is_file() and p.suffix == ".json"]
        )

    primary_model = pick_primary_model(baseline_metrics, catboost_metrics)

    return render_experiments(
        request=request,
        baseline_metrics=baseline_metrics,
        test_metrics=test_metrics,
        catboost_metrics=catboost_metrics,
        available_models=available_models,
        available_reports=available_reports,
        primary_model=primary_model,
    )

@app.post("/", response_class=HTMLResponse)
def ui_predict(
    request: Request,
    neighbourhood_group: str = Form(...),
    neighbourhood: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    room_type: str = Form(...),
    minimum_nights: int = Form(...),
    number_of_reviews: int = Form(...),
    reviews_per_month: float = Form(...),
    calculated_host_listings_count: int = Form(...),
    availability_365: int = Form(...),
    days_since_last_review: float = Form(...),
    has_last_review: int = Form(...),
):
    form_data = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_last_review": days_since_last_review,
        "has_last_review": has_last_review,
    }

    PREDICT_COUNT.inc()

    try:
        result = predict(app.state.model, form_data)
    except Exception as e:
        return render_index(
            request=request,
            prediction=None,
            error=str(e),
            form_data=form_data,
        )

    append_prediction_log(
        {
            "timestamp": utc_now_iso(),
            "request_data": form_data,
            **result,
        }
    )

    return render_index(
        request=request,
        prediction=result,
        error=None,
        form_data=form_data,
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
    PREDICT_COUNT.inc()

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
    RETRAIN_COUNT.inc()

    status, message = retrain_model()

    if status == "success":
        app.state.model = load_model()
        update_drift_metrics()
        return RetrainResponse(status=status, message=message)

    raise HTTPException(status_code=500, detail=message)


@app.get("/metrics/latest", response_model=LatestMetricsResponse)
def latest_metrics():
    update_drift_metrics()
    return LatestMetricsResponse(
        source="reports/baseline_metrics.json",
        metrics=read_latest_metrics(),
    )
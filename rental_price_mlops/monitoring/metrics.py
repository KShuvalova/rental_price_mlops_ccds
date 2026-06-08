import json
import time

from prometheus_client import Counter, Gauge, Histogram

from rental_price_mlops.config import PROJ_ROOT

REPORT_PATH = PROJ_ROOT / "reports" / "drift_report.json"

REQUEST_COUNT = Counter(
    "rental_api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_ERRORS = Counter(
    "rental_api_errors_total",
    "Total number of HTTP 5xx errors",
    ["method", "path"],
)

REQUEST_LATENCY = Histogram(
    "rental_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

PREDICT_COUNT = Counter(
    "rental_api_predict_requests_total",
    "Total number of predict calls",
)

RETRAIN_COUNT = Counter(
    "rental_api_retrain_requests_total",
    "Total number of retrain calls",
)

DRIFTED_FEATURES_COUNT = Gauge(
    "rental_drifted_features_count",
    "Number of features flagged as drifted",
)

TARGET_DRIFT_FLAG = Gauge(
    "rental_target_drift_flag",
    "1 if target drift detected, else 0",
)

TARGET_DRIFT_LOG_FLAG = Gauge(
    "rental_target_drift_log_flag",
    "1 if target drift detected on log scale, else 0",
)

CONCEPT_DRIFT_FLAG = Gauge(
    "rental_concept_drift_flag",
    "1 if concept drift proxy detected, else 0",
)

CURRENT_ROWS = Gauge(
    "rental_current_rows",
    "Current monitoring dataset size",
)

CURRENT_MAE = Gauge(
    "rental_current_mae_price",
    "Current MAE on monitored observations",
)

CURRENT_RMSE = Gauge(
    "rental_current_rmse_price",
    "Current RMSE on monitored observations",
)

CURRENT_MAPE = Gauge(
    "rental_current_mape",
    "Current MAPE on monitored observations",
)


def update_drift_metrics():
    if not REPORT_PATH.exists():
        return

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    pred = report.get("prediction_summary", {})

    DRIFTED_FEATURES_COUNT.set(summary.get("drifted_features_count", 0))
    TARGET_DRIFT_FLAG.set(1 if summary.get("target_drift_flag") else 0)
    TARGET_DRIFT_LOG_FLAG.set(1 if summary.get("target_drift_log_flag") else 0)
    CONCEPT_DRIFT_FLAG.set(1 if summary.get("concept_drift_flag") else 0)
    CURRENT_ROWS.set(report.get("current_rows", 0))

    if pred.get("current_mae") is not None:
        CURRENT_MAE.set(pred["current_mae"])
    if pred.get("current_rmse") is not None:
        CURRENT_RMSE.set(pred["current_rmse"])
    if pred.get("current_mape") is not None:
        CURRENT_MAPE.set(pred["current_mape"])


class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "UNKNOWN")
        start = time.perf_counter()
        status_code_holder = {"value": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code_holder["value"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            status_code = str(status_code_holder["value"])

            REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)

            if status_code.startswith("5"):
                REQUEST_ERRORS.labels(method=method, path=path).inc()

            update_drift_metrics()
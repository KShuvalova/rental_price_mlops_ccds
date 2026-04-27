from pathlib import Path
import json
from datetime import datetime, timezone

from rental_price_mlops.config import PROJ_ROOT

LOGS_DIR = PROJ_ROOT / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def append_prediction_log(payload: dict) -> None:
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_prediction_logs(limit: int = 50) -> list[dict]:
    if not PREDICTIONS_LOG.exists():
        return []

    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    return rows[-limit:][::-1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
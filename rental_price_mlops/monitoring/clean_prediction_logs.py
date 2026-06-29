from collections import Counter
from datetime import datetime
import json
import shutil

from rental_price_mlops.api.service import FEATURES_EXPECTED
from rental_price_mlops.config import PROJ_ROOT

LOGS_DIR = PROJ_ROOT / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

VALID_ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room"}


def is_valid_request_data(request_data):
    if not isinstance(request_data, dict):
        return False, ["request_data_not_dict"]

    reasons = []

    for feature in FEATURES_EXPECTED:
        if feature not in request_data:
            reasons.append(f"missing_{feature}")

    neighbourhood_group = request_data.get("neighbourhood_group")
    neighbourhood = request_data.get("neighbourhood")
    room_type = request_data.get("room_type")
    latitude = request_data.get("latitude")
    longitude = request_data.get("longitude")
    minimum_nights = request_data.get("minimum_nights")
    availability_365 = request_data.get("availability_365")
    has_last_review = request_data.get("has_last_review")

    if neighbourhood_group in {None, "", "string"}:
        reasons.append("bad_neighbourhood_group")

    if neighbourhood in {None, "", "string"}:
        reasons.append("bad_neighbourhood")

    if room_type not in VALID_ROOM_TYPES:
        reasons.append("bad_room_type")

    try:
        if latitude is None or not (40.3 <= float(latitude) <= 41.1):
            reasons.append("bad_latitude")
    except Exception:
        reasons.append("bad_latitude")

    try:
        if longitude is None or not (-74.3 <= float(longitude) <= -73.5):
            reasons.append("bad_longitude")
    except Exception:
        reasons.append("bad_longitude")

    try:
        if int(minimum_nights) < 1:
            reasons.append("bad_minimum_nights")
    except Exception:
        reasons.append("bad_minimum_nights")

    try:
        if not (0 <= int(availability_365) <= 365):
            reasons.append("bad_availability_365")
    except Exception:
        reasons.append("bad_availability_365")

    try:
        if int(has_last_review) not in {0, 1}:
            reasons.append("bad_has_last_review")
    except Exception:
        reasons.append("bad_has_last_review")

    return len(reasons) == 0, reasons


def main():
    if not PREDICTIONS_LOG.exists():
        raise FileNotFoundError(f"Predictions log not found: {PREDICTIONS_LOG}")

    backup_name = f"predictions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    backup_path = LOGS_DIR / backup_name
    shutil.copyfile(PREDICTIONS_LOG, backup_path)

    kept_rows = []
    removed_count = 0
    reason_counter = Counter()

    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            request_data = item.get("request_data", {})

            is_valid, reasons = is_valid_request_data(request_data)
            if is_valid:
                kept_rows.append(item)
            else:
                removed_count += 1
                reason_counter.update(reasons)

    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        for item in kept_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Original log backed up to:", backup_path)
    print("Rows kept:", len(kept_rows))
    print("Rows removed:", removed_count)
    print("Removal reasons:", dict(reason_counter))


if __name__ == "__main__":
    main()
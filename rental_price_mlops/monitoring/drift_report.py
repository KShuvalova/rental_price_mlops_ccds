from html import escape
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from rental_price_mlops.api.service import FEATURES_EXPECTED
from rental_price_mlops.config import PROJ_ROOT

REFERENCE_PATH = PROJ_ROOT / "data" / "reference" / "reference.parquet"
CURRENT_PATH = PROJ_ROOT / "data" / "current" / "current_predictions.parquet"
REPORTS_DIR = PROJ_ROOT / "reports"
REPORT_JSON_PATH = REPORTS_DIR / "drift_report.json"
REPORT_HTML_PATH = REPORTS_DIR / "drift_report.html"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def round_float(value, digits=3):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), digits)


def numeric_drift_report(ref_series, cur_series):
    ref = pd.to_numeric(ref_series, errors="coerce").dropna()
    cur = pd.to_numeric(cur_series, errors="coerce").dropna()

    if len(ref) == 0 or len(cur) == 0:
        return {
            "type": "numeric",
            "reference_count": int(len(ref)),
            "current_count": int(len(cur)),
            "drift_score": None,
            "ks_pvalue": None,
            "reference_mean": None,
            "current_mean": None,
            "reference_std": None,
            "current_std": None,
            "drift_flag": None,
        }

    ks_stat, ks_pvalue = ks_2samp(ref, cur)

    return {
        "type": "numeric",
        "reference_count": int(len(ref)),
        "current_count": int(len(cur)),
        "drift_score": round_float(ks_stat),
        "ks_pvalue": round_float(ks_pvalue),
        "reference_mean": round_float(ref.mean()),
        "current_mean": round_float(cur.mean()),
        "reference_std": round_float(ref.std()),
        "current_std": round_float(cur.std()),
        "drift_flag": bool(ks_pvalue < 0.05),
    }


def categorical_drift_report(ref_series, cur_series):
    ref = ref_series.fillna("MISSING").astype(str)
    cur = cur_series.fillna("MISSING").astype(str)

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)

    all_categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))
    ref_aligned = ref_dist.reindex(all_categories, fill_value=0.0)
    cur_aligned = cur_dist.reindex(all_categories, fill_value=0.0)

    total_variation = 0.5 * np.abs(ref_aligned - cur_aligned).sum()

    top_ref = ref_dist.head(5).to_dict()
    top_cur = cur_dist.head(5).to_dict()

    return {
        "type": "categorical",
        "reference_count": int(len(ref)),
        "current_count": int(len(cur)),
        "drift_score": round_float(total_variation),
        "drift_flag": bool(total_variation > 0.1),
        "top_reference_categories": {k: round_float(v) for k, v in top_ref.items()},
        "top_current_categories": {k: round_float(v) for k, v in top_cur.items()},
    }


def load_baseline_metrics():
    if TEST_METRICS_PATH.exists():
        with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f), "test_metrics.json"

    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f), "baseline_metrics.json"

    return None, None


def compute_current_performance(cur_df):
    if "actual_price" not in cur_df.columns or "predicted_price" not in cur_df.columns:
        return None

    valid = cur_df.dropna(subset=["actual_price", "predicted_price"]).copy()
    if valid.empty:
        return None

    actual = pd.to_numeric(valid["actual_price"], errors="coerce")
    pred = pd.to_numeric(valid["predicted_price"], errors="coerce")
    valid_mask = actual.notna() & pred.notna()
    actual = actual[valid_mask]
    pred = pred[valid_mask]

    if len(actual) == 0:
        return None

    abs_error = (actual - pred).abs()
    squared_error = (actual - pred) ** 2
    safe_actual = actual.replace(0, np.nan)
    mape = (abs_error / safe_actual).dropna()

    return {
        "rows_used": int(len(actual)),
        "actual_price_mean": round_float(actual.mean()),
        "predicted_price_mean": round_float(pred.mean()),
        "current_mae": round_float(abs_error.mean()),
        "current_rmse": round_float(np.sqrt(squared_error.mean())),
        "current_mape": round_float(mape.mean()) if len(mape) > 0 else None,
    }


def compute_target_drift(ref_df, cur_df):
    if "price" not in ref_df.columns or "actual_price" not in cur_df.columns:
        return None

    cur_actual = pd.to_numeric(cur_df["actual_price"], errors="coerce").dropna()
    if len(cur_actual) == 0:
        return None

    return numeric_drift_report(ref_df["price"], cur_actual)


def compute_target_drift_log(ref_df, cur_df):
    if "target" not in ref_df.columns or "actual_log_price" not in cur_df.columns:
        return None

    cur_actual_log = pd.to_numeric(cur_df["actual_log_price"], errors="coerce").dropna()
    if len(cur_actual_log) == 0:
        return None

    return numeric_drift_report(ref_df["target"], cur_actual_log)


def compute_concept_drift_proxy(current_perf, baseline_metrics):
    if current_perf is None or baseline_metrics is None:
        return None

    baseline_mae = baseline_metrics.get("mae_price")
    baseline_rmse = baseline_metrics.get("rmse_price")

    if baseline_mae is None or baseline_rmse is None:
        return None

    current_mae = current_perf.get("current_mae")
    current_rmse = current_perf.get("current_rmse")

    if current_mae is None or current_rmse is None:
        return None

    mae_ratio = current_mae / baseline_mae if baseline_mae else None
    rmse_ratio = current_rmse / baseline_rmse if baseline_rmse else None

    drift_flag = False
    if mae_ratio is not None and mae_ratio > 1.2:
        drift_flag = True
    if rmse_ratio is not None and rmse_ratio > 1.2:
        drift_flag = True

    return {
        "baseline_source": "test_or_validation_metrics",
        "baseline_mae_price": round_float(baseline_mae),
        "baseline_rmse_price": round_float(baseline_rmse),
        "current_mae_price": round_float(current_mae),
        "current_rmse_price": round_float(current_rmse),
        "mae_ratio_vs_baseline": round_float(mae_ratio) if mae_ratio is not None else None,
        "rmse_ratio_vs_baseline": round_float(rmse_ratio) if rmse_ratio is not None else None,
        "drift_flag": drift_flag,
        "interpretation": (
            "Current error materially exceeds baseline error."
            if drift_flag
            else "Current error is close to baseline error."
        ),
    }


def build_html_report(report):
    feature_rows_html = []

    for feature, item in report["feature_reports"].items():
        drift_flag = item.get("drift_flag")
        drift_text = "YES" if drift_flag else "NO"
        drift_color = "#b91c1c" if drift_flag else "#15803d"
        drift_score = item.get("drift_score")

        if item["type"] == "numeric":
            details = (
                f"mean_ref={item.get('reference_mean')}, "
                f"mean_cur={item.get('current_mean')}, "
                f"pvalue={item.get('ks_pvalue')}"
            )
        else:
            details = (
                f"top_ref={item.get('top_reference_categories')}, "
                f"top_cur={item.get('top_current_categories')}"
            )

        feature_rows_html.append(
            f"""
            <tr>
                <td>{escape(feature)}</td>
                <td>{escape(item['type'])}</td>
                <td>{drift_score}</td>
                <td style="color:{drift_color}; font-weight:700;">{drift_text}</td>
                <td>{escape(str(details))}</td>
            </tr>
            """
        )

    target_drift = report.get("target_drift")
    target_drift_log = report.get("target_drift_log")
    concept_drift = report.get("concept_drift_proxy")
    summary = report["summary"]
    prediction_summary = report.get("prediction_summary", {})

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Drift Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                background: #fafafa;
                color: #111827;
            }}
            h1, h2 {{
                margin-bottom: 12px;
            }}
            .card {{
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
            }}
            th, td {{
                border: 1px solid #e5e7eb;
                padding: 10px;
                text-align: left;
                vertical-align: top;
            }}
            th {{
                background: #f3f4f6;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
            }}
        </style>
    </head>
    <body>
        <h1>Drift Report</h1>

        <div class="card">
            <p><strong>Reference rows:</strong> {report['reference_rows']}</p>
            <p><strong>Current rows:</strong> {report['current_rows']}</p>
            <p><strong>Drifted features:</strong> {summary['drifted_features_count']} / {summary['total_features_checked']}</p>
            <p><strong>Sample size warning:</strong> {summary['sample_size_warning']}</p>
            <p><strong>Recommendation:</strong> {escape(summary['recommended_action'])}</p>
        </div>

        <div class="card">
            <h2>Prediction summary</h2>
            <pre>{escape(json.dumps(prediction_summary, ensure_ascii=False, indent=2))}</pre>
        </div>

        <div class="card">
            <h2>Target drift</h2>
            <pre>{escape(json.dumps(target_drift, ensure_ascii=False, indent=2))}</pre>
            <h2>Target drift (log scale)</h2>
            <pre>{escape(json.dumps(target_drift_log, ensure_ascii=False, indent=2))}</pre>
        </div>

        <div class="card">
            <h2>Concept drift proxy</h2>
            <pre>{escape(json.dumps(concept_drift, ensure_ascii=False, indent=2))}</pre>
        </div>

        <div class="card">
            <h2>Feature drift details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Type</th>
                        <th>Drift score</th>
                        <th>Drift flag</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(feature_rows_html)}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Current dataset not found: {CURRENT_PATH}")

    ref_df = pd.read_parquet(REFERENCE_PATH)
    cur_df = pd.read_parquet(CURRENT_PATH)

    feature_columns = [col for col in FEATURES_EXPECTED if col in ref_df.columns and col in cur_df.columns]

    baseline_metrics, baseline_source = load_baseline_metrics()
    current_performance = compute_current_performance(cur_df)
    target_drift = compute_target_drift(ref_df, cur_df)
    target_drift_log = compute_target_drift_log(ref_df, cur_df)
    concept_drift_proxy = compute_concept_drift_proxy(current_performance, baseline_metrics)

    report = {
        "reference_path": str(REFERENCE_PATH),
        "current_path": str(CURRENT_PATH),
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_df)),
        "features_checked": feature_columns,
        "feature_reports": {},
        "prediction_summary": current_performance or {},
        "target_drift": target_drift,
        "target_drift_log": target_drift_log,
        "concept_drift_proxy": concept_drift_proxy,
        "baseline_metrics_source": baseline_source,
    }

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(cur_df[col]):
            report["feature_reports"][col] = numeric_drift_report(ref_df[col], cur_df[col])
        else:
            report["feature_reports"][col] = categorical_drift_report(ref_df[col], cur_df[col])

    drifted_features = []
    for feature, item in report["feature_reports"].items():
        if item.get("drift_flag") is True:
            drifted_features.append(
                {
                    "feature": feature,
                    "type": item["type"],
                    "drift_score": item.get("drift_score"),
                }
            )

    drifted_features = sorted(
        drifted_features,
        key=lambda x: (x["drift_score"] is None, -(x["drift_score"] or 0)),
    )

    sample_size_warning = bool(len(cur_df) < 30)

    report["summary"] = {
        "drifted_features_count": int(len(drifted_features)),
        "total_features_checked": int(len(feature_columns)),
        "sample_size_warning": sample_size_warning,
        "recommended_action": (
            "Collect at least 30-100 current predictions before treating drift flags as operational alerts."
            if sample_size_warning
            else "Sample size is sufficient for a first operational interpretation."
        ),
        "drifted_features": drifted_features,
        "target_drift_flag": target_drift.get("drift_flag") if target_drift else None,
        "target_drift_log_flag": target_drift_log.get("drift_flag") if target_drift_log else None,
        "concept_drift_flag": concept_drift_proxy.get("drift_flag") if concept_drift_proxy else None,
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html_report = build_html_report(report)
    with open(REPORT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_report)

    print("Drift JSON report saved to:", REPORT_JSON_PATH)
    print("Drift HTML report saved to:", REPORT_HTML_PATH)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""
Data and model drift detection using Evidently.
Report can be logged to MLflow or served for dashboard.
"""
from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """Compute Evidently data drift report (reference vs current)."""
    if not HAS_EVIDENTLY:
        return {"error": "evidently not installed", "drift_detected": False}
    if feature_cols is None:
        feature_cols = [c for c in reference.columns if reference[c].dtype in ("int64", "float64")][:20]
    ref = reference[feature_cols].fillna(0).head(5000)
    cur = current[feature_cols].fillna(0).head(5000)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()
    # Simplify for API: overall drift
    drift_detected = False
    if "metrics" in result:
        for m in result["metrics"]:
            if m.get("result", {}).get("dataset_drift"):
                drift_detected = True
                break
    return {"drift_detected": drift_detected, "report_summary": result}


def run_drift_check(
    reference_path: Path,
    current_path: Path,
    feature_cols: list[str],
    output_path: Path | None = None,
) -> dict:
    """Load reference and current from feature store paths; run drift; optionally save."""
    ref = pd.read_parquet(reference_path)
    cur = pd.read_parquet(current_path)
    result = compute_drift_report(ref, cur, feature_cols)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result

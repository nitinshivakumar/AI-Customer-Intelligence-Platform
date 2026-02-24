"""
Bias and fairness testing on age, gender, region using fairlearn metrics.
Log results to MLflow.
"""
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import mlflow

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.features.feature_store import load_features
from src.models.churn_model import load_churn_artifact, predict_proba

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        MetricFrame,
    )
    from sklearn.metrics import recall_score
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False


def run_bias_analysis(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model,
    sensitive_attrs: list[str],
) -> dict:
    """Compute fairness metrics per sensitive attribute."""
    X = df[feature_cols]
    y_true = df[target_col].astype(int)
    y_pred = (predict_proba(model, X, feature_cols) >= 0.5).astype(int)

    results = {}
    for attr in sensitive_attrs:
        if attr not in df.columns:
            continue
        sens = df[attr].astype(str)
        try:
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sens)
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sens)
            mf = MetricFrame(
                metrics={"recall": recall_score},
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sens,
            )
            recall_by_group = mf.by_group["recall"].to_dict() if "recall" in mf.by_group else {}
            results[attr] = {
                "demographic_parity_difference": float(dp_diff),
                "equalized_odds_difference": float(eo_diff),
                "recall_by_group": {str(k): float(v) for k, v in recall_by_group.items()},
            }
        except Exception as e:
            results[attr] = {"error": str(e)}
    return results


def main():
    if not HAS_FAIRLEARN:
        print("fairlearn not installed; skipping bias tests. pip install fairlearn")
        return

    df, meta = load_features("churn_training")
    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    artifact_dir = settings.project_root / "models" / "churn"
    model, loaded_features = load_churn_artifact(artifact_dir)
    if loaded_features != feature_cols:
        feature_cols = [f for f in feature_cols if f in loaded_features]

    # Use _sensitive string columns saved for fairness evaluation
    sensitive = ["gender_sensitive", "region_sensitive", "age_group_sensitive"]
    sensitive = [s for s in sensitive if s in df.columns]
    if not sensitive:
        sensitive = getattr(settings, "sensitive_attributes", ["age_group", "gender", "region"])
        sensitive = [s for s in sensitive if s in df.columns]

    results = run_bias_analysis(df, feature_cols, target_col, model, sensitive)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="bias_fairness") as run:
        for attr, vals in results.items():
            for k, v in vals.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"bias_{attr}_{k}", v)
                elif isinstance(v, dict):
                    for g, gv in v.items():
                        if isinstance(gv, (int, float)):
                            mlflow.log_metric(f"bias_{attr}_recall_{g}", gv)
        mlflow.log_dict(results, "bias_results.json")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

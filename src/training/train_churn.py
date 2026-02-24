"""
Train churn model (XGBoost default), log to MLflow, register artifact.
"""
from pathlib import Path
import sys
import argparse
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.features.feature_store import load_features
from src.models.churn_model import (
    get_feature_target,
    train_logistic,
    train_xgboost,
    train_lightgbm,
    save_churn_artifact,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic", "xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if args.experiment:
        mlflow.set_experiment(args.experiment)
    else:
        mlflow.set_experiment(settings.mlflow_experiment_name)

    df, meta = load_features("churn_training")
    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]
    X, y = get_feature_target(df, feature_cols, target_col)

    with mlflow.start_run(run_name=f"churn_{args.model}") as run:
        if args.model == "logistic":
            model, metrics = train_logistic(X, y)
            mlflow.sklearn.log_model(model, "model")
        elif args.model == "xgboost":
            model, metrics = train_xgboost(X, y)
            mlflow.xgboost.log_model(model, "model")
        else:
            model, metrics = train_lightgbm(X, y)
            mlflow.lightgbm.log_model(model, "model")

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("feature_count", len(feature_cols))

        artifact_dir = settings.project_root / "models" / "churn"
        save_churn_artifact(model, feature_cols, metrics, artifact_dir)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="churn_artifact")
        print(f"Run ID: {run.info.run_id}, AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()

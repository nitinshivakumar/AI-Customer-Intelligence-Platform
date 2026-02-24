"""
Train anomaly model (Isolation Forest), log to MLflow, save artifact.
"""
from pathlib import Path
import sys
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.features.feature_store import load_features
from src.models.anomaly_model import (
    train_isolation_forest,
    save_anomaly_artifact,
)


def main():
    df, meta = load_features("usage_anomaly")
    feature_cols = meta["feature_cols"]
    X = df[feature_cols].fillna(0)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="anomaly_isolation_forest") as run:
        model, scaler, cols = train_isolation_forest(X, feature_cols=feature_cols)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("feature_count", len(cols))

        artifact_dir = settings.project_root / "models" / "anomaly"
        save_anomaly_artifact(model, scaler, cols, artifact_dir)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="anomaly_artifact")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()

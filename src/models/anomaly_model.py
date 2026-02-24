"""
Usage anomaly detection: Isolation Forest (and optional autoencoder).
Returns anomaly score per sample.
"""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def train_isolation_forest(
    X: pd.DataFrame,
    feature_cols: list | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[IsolationForest, StandardScaler, list]:
    if feature_cols is None:
        feature_cols = [c for c in X.columns if X[c].dtype in ("int64", "float64")]
    X = X[feature_cols].copy()
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(X_scaled)
    return model, scaler, feature_cols


def anomaly_score(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_cols: list,
) -> np.ndarray:
    """Return anomaly score (higher = more anomalous). Uses decision_function (neg = anomaly)."""
    X = X[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    # decision_function: negative = anomaly; flip so positive = more anomalous
    raw = model.decision_function(X_scaled)
    return -raw  # higher value = more anomalous


def save_anomaly_artifact(
    model: IsolationForest,
    scaler: StandardScaler,
    feature_cols: list,
    artifact_path: Path,
):
    path = Path(artifact_path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path / "model.joblib")
    with open(path / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)


def load_anomaly_artifact(artifact_path: Path) -> tuple[IsolationForest, StandardScaler, list]:
    path = Path(artifact_path)
    data = joblib.load(path / "model.joblib")
    with open(path / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return data["model"], data["scaler"], feature_cols

"""
Churn prediction: train and predict with Logistic Regression, XGBoost, LightGBM.
Provides unified interface for MLflow and API.
"""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_feature_target(df: pd.DataFrame, feature_cols: list, target_col: str = "churn"):
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y


def train_logistic(X: pd.DataFrame, y: pd.Series, **kwargs) -> tuple[LogisticRegression, dict]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    model.fit(X, y)
    metrics = {
        "auc": roc_auc_score(y, y_pred_proba),
        "recall": recall_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "precision": precision_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "f1": f1_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
    }
    return model, metrics


def train_xgboost(X: pd.DataFrame, y: pd.Series, **kwargs) -> tuple[xgb.XGBClassifier, dict]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        **kwargs,
    )
    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    model.fit(X, y)
    metrics = {
        "auc": roc_auc_score(y, y_pred_proba),
        "recall": recall_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "precision": precision_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "f1": f1_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
    }
    return model, metrics


def train_lightgbm(X: pd.DataFrame, y: pd.Series, **kwargs) -> tuple[lgb.LGBMClassifier, dict]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        **kwargs,
    )
    y_pred_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    model.fit(X, y)
    metrics = {
        "auc": roc_auc_score(y, y_pred_proba),
        "recall": recall_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "precision": precision_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
        "f1": f1_score(y, (y_pred_proba >= 0.5).astype(int), zero_division=0),
    }
    return model, metrics


def train_best(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
) -> tuple[object, dict]:
    if model_type == "logistic":
        return train_logistic(X, y)
    if model_type == "xgboost":
        return train_xgboost(X, y)
    if model_type == "lightgbm":
        return train_lightgbm(X, y)
    raise ValueError(f"Unknown model_type: {model_type}")


def save_churn_artifact(
    model,
    feature_cols: list,
    metrics: dict,
    artifact_path: Path,
):
    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path / "model.joblib")
    with open(artifact_path / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(artifact_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_churn_artifact(artifact_path: Path) -> tuple[object, list]:
    artifact_path = Path(artifact_path)
    model = joblib.load(artifact_path / "model.joblib")
    with open(artifact_path / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def predict_proba(model, X: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = X[feature_cols] if isinstance(X, pd.DataFrame) else X
    return model.predict_proba(X)[:, 1]

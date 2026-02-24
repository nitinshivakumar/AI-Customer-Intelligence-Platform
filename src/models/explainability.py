"""
Explainability: SHAP for churn model (global and per-request).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_shap_explainer(model, X_background: pd.DataFrame, feature_names: list):
    """Create explainer (TreeExplainer for tree models, else KernelExplainer)."""
    X_bg = X_background[feature_names].fillna(0).head(100)
    try:
        explainer = shap.TreeExplainer(model, X_bg)
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_bg, feature_names=feature_names)
    return explainer


def explain_local(
    explainer,
    X: pd.DataFrame,
    feature_names: list,
    top_k: int = 10,
) -> list[dict]:
    """Per-row SHAP values; return list of {feature: contribution} for each row."""
    X = X[feature_names].fillna(0)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class for binary
    out = []
    for i in range(len(X)):
        sv = shap_values[i]
        order = np.argsort(np.abs(sv))[::-1][:top_k]
        out.append({
            feature_names[j]: float(sv[j])
            for j in order
        })
    return out


def explain_global(
    explainer,
    X: pd.DataFrame,
    feature_names: list,
) -> dict[str, float]:
    """Global feature importance (mean |SHAP|)."""
    X = X[feature_names].fillna(0)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}

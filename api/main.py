"""
FastAPI inference service: /predict, /explain, /anomaly, /insight, /health, /metrics.
"""
from pathlib import Path
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from api.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainResponse,
    AnomalyRequest,
    AnomalyResponse,
    InsightRequest,
    InsightResponse,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Request latency", ["endpoint"])
PREDICT_COUNT = Counter("churn_predictions_total", "Churn predictions")
ANOMALY_COUNT = Counter("anomaly_checks_total", "Anomaly checks")

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def load_models():
    """Load churn and anomaly models + explainer (lazy)."""
    from src.models.churn_model import load_churn_artifact, predict_proba
    from src.models.anomaly_model import load_anomaly_artifact
    from src.models.explainability import get_shap_explainer, explain_local
    import pandas as pd

    churn_path = ROOT / "models" / "churn"
    anomaly_path = ROOT / "models" / "anomaly"
    if not churn_path.exists() or not (churn_path / "model.joblib").exists():
        return None, None, None, None, None, None
    churn_model, churn_features = load_churn_artifact(churn_path)
    anomaly_model, anomaly_scaler, anomaly_features = load_anomaly_artifact(anomaly_path)

    # Build small background set for SHAP (from feature store if available)
    try:
        from src.features.feature_store import load_features
        df, _ = load_features("churn_training")
        X_bg = df[churn_features].fillna(0).head(50)
    except Exception:
        X_bg = pd.DataFrame(columns=churn_features)
    explainer = get_shap_explainer(churn_model, X_bg, churn_features) if len(X_bg) > 0 else None

    return churn_model, churn_features, anomaly_model, anomaly_scaler, anomaly_features, explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = None
    app.state.churn_features = None
    app.state.anomaly_model = None
    app.state.anomaly_scaler = None
    app.state.anomaly_features = None
    app.state.explainer = None
    try:
        churn_model, churn_features, anom_model, anom_scaler, anom_features, explainer = load_models()
        app.state.churn_model = churn_model
        app.state.churn_features = churn_features
        app.state.anomaly_model = anom_model
        app.state.anomaly_scaler = anom_scaler
        app.state.anomaly_features = anom_features
        app.state.explainer = explainer
        logger.info("Models loaded")
    except Exception as e:
        logger.warning("Models not loaded (run training first): %s", e)
    yield
    # shutdown
    app.state.churn_model = None
    app.state.explainer = None


app = FastAPI(title="AI Customer Intelligence API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Churn probability and binary label."""
    REQUEST_COUNT.labels(method="POST", endpoint="predict").inc()
    with REQUEST_LATENCY.labels(endpoint="predict").time():
        model = getattr(app.state, "churn_model", None)
        features = getattr(app.state, "churn_features", None)
        if model is None or features is None:
            raise HTTPException(status_code=503, detail="Churn model not loaded")
        import pandas as pd
        row = pd.DataFrame([req.features])
        row = row.reindex(columns=features).fillna(0)
        prob = float(model.predict_proba(row)[0, 1])
        label = 1 if prob >= 0.5 else 0
        PREDICT_COUNT.inc()
        return PredictResponse(churn_probability=prob, label=label)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    """Churn probability + local SHAP contributions."""
    REQUEST_COUNT.labels(method="POST", endpoint="explain").inc()
    with REQUEST_LATENCY.labels(endpoint="explain").time():
        model = getattr(app.state, "churn_model", None)
        features = getattr(app.state, "churn_features", None)
        explainer = getattr(app.state, "explainer", None)
        if model is None or features is None:
            raise HTTPException(status_code=503, detail="Churn model not loaded")
        import pandas as pd
        row = pd.DataFrame([req.features])
        row = row.reindex(columns=features).fillna(0)
        prob = float(model.predict_proba(row)[0, 1])
        if explainer is not None:
            from src.models.explainability import explain_local as _explain_local
            contribs = _explain_local(explainer, row, features, top_k=15)
            contributions = contribs[0] if contribs else {}
        else:
            contributions = {}
        return ExplainResponse(churn_probability=prob, contributions=contributions)


@app.post("/anomaly", response_model=AnomalyResponse)
def anomaly(req: AnomalyRequest):
    """Usage anomaly score (higher = more anomalous)."""
    REQUEST_COUNT.labels(method="POST", endpoint="anomaly").inc()
    with REQUEST_LATENCY.labels(endpoint="anomaly").time():
        model = getattr(app.state, "anomaly_model", None)
        scaler = getattr(app.state, "anomaly_scaler", None)
        features = getattr(app.state, "anomaly_features", None)
        if model is None or features is None:
            raise HTTPException(status_code=503, detail="Anomaly model not loaded")
        from src.models.anomaly_model import anomaly_score
        import pandas as pd
        row = pd.DataFrame([req.features])
        row = row.reindex(columns=features).fillna(0)
        score = float(anomaly_score(model, scaler, row, features)[0])
        is_anomaly = score > 0.5  # tunable threshold
        ANOMALY_COUNT.inc()
        return AnomalyResponse(anomaly_score=score, is_anomaly=is_anomaly)


@app.post("/insight", response_model=InsightResponse)
def insight(req: InsightRequest):
    """LLM-generated risk summary and retention strategy."""
    REQUEST_COUNT.labels(method="POST", endpoint="insight").inc()
    with REQUEST_LATENCY.labels(endpoint="insight").time():
        from src.models.llm_insight import generate_risk_insight
        result = generate_risk_insight(
            customer_id=req.customer_id or "unknown",
            churn_probability=req.churn_probability,
            profile=req.profile,
        )
        return InsightResponse(
            customer_id=result["customer_id"],
            risk_summary=result["risk_summary"],
            retention_strategy=result["retention_strategy"],
            model_used=result.get("model_used", ""),
        )



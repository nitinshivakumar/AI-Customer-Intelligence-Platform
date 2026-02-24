"""API endpoint tests (run after models are trained for full coverage)."""
import pytest
from fastapi.testclient import TestClient

# Import after setting path
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "api_requests_total" in r.text or "churn_predictions" in r.text or "# " in r.text


def test_predict_no_model():
    r = client.post("/predict", json={"features": {"recency_days": 30, "tx_frequency": 5}})
    # 503 if model not loaded, 400 if features missing
    assert r.status_code in (400, 503)


def test_insight_mock():
    r = client.post(
        "/insight",
        json={
            "customer_id": "C000001",
            "churn_probability": 0.7,
            "profile": {"recency_days": 60, "tx_frequency": 2},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "risk_summary" in data
    assert "retention_strategy" in data

"""
Streamlit admin dashboard: predictions, drift metrics, feature importance, confusion matrix.
"""
import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Customer Intelligence", layout="wide")
st.title("AI Customer Intelligence Platform")
st.caption("Churn prediction, anomaly detection, explainability, drift")

sidebar = st.sidebar
sidebar.header("Navigation")
page = sidebar.radio(
    "Page",
    ["Overview", "Churn Predictions", "Explainability", "Anomaly", "Drift & Metrics", "Model Performance"],
)

if page == "Overview":
    st.subheader("Overview")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        st.success("API is up." if r.ok else "API returned an error.")
    except Exception as e:
        st.warning("API not reachable. Start with: uvicorn api.main:app --port 8000")
    st.markdown("""
    - **Churn Predictions**: Run batch or single customer churn probability.
    - **Explainability**: View SHAP contributions and feature importance.
    - **Anomaly**: Check usage anomaly scores.
    - **Drift & Metrics**: Prometheus metrics and drift status.
    - **Model Performance**: Confusion matrix and metrics from training.
    """)

elif page == "Churn Predictions":
    st.subheader("Churn prediction")
    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("Recency (days)", 0, 500, 30)
        tx_freq = st.number_input("Transaction frequency", 0, 100, 5)
        total_amount = st.number_input("Total amount", 0.0, 100000.0, 500.0)
    with col2:
        session_count = st.number_input("Session count (90d)", 0, 500, 20)
        ticket_count = st.number_input("Support tickets", 0, 50, 0)
    if st.button("Predict"):
        features = {
            "recency_days": float(recency),
            "tx_frequency": float(tx_freq),
            "total_amount": total_amount,
            "session_count": float(session_count),
            "ticket_count": float(ticket_count),
        }
        # Add defaults for other required features if API needs full set
        try:
            r = requests.post(f"{API_BASE}/predict", json={"features": features}, timeout=5)
            if r.ok:
                d = r.json()
                st.metric("Churn probability", f"{d['churn_probability']:.2%}")
                st.metric("Label", "Churn" if d["label"] == 1 else "No churn")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

elif page == "Explainability":
    st.subheader("Explainability (SHAP)")
    st.info("Call /explain with same feature dict to see contributions.")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1)
        if r.ok:
            st.success("Use API POST /explain with features to get local SHAP values.")
    except Exception:
        pass
    # Show global feature importance from saved metrics if available
    metrics_path = ROOT / "models" / "churn" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        st.metric("Model AUC", m.get("auc", 0))
        st.metric("Recall", m.get("recall", 0))

elif page == "Anomaly":
    st.subheader("Usage anomaly")
    st.info("POST /anomaly with usage features to get anomaly score (higher = more anomalous).")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1)
        st.success("API ready for anomaly checks." if r.ok else "")
    except Exception:
        pass

elif page == "Drift & Metrics":
    st.subheader("Drift & metrics")
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=2)
        if r.ok:
            st.text_area("Prometheus metrics", r.text, height=300)
    except Exception as e:
        st.warning("Could not fetch metrics: " + str(e))
    st.subheader("Drift")
    drift_path = ROOT / "data" / "feature_store" / "drift_report.json"
    if drift_path.exists():
        with open(drift_path) as f:
            drift = json.load(f)
        st.json(drift)
    else:
        st.info("Run drift check (e.g. src.monitoring.drift) to populate drift report.")

elif page == "Model Performance":
    st.subheader("Model performance")
    metrics_path = ROOT / "models" / "churn" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{m.get('auc', 0):.4f}")
        c2.metric("Recall", f"{m.get('recall', 0):.4f}")
        c3.metric("Precision", f"{m.get('precision', 0):.4f}")
        c4.metric("F1", f"{m.get('f1', 0):.4f}")
        st.json(m)
    else:
        st.info("Train churn model first to see metrics.")
    st.subheader("Bias results")
    # MLflow or saved bias_results
    bias_path = ROOT / "mlruns"
    if bias_path.exists():
        st.caption("Bias metrics are in MLflow runs (bias_fairness).")

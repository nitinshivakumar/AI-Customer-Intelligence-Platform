# AI Customer Intelligence Platform

Production-grade ML platform for **customer churn prediction**, **risk insights (LLM)**, **usage anomaly detection**, with **explainability**, **bias testing**, and **MLOps** (Feature Store, MLflow, Docker, CI/CD, drift monitoring).

## Architecture

```
Raw Data → Data Pipeline → Feature Store → Model Training
                ↓
        Bias Testing + Evaluation
                ↓
        Model Registry (MLflow)
                ↓
        FastAPI Inference Service
                ↓
        Docker → Cloud (GCP/AWS/Railway)
                ↓
        Monitoring (Drift + Logs + Metrics)
```

## How to run (step-by-step)

**From the project root** (`ai-customer-intelligence-platform/`):

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Data pipeline (generates synthetic customers, ETL, features)
python -m src.data.generate_data
python -m src.data.etl_pipeline
python -m src.features.build_features

# 3. Train models (churn, anomaly, bias tests → MLflow)
python -m src.training.train_churn --model xgboost
python -m src.training.train_anomaly
python -m src.training.run_bias_tests   # optional; needs fairlearn

# 4. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
# → Open http://localhost:8000/docs for Swagger UI

# 5. (Optional) Start the Streamlit dashboard (in another terminal)
streamlit run dashboard/app.py
```

**Using Make:**

```bash
make install    # pip install
make data      # generate + ETL + features
make train     # churn + anomaly + bias
make api       # start API on port 8000
make dashboard # start Streamlit
```

**Quick run with fewer customers:**

```bash
SAMPLE_CUSTOMERS=500 python -m src.data.generate_data
# then ETL and features as above
```

---

## How it works (explanation)

**High-level flow**

1. **Data** — `generate_data` creates synthetic customers, transactions, sessions, support tickets, and churn labels.  
2. **ETL** — `etl_pipeline` loads that data into SQLite (or PostgreSQL via `DATABASE_URL`).  
3. **Features** — `build_features` computes RFM, rolling-window usage, and time aggregates, then writes to a file-based feature store under `data/feature_store/`.  
4. **Training** — `train_churn` (XGBoost/LogReg/LightGBM) and `train_anomaly` (Isolation Forest) train on those features; runs are logged to MLflow in `mlruns/`.  
5. **Bias tests** — `run_bias_tests` uses Fairlearn on age/gender/region and logs fairness metrics to MLflow.  
6. **API** — FastAPI loads the saved churn and anomaly models and serves `/predict`, `/explain`, `/anomaly`, `/insight`, plus `/health` and `/metrics`.  
7. **Dashboard** — Streamlit calls the API to show predictions, metrics, and model performance.

**What each endpoint does**

| Endpoint | Input | Output |
|----------|--------|--------|
| `POST /predict` | `{"features": {"recency_days": 30, "tx_frequency": 5, ...}}` | Churn probability and 0/1 label. |
| `POST /explain` | Same feature dict. | Churn probability + SHAP contributions per feature. |
| `POST /anomaly` | Usage feature dict. | Anomaly score (higher = more anomalous) and `is_anomaly`. |
| `POST /insight` | `{"customer_id", "churn_probability", "profile": {...}}` | LLM risk summary and retention strategy (OpenAI or mock). |
| `GET /health` | — | `{"status": "ok"}`. |
| `GET /metrics` | — | Prometheus metrics. |

**Try the API**

After starting the API, open **http://localhost:8000/docs**. From there you can call `/predict`, `/explain`, `/anomaly`, and `/insight` with sample JSON. The API fills missing features with 0, so you can send a minimal payload like `{"features": {"recency_days": 60, "tx_frequency": 2}}` for `/predict`.

---

## Quick Start (reference)

```bash
# Create venv and install
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. Generate sample data & run ETL
python -m src.data.generate_data
python -m src.data.etl_pipeline

# 2. Build features & train models
python -m src.features.build_features
python -m src.training.train_churn
python -m src.training.train_anomaly
python -m src.training.run_bias_tests

# 3. Start API (after training)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Dashboard
streamlit run dashboard/app.py
```

## Docker

```bash
docker build -t ai-customer-intelligence:latest .
docker run -p 8000:8000 --env-file .env ai-customer-intelligence:latest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Churn probability |
| `/explain` | POST | SHAP/local explainability |
| `/anomaly` | POST | Usage anomaly score |
| `/insight` | POST | LLM risk explanation & retention strategy |

## Environment Variables

- `DATABASE_URL` — PostgreSQL connection (optional; falls back to SQLite)
- `MLFLOW_TRACKING_URI` — MLflow server (default: `./mlruns`)
- `OPENAI_API_KEY` — For LLM insights (optional; uses mock if unset)
- `LOG_LEVEL` — DEBUG, INFO, WARNING, ERROR

## Project Structure

```
ai-customer-intelligence-platform/
├── api/                 # FastAPI service
├── config/              # Settings
├── dashboard/           # Streamlit admin UI
├── src/
│   ├── data/            # ETL, data generation
│   ├── features/        # Feature engineering, feature store
│   ├── models/          # Churn, anomaly, explainability
│   ├── training/        # Training scripts, bias tests, MLflow
│   └── monitoring/      # Drift (Evidently), logging
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/   # CI/CD
```

## Optional: Smaller sample for quick runs

```bash
SAMPLE_CUSTOMERS=500 python -m src.data.generate_data
```

## Cloud deployment

- **GCP Cloud Run**: `gcloud run deploy --source . --region us-central1`
- **Railway**: Connect repo and set root to this directory; add build command `pip install -r requirements.txt` and start `gunicorn api.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT`
- **AWS ECS**: Use the provided Dockerfile in your task definition.

## License

MIT.

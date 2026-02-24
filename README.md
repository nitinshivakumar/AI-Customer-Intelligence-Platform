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

## Quick Start

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

# AI Customer Intelligence Platform
# Usage: make install, make data, make train, make api, make dashboard

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.data.generate_data
	$(PYTHON) -m src.data.etl_pipeline
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.training.train_churn --model xgboost
	$(PYTHON) -m src.training.train_anomaly
	$(PYTHON) -m src.training.run_bias_tests

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

docker-build:
	docker build -t ai-customer-intelligence:latest .

docker-run:
	docker run -p 8000:8000 -v $$(pwd)/data:/app/data -v $$(pwd)/models:/app/models ai-customer-intelligence:latest

.PHONY: install data train api dashboard test docker-build docker-run

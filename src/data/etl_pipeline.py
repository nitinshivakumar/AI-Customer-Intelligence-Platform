"""
ETL: load raw parquet → clean → join → write to processed DB (SQLite/PostgreSQL).
"""
from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings


def run_etl():
    raw_dir = settings.project_root / "data" / "raw"
    processed_dir = settings.project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    db_url = settings.database_url
    if db_url.startswith("sqlite"):
        # Ensure directory exists for SQLite
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)

    # Load raw
    customers = pd.read_parquet(raw_dir / "customers.parquet")
    transactions = pd.read_parquet(raw_dir / "transactions.parquet")
    sessions = pd.read_parquet(raw_dir / "sessions.parquet")
    tickets = pd.read_parquet(raw_dir / "support_tickets.parquet")
    churn = pd.read_parquet(raw_dir / "churn_labels.parquet")

    # Clean: drop nulls, dedupe
    customers = customers.dropna(subset=["customer_id", "age", "region"])
    customers = customers.drop_duplicates(subset=["customer_id"])
    transactions["tx_date"] = pd.to_datetime(transactions["tx_date"])
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    tickets["ticket_created"] = pd.to_datetime(tickets["ticket_created"])

    # Write to DB (overwrite tables for idempotency)
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    transactions.to_sql("transactions", engine, if_exists="replace", index=False)
    sessions.to_sql("sessions", engine, if_exists="replace", index=False)
    tickets.to_sql("support_tickets", engine, if_exists="replace", index=False)
    churn.to_sql("churn_labels", engine, if_exists="replace", index=False)

    # Also write single merged table for feature building (optional)
    merged = customers.merge(churn, on="customer_id", how="left")
    merged.to_sql("customers_with_churn", engine, if_exists="replace", index=False)

    print(f"ETL done. Data in {db_url}")
    return engine


if __name__ == "__main__":
    run_etl()

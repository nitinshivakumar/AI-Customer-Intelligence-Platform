"""
Generate synthetic customer and usage data for training and evaluation.
Simulates: demographics, transactions, sessions, support tickets, churn labels.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings


def generate_customers(n: int, seed: int) -> pd.DataFrame:
    """Generate customer demographic and signup data."""
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West", "Central"]
    genders = ["M", "F", "O"]
    age_min, age_max = 18, 75

    customer_id = [f"C{i:06d}" for i in range(n)]
    age = rng.integers(age_min, age_max + 1, size=n)
    gender = rng.choice(genders, size=n, p=[0.48, 0.48, 0.04])
    region = rng.choice(regions, size=n)
    signup_date = pd.to_datetime(
        rng.integers(0, 365 * 3, size=n), unit="D", origin="2020-01-01"
    )
    tenure_days = (pd.Timestamp.now() - signup_date).days

    return pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "region": region,
        "signup_date": signup_date,
        "tenure_days": np.maximum(tenure_days, 1),
    })


def generate_transactions(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate transaction history (amount, frequency)."""
    rng = np.random.default_rng(seed)
    n_cust = len(customers)
    # 3–60 transactions per customer
    n_tx = rng.integers(3, 61, size=n_cust).sum()
    customer_ids = rng.choice(customers["customer_id"], size=n_tx, replace=True)
    amount = np.round(rng.exponential(50, size=n_tx) + 10, 2)
    amount = np.clip(amount, 5, 2000)
    days_ago = rng.integers(0, 365 * 2, size=n_tx)
    tx_date = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit="D")

    return pd.DataFrame({
        "customer_id": customer_ids,
        "amount": amount,
        "tx_date": tx_date,
    })


def generate_sessions(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate app/website session counts and duration (last 90 days)."""
    rng = np.random.default_rng(seed)
    n_cust = len(customers)
    n_sessions = rng.integers(0, 100, size=n_cust).sum()
    customer_ids = rng.choice(customers["customer_id"], size=n_sessions, replace=True)
    duration_sec = rng.integers(60, 3600, size=n_sessions)
    days_ago = rng.integers(0, 90, size=n_sessions)
    session_date = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit="D")

    return pd.DataFrame({
        "customer_id": customer_ids,
        "duration_sec": duration_sec,
        "session_date": session_date,
    })


def generate_support_tickets(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate support ticket counts per customer (last 180 days)."""
    rng = np.random.default_rng(seed)
    n_cust = len(customers)
    # Sparse: only ~15% have tickets
    n_tickets = rng.poisson(0.2, size=n_cust).sum()
    if n_tickets == 0:
        n_tickets = 1
    customer_ids = rng.choice(customers["customer_id"], size=n_tickets, replace=True)
    days_ago = rng.integers(0, 180, size=n_tickets)
    created = pd.Timestamp.now() - pd.to_timedelta(days_ago, unit="D")

    return pd.DataFrame({
        "customer_id": customer_ids,
        "ticket_created": created,
    })


def generate_churn_labels(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Churn = 1 if customer is considered churned (e.g. no activity in 60 days
    or explicit churn flag). We simulate based on tenure and random noise.
    """
    rng = np.random.default_rng(seed)
    n = len(customers)
    # Base rate ~20% churn; higher for low tenure and certain regions
    base = 0.2
    tenure = customers["tenure_days"].values
    p = base + 0.3 * (1 - np.minimum(tenure / 365, 1))  # newer = more churn
    p = np.clip(p + rng.normal(0, 0.1, n), 0.05, 0.85)
    churn = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "customer_id": customers["customer_id"],
        "churn": churn,
    })


def main():
    n = getattr(settings, "sample_customers", 10_000)
    seed = getattr(settings, "seed", 42)
    raw_dir = settings.project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Generating customers...")
    customers = generate_customers(n, seed)
    customers.to_parquet(raw_dir / "customers.parquet", index=False)

    print("Generating transactions...")
    transactions = generate_transactions(customers, seed)
    transactions.to_parquet(raw_dir / "transactions.parquet", index=False)

    print("Generating sessions...")
    sessions = generate_sessions(customers, seed)
    sessions.to_parquet(raw_dir / "sessions.parquet", index=False)

    print("Generating support tickets...")
    tickets = generate_support_tickets(customers, seed)
    tickets.to_parquet(raw_dir / "support_tickets.parquet", index=False)

    print("Generating churn labels...")
    churn = generate_churn_labels(customers, seed)
    churn.to_parquet(raw_dir / "churn_labels.parquet", index=False)

    print(f"Done. Raw data in {raw_dir}")


if __name__ == "__main__":
    main()

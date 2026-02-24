"""
Feature engineering: RFM, rolling windows, time aggregates, categorical encoding.
Output: single feature matrix for churn model + usage stats for anomaly.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.features.feature_store import save_features, get_store_path


def load_from_db(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_sql("SELECT * FROM customers", engine)
    transactions = pd.read_sql("SELECT * FROM transactions", engine)
    sessions = pd.read_sql("SELECT * FROM sessions", engine)
    tickets = pd.read_sql("SELECT * FROM support_tickets", engine)
    churn = pd.read_sql("SELECT * FROM churn_labels", engine)
    transactions["tx_date"] = pd.to_datetime(transactions["tx_date"])
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    tickets["ticket_created"] = pd.to_datetime(tickets["ticket_created"])
    return customers, transactions, sessions, tickets, churn


def rfm_and_usage(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    sessions: pd.DataFrame,
    tickets: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute RFM and usage aggregates per customer."""
    ref = reference_date or pd.Timestamp.now()
    tx = transactions.copy()
    tx["tx_date"] = pd.to_datetime(tx["tx_date"])
    tx = tx[tx["tx_date"] <= ref]

    # Recency: days since last transaction
    last_tx = tx.groupby("customer_id")["tx_date"].max().reset_index()
    last_tx.columns = ["customer_id", "last_tx_date"]
    last_tx["recency_days"] = (ref - last_tx["last_tx_date"]).dt.days

    # Frequency & Monetary
    freq = tx.groupby("customer_id").size().reset_index(name="tx_frequency")
    monetary = tx.groupby("customer_id")["amount"].agg(["sum", "mean"]).reset_index()
    monetary.columns = ["customer_id", "total_amount", "avg_amount"]

    # Session stats (e.g. last 90 days)
    win = ref - pd.Timedelta(days=90)
    sess = sessions[sessions["session_date"] >= win].copy()
    sess_agg = sess.groupby("customer_id").agg(
        session_count=("duration_sec", "count"),
        total_duration_sec=("duration_sec", "sum"),
        avg_duration_sec=("duration_sec", "mean"),
    ).reset_index()

    # Support tickets (last 180 days)
    ticket_win = ref - pd.Timedelta(days=180)
    tick = tickets[tickets["ticket_created"] >= ticket_win].copy()
    ticket_count = tick.groupby("customer_id").size().reset_index(name="ticket_count")

    # Merge all
    out = customers[["customer_id", "age", "gender", "region", "tenure_days"]].copy()
    out = out.merge(last_tx[["customer_id", "recency_days"]], on="customer_id", how="left")
    out = out.merge(freq, on="customer_id", how="left")
    out = out.merge(monetary, on="customer_id", how="left")
    out = out.merge(sess_agg, on="customer_id", how="left")
    out = out.merge(ticket_count, on="customer_id", how="left")

    out["recency_days"] = out["recency_days"].fillna(999)
    out["tx_frequency"] = out["tx_frequency"].fillna(0)
    out["total_amount"] = out["total_amount"].fillna(0)
    out["avg_amount"] = out["avg_amount"].fillna(0)
    out["session_count"] = out["session_count"].fillna(0)
    out["total_duration_sec"] = out["total_duration_sec"].fillna(0)
    out["avg_duration_sec"] = out["avg_duration_sec"].fillna(0)
    out["ticket_count"] = out["ticket_count"].fillna(0)

    return out


def time_based_aggregates(
    transactions: pd.DataFrame,
    sessions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Rolling/window aggregates: last 7d, 30d, 90d activity."""
    ref = pd.Timestamp.now()
    cust_ids = customers["customer_id"].unique()

    tx = transactions.copy()
    tx["tx_date"] = pd.to_datetime(tx["tx_date"])
    sess = sessions.copy()
    sess["session_date"] = pd.to_datetime(sess["session_date"])

    def agg_in_window(tx_df, sess_df, days):
        win = ref - pd.Timedelta(days=days)
        t = tx_df[tx_df["tx_date"] >= win].groupby("customer_id").agg(
            tx_count=("amount", "count"),
            tx_sum=("amount", "sum"),
        ).reset_index()
        s = sess_df[sess_df["session_date"] >= win].groupby("customer_id").agg(
            session_count=("duration_sec", "count"),
        ).reset_index()
        t.columns = ["customer_id", f"tx_count_{days}d", f"tx_sum_{days}d"]
        s.columns = ["customer_id", f"session_count_{days}d"]
        return t, s

    t7, s7 = agg_in_window(tx, sess, 7)
    t30, s30 = agg_in_window(tx, sess, 30)
    t90, s90 = agg_in_window(tx, sess, 90)

    out = pd.DataFrame({"customer_id": cust_ids})
    for df in [t7, s7, t30, s30, t90, s90]:
        out = out.merge(df, on="customer_id", how="left")
    for c in out.columns:
        if c != "customer_id":
            out[c] = out[c].fillna(0)
    return out


def encode_categorical(df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label-encode categoricals; return mapping for inference."""
    df = df.copy()
    encodings = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encodings[col] = {i: str(v) for i, v in enumerate(le.classes_)}
    return df, encodings


def build_churn_feature_matrix(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    sessions: pd.DataFrame,
    tickets: pd.DataFrame,
    churn: pd.DataFrame,
) -> pd.DataFrame:
    """Single feature matrix for churn model (with target)."""
    base = rfm_and_usage(customers, transactions, sessions, tickets)
    time_agg = time_based_aggregates(transactions, sessions, customers)
    base = base.merge(time_agg, on="customer_id", how="left")
    base = base.merge(churn, on="customer_id", how="left")
    base["churn"] = base["churn"].fillna(0).astype(int)

    # Age groups for bias analysis (keep string copy for fairness eval)
    base["age_group"] = pd.cut(
        base["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    )
    base["age_group"] = base["age_group"].astype(str)
    base["gender_sensitive"] = base["gender"].astype(str)
    base["region_sensitive"] = base["region"].astype(str)
    base["age_group_sensitive"] = base["age_group"].astype(str)

    cat_cols = ["gender", "region", "age_group"]
    base, _ = encode_categorical(base, cat_cols)
    return base


def main():
    engine = create_engine(settings.database_url)
    customers, transactions, sessions, tickets, churn = load_from_db(engine)

    print("Building churn feature matrix...")
    feature_df = build_churn_feature_matrix(
        customers, transactions, sessions, tickets, churn
    )

    # Store feature names for model (exclude target, id, and sensitive _sensitive cols)
    target_col = "churn"
    id_col = "customer_id"
    sensitive_suffix = ["gender_sensitive", "region_sensitive", "age_group_sensitive"]
    feature_cols = [
        c for c in feature_df.columns
        if c not in (target_col, id_col) and c not in sensitive_suffix
    ]
    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "id_col": id_col,
    }
    save_features(feature_df, "churn_training", metadata=metadata)
    print(f"Saved churn features: {len(feature_df)} rows, {len(feature_cols)} features")

    # Usage-only matrix for anomaly (no target)
    usage_df = rfm_and_usage(customers, transactions, sessions, tickets)
    usage_df = usage_df.merge(
        time_based_aggregates(transactions, sessions, customers),
        on="customer_id",
        how="left",
    )
    usage_df["age_group"] = pd.cut(
        usage_df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)
    usage_df, _ = encode_categorical(
        usage_df,
        ["gender", "region", "age_group"],
    )
    usage_cols = [c for c in usage_df.columns if c != "customer_id"]
    save_features(
        usage_df,
        "usage_anomaly",
        metadata={"feature_cols": usage_cols, "id_col": "customer_id"},
    )
    print(f"Saved usage/anomaly features: {len(usage_df)} rows")


if __name__ == "__main__":
    main()

"""
Simple file-based feature store: save/load feature matrices with metadata.
Can be replaced by Feast or cloud feature store later.
"""
from pathlib import Path
import json
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings


def get_store_path() -> Path:
    path = settings.project_root / "data" / "feature_store"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_features(
    df: pd.DataFrame,
    name: str,
    metadata: dict | None = None,
    version: str = "latest",
) -> Path:
    """Save feature matrix and optional metadata."""
    store = get_store_path()
    name_dir = store / name
    name_dir.mkdir(parents=True, exist_ok=True)
    version_dir = name_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(version_dir / "features.parquet", index=False)
    meta = metadata or {}
    meta["rows"] = len(df)
    meta["columns"] = list(df.columns)
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return version_dir


def load_features(name: str, version: str = "latest") -> tuple[pd.DataFrame, dict]:
    """Load feature matrix and metadata."""
    store = get_store_path()
    version_dir = store / name / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Feature set {name}/{version} not found")
    df = pd.read_parquet(version_dir / "features.parquet")
    with open(version_dir / "metadata.json") as f:
        meta = json.load(f)
    return df, meta


def list_feature_sets() -> list[str]:
    """List available feature set names."""
    store = get_store_path()
    if not store.exists():
        return []
    return [d.name for d in store.iterdir() if d.is_dir()]

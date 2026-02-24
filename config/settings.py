"""Application settings loaded from environment."""
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings."""
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    feature_store_path: Path = Path("data/feature_store")
    models_dir: Path = Path("models")
    mlruns_dir: Path = Path("mlruns")

    # Data
    database_url: str = "sqlite:///data/customer.db"
    sample_customers: int = 10_000
    seed: int = 42

    # MLflow
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "customer-intelligence"

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_mock_if_no_key: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Bias
    sensitive_attributes: list[str] = ["age_group", "gender", "region"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = self.project_root / self.data_dir
        self.raw_data_dir = self.project_root / self.raw_data_dir
        self.processed_data_dir = self.project_root / self.processed_data_dir
        self.feature_store_path = self.project_root / self.feature_store_path
        self.models_dir = self.project_root / self.models_dir
        self.mlruns_dir = self.project_root / self.mlruns_dir
        if not self.mlflow_tracking_uri:
            self.mlflow_tracking_uri = str(self.mlruns_dir)

settings = Settings()

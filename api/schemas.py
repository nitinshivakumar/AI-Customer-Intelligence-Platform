"""Pydantic request/response schemas."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class PredictRequest(BaseModel):
    """Features for a single customer (keyed by feature name)."""
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(..., description="Feature name -> value")


class PredictResponse(BaseModel):
    churn_probability: float
    label: int = Field(..., description="1 if churn, 0 otherwise (threshold 0.5)")


class ExplainResponse(BaseModel):
    churn_probability: float
    contributions: dict[str, float] = Field(..., description="Feature -> SHAP contribution")


class AnomalyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(..., description="Usage feature name -> value")


class AnomalyResponse(BaseModel):
    anomaly_score: float = Field(..., description="Higher = more anomalous")
    is_anomaly: bool = Field(..., description="True if score above threshold")


class InsightRequest(BaseModel):
    customer_id: str = ""
    churn_probability: float = Field(..., ge=0, le=1)
    profile: dict[str, Any] = Field(default_factory=dict)


class InsightResponse(BaseModel):
    customer_id: str
    risk_summary: str
    retention_strategy: str
    model_used: str = ""
